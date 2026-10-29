# VidSeq operating workflow

## Launch topology

VidSeq has two development services:

- FastAPI backend: `uv run vidseq` from the repository, port 8000.
- Vite frontend: `npm run dev` from `frontend/`, port 5173.

The backend binds `0.0.0.0:8000`, has no authentication, and enables wildcard
CORS. Keep it behind the cluster network and an SSH tunnel; never expose it to
an untrusted network. The frontend sends requests to `/api`; Vite proxies those requests to
`http://localhost:8000`. If the backend is on an O2 GPU compute node and the
frontend/browser is on a GUI node, forward the compute node's port 8000 onto
the GUI node's localhost port 8000. Bind services only as broadly as the site's
supported tunnel requires. Do not publicly expose VidSeq.

For an O2 launch, always load `$using-slurm`, inspect current eligible GPU
resources, and obtain a non-preemptible allocation. A durable batch allocation
is appropriate when the frontend remains on the GUI node; an interactive shell
is also fine for active debugging. Prefer Ampere-or-newer for VidSeq inference.
Do not reuse a historical node list: derive currently eligible nodes from O2.
Start the backend inside the allocation from the VidSeq checkout with
unbuffered output and write stdout/stderr to a known log. Report the job ID,
node, expiry, GPU, log path, and exact tunnel/client commands to the user.

The backend command is:

```bash
cd /n/groups/datta/john/projects/vidseq
PYTHONUNBUFFERED=1 exec uv run vidseq
```

After the job is RUNNING, verify `/api/projects` directly on the compute node,
then keep this tunnel alive on the GUI node:

```bash
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:8000:127.0.0.1:8000 COMPUTE_NODE
```

In another durable GUI-node terminal:

```bash
cd /n/groups/datta/john/projects/vidseq/frontend
npm run dev
```

The frontend binds `0.0.0.0` with `allowedHosts: true` (`frontend/vite.config.ts`),
so it can run on any O2 node — a GUI node or a plain compute node — and be reached
from a browser elsewhere at `http://<frontend-node>:5173`. It always proxies `/api`
to its own `localhost:8000`, so run the SSH tunnel to the backend on whichever node
hosts the frontend, not necessarily the GUI node. (Node-to-node SSH within O2 works
directly, e.g. `ssh -fN -L 127.0.0.1:8000:localhost:8000 BACKEND_NODE` from the
frontend node.)

Verify all three layers before handing off:

```bash
curl -fsS http://127.0.0.1:8000/api/projects
curl -fsS http://127.0.0.1:5173/
curl -fsS http://127.0.0.1:5173/api/projects
```

Open `http://localhost:5173`. Follow live backend output with
`tail -n 100 -F /absolute/path/to/vidseq-server-JOBID.out`.

Do not use a preemptible partition for an interactive GUI session. It is useful
only for short, restart-safe, idempotent inference jobs.

## API root and health check

The backend root is normally `http://127.0.0.1:8000`; routes are under `/api`.
Keep the same `HOME`/`XDG_DATA_HOME` across backend allocations so they see the
same registry and worker coordination files.

```bash
curl -fsS http://127.0.0.1:8000/api/projects
```

SAM/GPU-worker readiness is separate from FastAPI readiness. The GUI normally
triggers `POST /api/segmentation/loaded-model`; poll
`GET /api/segmentation/status` until its `status` is `ready`. Detector apply
currently depends on that worker too. The worker records a node-local PID/port
in shared-home files under `~/.local/share/vidseq/`. After a dead allocation,
those files can be stale on the replacement node. Never remove them until the
old backend/worker is proven dead; do not confuse them with HDF5 lock files,
which must not be removed automatically.

The helper accepts a root URL with or without `/api`:

```bash
VIDSEQCTL="$HOME/.claude/skills/using-vidseq/scripts/vidseqctl.py"
python "$VIDSEQCTL" --base-url http://127.0.0.1:8000 list-projects
python "$VIDSEQCTL" worker-status
python "$VIDSEQCTL" load-worker --execute --wait
```

## Project setup

List the registry and check for name/path collisions:

```bash
python "$VIDSEQCTL" list-projects
python "$VIDSEQCTL" create-project \
  --name cohort-bbox --parent /absolute/parent
python "$VIDSEQCTL" create-project \
  --name cohort-bbox --parent /absolute/parent --execute
```

`--parent` must already exist. VidSeq creates `<parent>/<name>`, initializes
`vidseq.db`, and registers the new project in the user-level registry.
If creation errors after the directory is made, inspect both that target and
the registry before retrying; do not delete or create over a partial project.

There is no ordinary API route for registering an already-existing project
directory. If a target directory exists but is missing from the registry, stop
and diagnose rather than creating over it or editing `registry.db` casually.

## Video import

Create and preserve a cohort manifest first. Preview every import:

```bash
python "$VIDSEQCTL" add-videos --project-id 12 \
  --glob '/absolute/videos/*.mp4'
python "$VIDSEQCTL" add-videos --project-id 12 \
  --glob '/absolute/videos/*.mp4' --execute
python "$VIDSEQCTL" list-videos --project-id 12
python "$VIDSEQCTL" verify-import --project-id 12
```

Alternatively pass absolute paths as positionals or use `--path-list FILE`.
The helper expands list files, ignores blank/comment lines, resolves paths,
deduplicates against both the request and existing project videos, and then
calls the API. VidSeq validates metadata with OpenCV and creates array containers
after committing video records. That sequence is not atomic: a container failure
can leave the database row behind. Inspect `verify-import` before retrying, since
a blind retry can create a duplicate video row.

An HTTP timeout or lost client is also an ambiguous outcome, not evidence that
the backend stopped. Durable automation should write an exact request marker
before mutation and reconcile the registry, the complete SQLite `videos` table,
and HDF5 state afterward. Accept only an exact completed outcome. A partial or
wrong outcome is a manual stop; an unchanged outcome may be retried only after
the original backend request is proven dead and the operator explicitly
acknowledges that proof. Do not clear a marker merely because its client PID is
gone—the request may still be running remotely.

`verify-import` deliberately performs a shallow, lock-safe check (metadata,
presence, minimum container size, and no sidecar lock). After a partial failure
or when deeper validation is warranted, first confirm all locks are absent, then
verify the `data` dataset shapes/dtypes against `storage-and-api.md` using the
VidSeq environment. Never deep-open an actively locked HDF5 file.

Import order determines project-local video IDs. Always join later work by the
stored absolute path or a preserved manifest, not by assuming two projects use
the same IDs.

Before import, check free space and inodes/quota on the project filesystem.
Initial HDF5 containers are sparse, but tracker/detector masks and model runs can
grow substantially.

## Human bbox checkpoint

At this point the user works in the GUI:

1. Open the bbox project and confirm all videos render.
2. Create masks/training ranges across representative videos and conditions.
3. Train the bbox detector.
4. Apply it to a subset, visually inspect, add hard examples, and retrain.
5. Apply the accepted detector to all intended raw videos.

VidSeq derives bbox detector labels from user tracker masks on frames whose
`frame_data.frame_type` is `train`.

Codex can inspect readiness without HDF5:

```bash
python "$VIDSEQCTL" inspect-project --project-dir /absolute/cohort-bbox --summary
python "$VIDSEQCTL" detector-status --project-id 12 --kind bbox
```

If the user explicitly asks Codex to train after labeling:

```bash
python "$VIDSEQCTL" start-training --project-id 12 --kind bbox \
  --video-id LABELED_ID --video-id ANOTHER_LABELED_ID --detector-type yolo
python "$VIDSEQCTL" start-training --project-id 12 --kind bbox \
  --video-id LABELED_ID --video-id ANOTHER_LABELED_ID \
  --detector-type yolo --execute
python "$VIDSEQCTL" wait-training --project-id 12 --kind bbox
```

The API defaults are 100 epochs, batch size 4, learning rate 1e-4, and early
stop patience 20. Preserve the GUI/user's chosen settings when known.
Before retraining an accepted model, preserve its model/config/training metrics
under a versioned analysis artifact and record why it is being replaced.

For full application, the call blocks until the selected videos finish:

```bash
python "$VIDSEQCTL" apply --project-id 12 --kind bbox --all-videos
python "$VIDSEQCTL" apply --project-id 12 --kind bbox --all-videos --execute
```

The helper sends one blocking request per video in sequence and reports each
response. This creates durable per-video boundaries and avoids the worker's
one-hour apply timeout on a huge multi-video request. Monitor from another
process:

```bash
python "$VIDSEQCTL" wait-apply --project-dir /absolute/cohort-bbox --kind bbox
```

After every successful response, the helper performs a read-only local
database/HDF5-lock postcheck when the shared project path is accessible.
It also holds `<project>/.vidseqctl-apply.lock` across the sequence to exclude
other helper instances. A crash can leave this advisory lock stale: inspect its
recorded PID/host/job and the backend request before removing it. The GUI does
not honor the helper lock, so rule out a live GUI apply from the request/log.

The training progress object also reports automatic application back onto
training data, but it does not describe a separate full-video apply. Bbox
results are buffered and committed to SQLite only after the current request
finishes; they do not accumulate live.

Do not compare that automatic labeled-frame rate directly with cohort video
application. In the inspected implementation, the post-training loop groups
rows by video but still creates a fresh `cv2.VideoCapture`, performs a random
`CAP_PROP_POS_FRAMES` seek, reads one frame, closes the movie, and calls the
Ultralytics predictor at batch size one for every training row. Contiguous
annotation runs therefore discard decoder state. The full-video worker instead
keeps one `VideoFrameSource` open, reads sequentially, and predicts batches of
32. A low labeled-frame rate with an idle GPU can be an I/O/orchestration issue,
not evidence that the checkpoint is intrinsically slow. When this matters:

- measure wall time and progress on the exact phase;
- inspect GPU duty cycle rather than allocated memory alone;
- count movie opens, random-seek boundaries, and predictor calls;
- optimize by sorting explicit frame indices, reusing one reader per video,
  seeking only at discontinuous runs, and retaining source indices through
  batching;
- preserve one coherent result transaction and verify batch-1/batch-N box and
  score agreement before accepting a refactor.

Dataset materialization may contain the same per-frame open/seek pattern even
without inference. Similar throughput there is strong evidence that decoding,
not the model forward pass, dominates.

## Crop handoff

Read detector boxes from `frame_data.detector_bbox_*`, not the user-mask
`bbox_*` fields. Before deriving crops:

- establish processed-frame and positive-bbox coverage for every movie;
- inspect missed/zero-detection frames and bbox outliers;
- decide whether crop extent is fixed or adaptive;
- declare how true absence differs from detector failure;
- define edge padding deterministically;
- make every output exactly the same width and height;
- render a short preview and get user approval;
- verify output count, frame count, FPS, dimensions, and readable tail frames.

Treat crop extent, gap semantics, camera rectification, temporal rate, and
absent-frame rendering as explicit project policies. Do not copy a branch from
another cohort without checking its assumptions.

Persist a policy record before crop derivation. Use project-specific values,
but cover at least these keys so no choice is implied by the latest example:

```yaml
crop_extent: fixed | adaptive
output_size: [height, width]
visibility_model: always_visible | intermittent | mixed_or_unknown
bbox_gap_policy: bounded_interpolation | split_visible_runs | per_interval
centroid_smoothing: METHOD_AND_WINDOW
scale_smoothing: not_applicable | METHOD_AND_WINDOW
camera_coordinates: native | rectified
temporal_rate: full_rate | declared_subsample
missing_frame_rgb: interpolated_subject_crop | real_image_fallback | blank | excluded
mask_gap_policy: continuous | bounded_score_interpolation | visible_runs | per_interval
```

These are selection values, not universal defaults. Record parameters such as
maximum bridged gap, occupancy fraction, output size, and sampling rate beside
the selected value. A mixed project may need a frame/interval state manifest in
addition to this project-level record.

### Select fixed or adaptive crop extent

Before constructing either crop type, verify whether the detector artifact is a
true target track or merely a per-frame top-1 series. VidSeq's current
full-video YOLO bbox path takes `argmax(boxes.conf)` and persists only that one
box. When handlers, reflections, or other distractors receive the same learned
class, the top-1 box can switch away from the subject for many frames while the
correct box remains a lower-confidence candidate. Median/Gaussian smoothing of
the persisted top-1 coordinates cannot recover information that was discarded.
Inspect abrupt center/scale changes, re-run suspicious frames with all
candidates retained, and, when the phenotype is systematic, perform temporal
candidate selection for the full cohort before smoothing. Record the model,
candidate threshold, selection rule, and chosen candidate rank per frame.

Use a **fixed crop window** when subject scale is reasonably stable. Select one
square side from robust cohort-wide outer bbox-size bounds plus explicit
headroom, and smooth the detector centroid through time.

Use an **adaptive crop window** when perspective or subject motion causes large
apparent-size changes. Let \(\rho\) be the desired fraction of the output crop
occupied by the subject's largest bbox dimension. For a valid bbox of width
\(w_t\) and height \(h_t\), define the raw square side

\[
s_t^{\mathrm{raw}} = \frac{\max(w_t,h_t)}{\rho}.
\]

Apply declared robust lower/upper side limits, then smooth both the centroid and
\(\log s_t\) within contiguous valid-detection runs. Smoothing log side keeps
the scale positive and treats zoom changes multiplicatively. Crop a square of
the resulting side \(s_t\), with deterministic boundary padding, and resample
every frame to a common output size \(Q\times Q\). A \(400\times400\) output is
a useful precedent, not a universal default. Record \(\rho\), scale limits,
smoothing method/window, and interpolation.

Use a robust two-stage temporal filter when detector tracks can contain isolated
wrong-object boxes or large one-frame leaps:

1. apply a short odd-window median filter (a 5-frame window is a useful 30-Hz
   starting point) independently to center x, center y, and \(\log s_t\);
2. apply the declared Gaussian filter to those median-prefiltered signals.

The order matters. A Gaussian alone averages a large spike into several nearby
frames, so one bad detector box can bend an otherwise good crop trajectory. The
median rejects an isolated spike first, while the Gaussian still removes normal
frame-to-frame jitter afterward. Perform both stages independently inside each
contiguous subject-derived run. First interpolate only short gaps that the
project has explicitly classified as detector failures; then median-filter and
Gaussian-filter the joined run. Never pad either filter across a known-absence
boundary. Record the window, Gaussian sigma/truncation, run IDs, and raw inputs;
when practical, also record which frames the median stage changed.

Flag rather than trust boxes that touch an image boundary: a truncated subject
can make its bbox artificially small and cause the adaptive policy to zoom in
too aggressively. Use extra headroom, a scale floor, or downstream exclusion
according to the project's goal.

### Treat missing detections according to their meaning

Do not assume every gap should be interpolated.

- If the subject is known to be visible in every frame, do not create an
  `absent` state from an empty detector result. Treat audited short gaps as
  detector failures, interpolate the crop trajectory between reliable flanking
  detections up to a declared maximum duration, and audit the result visually.
  Longer or edge gaps need correction, exclusion, or another declared fallback;
  do not silently extrapolate them.
- If missing detections usually mean genuine out-of-frame absence, do not bridge
  them. End the current smoothing run, mark detection/visibility validity in the
  manifest, and restart smoothing at the next valid detection.
- If a project contains both cases, classify frames or intervals as
  `visible_valid`, `visible_detector_failure`, `known_absent`, or `unknown` and
  apply the corresponding branch locally. An empty result alone proves none of
  these states.
- If the output must preserve every input frame, use a declared actual-image
  fallback for absent frames, such as a fixed/full-field-of-view crop, and mark
  it invalid for subject-centered analyses. Use a blank placeholder only when
  no downstream consumer needs the RGB. Alternatively retain the frame mapping
  but exclude that frame from a later derived dataset.

Never silently carry, interpolate, or fabricate a subject-centered crop across
an interval known to represent absence. Distinguish `no detection`,
`subject absent`, and `detector failure/unknown` whenever the evidence permits.

Keep semantic state separate from crop validity. For an explicitly bridged
short detector-failure run, retain `detector_failure` as the semantic status and
write separate `interpolated_detector_failure` and `has_crop_transform` flags.
The required invariant is:

```text
has_crop_transform = visible OR validated_interpolated_detector_failure
```

An unbridged failure and a known-absence frame have no subject-centered crop
transform, but they remain different semantic states. Recheck the invariant
when importing the crop into a segmentation project and when constructing a
mask-cleaning visibility/run mask; downstream code must not substitute
`status == visible` for `has_crop_transform`.

### Calibrated multi-view projects

For a calibrated project whose downstream model assumes pinhole cameras,
consider rectifying before creating the segmentation crops:

1. choose and record one virtual rectified camera for each session/view;
2. transform detector centers and extents from the raw distorted image into the
   rectified image; because distortion is nonlinear, transform a box perimeter
   or sufficient samples rather than assuming a corner-only affine;
3. derive the smoothed crop in rectified coordinates;
4. serialize the rectified intrinsic matrix and every frame's crop affine;
5. preserve enough metadata to map crop pixels back through rectified and raw
   native coordinates.

For rectified crop origin \((x_{0t},y_{0t})\), square side \(s_t\), and output
side \(Q\),

\[
A_t =
\begin{bmatrix}
Q/s_t & 0 & -Qx_{0t}/s_t \\
0 & Q/s_t & -Qy_{0t}/s_t \\
0 & 0 & 1
\end{bmatrix},
\qquad K_{t,\mathrm{out}} = A_tK_{\mathrm{rect}}.
\]

Prefer one composite output-to-raw remap so rectification and crop/resize incur
one image interpolation pass rather than two. Use an interpolation appropriate
for RGB; masks later map with nearest-neighbor or a declared probability
resampling/threshold rule.

Rectification can introduce invalid borders or trade field of view for fewer
black pixels, and the derived project becomes dependent on the recorded
calibration. Preserve the untouched raw movies and complete transforms so this
does not become irreversible lock-in. Inspect edge-of-frame subjects before
choosing the virtual-camera field of view.

### Preserve full-rate alignment when downstream work needs it

Do not temporally subsample the crop or segmentation movies when downstream mask
smoothing, temporal reconstruction, tracking, or full-rate inference needs one
result per source frame. Preserve frame index, FPS, frame count, and an explicit
source-to-crop mapping even for absent/invalid frames. A later consumer may
materialize a sparse subset such as 1 Hz without discarding the full-rate
detector output.

A centered, scale-normalized adaptive crop deliberately defines a common
**subject-centric** output grid. Perform subject-shape PCA and temporal
PCA-score smoothing directly in that grid; changing camera rays do not make
this invalid. Preserve each frame's crop/rectification transforms and apply the
inverse only when a later operation needs camera-fixed rectified/native masks,
multi-view geometry, or reprojection into the source movie.

Keep this crop implementation in the analysis project, not in the reusable
VidSeq skill, because crop policy is cohort-specific.
Write to a new staged/versioned output directory, produce each movie via a
temporary path plus atomic rename, preserve a crop manifest, and never overwrite
an accepted crop set. At minimum record source project/video/frame identity,
raw bbox and score, visibility/validity state, raw and smoothed centroid/side,
rectification identity when applicable, crop origin/side, padding, output
dimensions, \(A_t\), \(K_{t,\mathrm{out}}\), codec/pixel format, and overwrite
policy.

## Fresh segmentation project

Create a distinct project and add the common-output-size crops:

```bash
python "$VIDSEQCTL" create-project \
  --name cohort-seg-crops --parent /absolute/parent --execute
python "$VIDSEQCTL" add-videos --project-id 13 \
  --glob '/absolute/crops/*.mp4' --execute
python "$VIDSEQCTL" list-videos --project-id 13
```

Reconcile this listing with the crop manifest before GUI work.

## Human segmentation checkpoint

The user now:

1. Creates tight tracker masks on representative cropped-video frames/ranges.
2. Marks training frames and trains the segmentation detector.
3. Applies it to a small held-out set and visually checks masks.
4. Adds hard examples and retrains until satisfied.
5. Authorizes cohort-wide application.

Segmentation training converts the tracker masks on `frame_type='train'` frames
to polygon labels. The model is stored separately from the bbox model.

Programmatic training, when requested:

```bash
python "$VIDSEQCTL" start-training --project-id 13 --kind seg \
  --video-id LABELED_ID --video-id ANOTHER_LABELED_ID --execute
python "$VIDSEQCTL" wait-training --project-id 13 --kind seg
```

Apply to selected held-out IDs first:

```bash
python "$VIDSEQCTL" apply --project-id 13 --kind seg \
  --video-id 4 --video-id 7 --execute
```

Then apply cohort-wide only after visual approval. Ordinary API application is
serial. Segmentation masks grow in the current video's HDF5 while its sidecar
lock is held, but detector database fields commit after the request. If runtime
motivates a job array, design isolated per-video outputs, atomic completion
sentinels, and a serialized database/HDF5 merge; do not point several workers
at the live project database.

When temporal mask smoothing or reconstruction needs full-rate data, require
processed-frame coverage for every source index rather than exporting only
positive masks or a temporal sample. Preserve the raw segmentation-detector
mask, score, and empty/no-detection state as the immutable first-stage result.
After all locks are absent and validation passes, create any smoothed masks or
reconstruction inputs as separate versioned analysis artifacts; never overwrite
VidSeq's detector output. Join them back to native/rectified coordinates through
the crop manifest rather than assuming the \(400\times400\) pixel system is
the original camera.

## Post-detector mask cleaning

After the user accepts segmentation quality and the full cohort is complete and
lock-free, the validated single-animal, centered-crop reference pipeline can
derive a cleaned mask lineage. Centered adaptive crops are valid because their
common output grid is intentionally subject-centric. This is not an API endpoint
and must not modify VidSeq's native `detector_masks.h5`:

```text
raw detector mask
  -> per-video PCA fit under the declared visibility/gap policy
  -> temporal Gaussian smoothing of retained PCA scores under that policy
  -> reconstruction and strict >127 re-binarization
  -> one centered 8-connected component
```

Choose one path per video or labeled interval:

- If the animal is known to be continuously visible and every accepted mask is
  present, use the canonical all-frame PCA and one continuous score smoother.
- If the animal is known to be visible across audited short detector failures,
  exclude failed masks from the fit, interpolate their PCA scores between
  reliable flanking frames up to a declared maximum gap, and then smooth.
- If the animal is genuinely absent, fit only visible/valid masks, smooth each
  contiguous visible run independently, and keep known-absent output empty.
- If state is mixed or unknown, label intervals and apply the corresponding
  path; never infer absence from an empty detector result alone.

The unchanged canonical scripts implement only the first path. The other paths
require a tested, fingerprinted project-local adaptation and a preserved state
manifest.

Run each video in an isolated, restart-safe stage and publish only after content
validation. The scripts can otherwise overwrite partial derived files or skip a
missing input without failing the process. See
[mask-cleaning.md](mask-cleaning.md) for the exact scripts, reference parameters,
component-selection rule, resource sizing, output contracts, and QC.

## Monitoring and recovery

- Training status is process-global and in memory. Restarting the backend can
  make it appear `idle`; inspect logs and model timestamps before deciding what
  happened.
- `wait-training` refuses an already-completed first observation by default,
  because it may be stale. Accept it explicitly only after checking the active
  model/log timestamps.
- Bbox model: `models/detector.pt` plus detector-specific training directory.
- Segmentation model: `models/seg_detector.pt` plus `models/seg_train/`.
- Full apply has no status endpoint. Tail backend lines such as
  `[Segmentation Service] Progress: n/total`; for segmentation, report the
  active HDF5 sidecar lock and size/mtime without opening it. The synchronous
  HTTP response is authoritative.
- After each request, count `frame_data` rows with `detector_score >= 0` for
  that video. Positive detections/masks are a separate quality metric.
- `wait-apply --kind bbox|seg` defaults to exact, in-range frame coverage and
  per-row bbox/mask consistency. Use `--allow-missing N` only after investigating
  a small codec/container discrepancy. The helper limits ordinary tolerance to
  a one-frame minimum for short videos, then at most 10 frames or 0.1%; a larger
  value requires `--allow-large-missing`.
- `wait-apply` has no run ID. On a rerun over already-complete results, it can
  only corroborate locks/mtime; the new synchronous response/log is authoritative.
- A long-running blocking API call belongs in a durable shell/log. If its client
  disconnects, confirm backend and database activity rather than immediately
  resubmitting.
- If an HDF5 `.lock` exists, inspect process/job state and logs. Never delete the
  lock while a writer may still be active.
- Sparse access logs are normal between request start and completion, while the
  worker's streamed progress is printed to backend stdout. Stable database
  counts during an active bbox request are expected, not evidence of a stall.
