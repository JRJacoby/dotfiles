---
name: using-vidseq
description: Set up, operate, inspect, and monitor VidSeq projects through its HTTP API and read-only on-disk state. Use for starting a VidSeq backend/client, creating a project, importing videos, handing off GUI annotation, training or applying bbox/segmentation detectors, monitoring progress, diagnosing project state, moving from a raw-video bbox project to a fresh common-output-size crop segmentation project, or cleaning accepted segmentation masks with visibility-aware per-video PCA reconstruction, temporal score smoothing, re-binarization, and connected-component selection. Covers the VidSeq registry, project SQLite database, per-video HDF5 arrays, model artifacts, derived mask artifacts, and safe O2 GPU operation.
---

# Using VidSeq

Use VidSeq's API for supported mutations and read-only SQLite/file metadata for
inspection. Keep the user's visual annotation and visual quality-control steps
as explicit handoffs.

## Fixed responsibilities

- The user reviews videos in the GUI, creates bbox training examples,
  iterates bbox training/application, creates tight segmentation masks in the
  cropped project, and visually judges detector quality.
- Codex may prepare projects and manifests, add videos, inspect labeled-frame
  counts, start training after the user says labels are ready, monitor work,
  derive the declared crop trajectory and common output size, create cropped
  movies, create the fresh segmentation project, add those movies, and—after
  the user accepts segmentation quality—create staged, validated mask-cleaning
  derivatives without changing VidSeq's detector output.
- Treat bbox, segmentation, and flip models as per-project. Do not silently
  reuse a model or project from another cohort.
- Use a fresh VidSeq project for the fixed-size cropped movies. Never replace
  the raw paths inside the bbox project.

## Guardrails

1. Never mutate `vidseq.db` or an HDF5 file directly during ordinary VidSeq
   operation. Use the API. Direct writes require a separately reviewed recovery
   or parallel-inference workflow.
2. Never call VidSeq's project DELETE endpoint without explicit confirmation:
   it recursively deletes the registered project directory.
3. Video deletion removes project data but leaves the source movie. It is still
   destructive to annotations and detector output, so require confirmation.
4. Source videos are referenced, not copied. Confirm their absolute paths will
   remain readable from the backend node.
5. Import preallocates HDF5 datasets. An existing `detector_masks.h5` therefore
   does **not** show that detector inference ran.
6. Do not open any HDF5 file while its sibling `.h5.lock` exists. Never remove a
   lock merely because it looks old; first prove that no writer is alive.
7. Read SQLite through `mode=ro` with a busy timeout. Do not copy a live database
   merely to avoid its normal locking behavior.
8. Do not run multiple ordinary API detector applications concurrently against
   one project. VidSeq shares one SQLite database and process-global training
   state. The helper uses an atomic project-local `.vidseqctl-apply.lock` to
   exclude other helper instances, but the GUI does not honor it and VidSeq has
   no apply-status API. Confirm no GUI/blocking request is active from the live
   request/log before executing. Parallel inference needs isolated artifacts
   plus a serialized merge.
9. Retraining replaces the project's active model artifacts, and reapplication
   replaces detector database/HDF5 results. Preserve an accepted model/run and
   require the user's intended scope before overwriting accepted outputs.
10. The canonical mask-cleaning scripts are external example scripts, not API
    operations. They unlink their own derived output names, write non-atomically,
    and can skip a missing input while exiting successfully. Run them only in a
    private per-video stage, validate content, and atomically publish a versioned
    result with a completion record. Keep `detector_masks.h5` immutable.
11. Do not interpret post-training labeled-frame throughput as full-video
    detector throughput. In the inspected VidSeq path, automatic application to
    training rows opens/seeks/closes the source movie and invokes the predictor
    once per frame, while full-video application reuses a sequential reader and
    batches 32 frames. Diagnose unexpected rates from the exact code path, GPU
    duty cycle, and decode pattern before resizing resources or changing models.
12. After moving a backend or replacing an SSH tunnel, do not trust an ambient
    `VIDSEQ_URL`, a numeric project ID alone, or a listening local port. Pin the
    intended base URL, prove the listener/tunnel targets the assigned backend
    node, and verify the API project ID, name, and resolved absolute path before
    any model or project mutation.

## Locate and check the installation

The working checkout has been `/n/groups/datta/john/projects/vidseq`, but verify
the current path and revision rather than assuming it forever. Inspect the live
route schemas before relying on undocumented behavior if the checkout changed.

From the VidSeq repository, the backend normally starts with:

```bash
uv run vidseq
```

It listens on port 8000 with no authentication and permissive CORS. The
development frontend normally starts from
`frontend/` with `npm run dev` and listens on port 5173. Its `/api` proxy expects
the backend at `localhost:8000`, so a split-node launch needs an SSH tunnel that
makes the GPU backend appear on port 8000 of the GUI/client node.
Never publish port 8000 to an untrusted network.

When O2 resources are requested, load and follow `$using-slurm` before issuing
any SLURM command. VidSeq inference should use an Ampere-or-newer GPU when the
user requests the fast interactive workflow. Do not reserve a GPU until the
user is ready. See [workflow.md](references/workflow.md) for the launch and
handoff sequence.

## Use the helper first

The bundled helper uses only the Python standard library:

```bash
VIDSEQCTL="$HOME/.claude/skills/using-vidseq/scripts/vidseqctl.py"
python "$VIDSEQCTL" --help
```

Its mutating commands preview by default and require `--execute`. Use it for:

- listing/creating projects;
- validating, deduplicating, and adding absolute video paths;
- verifying video rows and base array containers after import;
- loading/checking the segmentation GPU worker;
- listing videos and training-frame counts;
- starting bbox or segmentation training after the user's handoff;
- watching process-local training progress;
- applying a trained detector through the blocking API;
- inspecting a project without opening HDF5;
- watching per-video completion boundaries and read-only detector coverage.

Do not improvise manual registry, SQLite, or HDF5 writes if the helper lacks an
operation. Inspect the API and extend the helper or pause for review.
Use `inspect-project --summary` for a whole cohort; request full per-file output
only for selected video IDs.

## Declare project policies before deriving data

Do not turn a choice from one cohort into a universal rule. There is no default
visibility, gap, crop, camera, or frame-rate policy. Before deriving crops or
masks, inspect the data, ask when the evidence cannot decide, and record one
project-specific choice for every axis below. Never inherit the most recently
used branch merely because it appears in an example.

Use explicit frame states whenever gaps can have more than one meaning:

- `visible_valid`: use the observed detection/mask.
- `visible_detector_failure`: the subject is known to be present; this state is
  eligible for a declared, bounded interpolation or reconstruction policy.
- `known_absent`: the subject is independently known to be out of view; do not
  interpolate a subject trajectory or mask through it. Exact-empty masks are a
  requirement only for this branch when full-length mask arrays are retained.
- `unknown`: do not relabel it as absence or detector failure from an empty
  detector result alone. Break a run or use a separately declared fallback.

Record all of these decision axes in the crop/mask manifest:

- **Crop extent:** use a fixed square when apparent subject size is stable; use
  a per-frame adaptive square when it changes substantially. Smooth centroid in
  either path and positive crop scale in the adaptive path.
- **Detection gaps:** if the subject is known to remain visible and gaps are
  detector failures, interpolate only audited gaps up to a declared duration.
  If gaps are true absence, do not interpolate; split trajectories into visible
  runs. For mixed/unknown cases, classify frame state instead of choosing one
  policy globally.
- **Camera geometry:** crop directly for ordinary projects. For calibrated
  multi-view projects whose downstream consumer needs pinhole geometry,
  optionally rectify first and preserve intrinsics plus forward/inverse maps.
- **Temporal rate:** retain one output per source frame when smoothing,
  tracking, temporal reconstruction, or later full-rate use needs it. A declared
  temporal sample is acceptable only when every downstream consumer permits it.
- **Absent-frame image:** choose a real-image fallback, a blank placeholder, or
  exclusion from a derived dataset according to whether RGB and one-to-one
  alignment are needed. Always retain validity/frame mapping.
- **Mask coordinates:** clean shape in a centered common crop grid, fixed or
  adaptive. Map through per-frame transforms only for camera-fixed/native
  operations.
- **Mask gaps:** for an always-visible subject, treat missing masks as detector
  failures and interpolate audited short gaps in PCA-score space. For genuine
  absence, fit only visible/valid masks, smooth each visible run independently,
  and keep absent output empty. Unknown/invalid frames break a run until their
  state is resolved or an explicit fallback is declared.

The visibility choice controls both crop-trajectory and mask-score handling.
For an always-visible project, an empty detector result is a failure to repair,
not an absent frame to preserve. For an intermittent-visibility project, only
frames independently classified as absent receive the no-bridge/empty-mask
behavior. A mixed project applies these rules per interval.

## Workflow: bbox project through cleaned segmentation masks

### 1. Define the cohort and durable paths

Create a manifest of the intended raw videos before touching VidSeq. Record an
absolute path, session identity, expected count, and any cohort metadata needed
later. Check for duplicate resolved paths and basename collisions. Decide two
distinct project names, such as `<cohort>-bbox` and `<cohort>-seg-crops`.
Check free space, inode/quota headroom, and the durability of source/output paths
before import. Sparse base arrays are initially small but masks and models grow.

### 2. Start VidSeq and create the bbox project

Verify the backend with `list-projects`. Preview, then create the project under
an existing parent directory:

```bash
python "$VIDSEQCTL" create-project \
  --name COHORT-bbox --parent /absolute/project-parent
python "$VIDSEQCTL" create-project \
  --name COHORT-bbox --parent /absolute/project-parent --execute
```

The API's `path` field is the **parent** directory; VidSeq creates
`<parent>/<name>`. The target must not already exist.

Preview and import the raw movies:

```bash
python "$VIDSEQCTL" add-videos --project-id PROJECT_ID \
  --glob '/absolute/raw/path/*.mp4'
python "$VIDSEQCTL" add-videos --project-id PROJECT_ID \
  --glob '/absolute/raw/path/*.mp4' --execute
```

Re-list videos and reconcile IDs, paths, dimensions, FPS, frame counts, and
manifest count before handing the GUI to the user. Run `verify-import`; VidSeq
commits video rows before it creates HDF5 containers, so a failed import must be
inspected before retrying or it may create duplicate rows.

A client timeout, disconnect, or killed helper does **not** prove that a VidSeq
create/import request stopped on the backend. For a restartable automated setup,
publish a durable operation intent and per-request in-flight marker before each
mutation. After an ambiguous transport failure, reconcile the registry, complete
SQLite `videos` table, and HDF5 containers against the exact intended path. Reuse
an exact complete outcome; stop on a partial/wrong outcome; and do not resubmit an
unchanged outcome until the original backend request is independently proven dead
and that fact is explicitly acknowledged. Prefer an unbounded HTTP read timeout
inside a separately supervised durable process when the backend route itself is
blocking.

The project `/videos` endpoint lists main videos only and omits rows with
`is_associated=1`. For an exact all-row cohort or fresh-project proof, also query
the complete SQLite `videos` table read-only and reconcile every ID/path; an API
count alone can hide an extra associated row.

### 3. Hand off bbox annotation and visual iteration

Tell the user exactly which project is ready. The user will create training
examples, train, apply to a subset, inspect, add examples, retrain, and finally
apply to the cohort. Do not bypass this checkpoint.

During the handoff, monitor training through the bbox training endpoint. Run
full-video application as one blocking request per video, serialized. Bbox
results are committed only when that video's request finishes, so backend log
progress and the live HTTP request are the within-video signals; read-only
database coverage is a post-request check. A scored detector row has
`detector_score >= 0`; a positive detection has `detector_score > 0` and usually
a detector bbox. These are different counts.

### 4. Validate bbox output and create common-output-size crops

Require coverage for every intended video, inspect the small number of missing
or zero detections, and render at least one short crop preview. Choose the crop
strategy from the observed subject-size variation:

The current VidSeq full-video YOLO bbox application stores only the
highest-confidence candidate in each frame, even when Ultralytics produced
several boxes. Audit temporal center/scale jumps and known distractors before
treating that top-1 series as a track. A short median filter can repair isolated
top-1 flips, but it cannot recover a sustained wrong-candidate run because the
correct candidate was discarded from the stored result. For cohorts with
same-class distractors, retain/recompute all candidates and perform temporal
target selection before crop smoothing, or add the distractors as hard
negatives and retrain. Preserve the selected-candidate provenance.

- **Fixed window:** choose one square crop side for the cohort from robust outer
  bbox-size bounds with headroom; smooth the bbox centroid.
- **Adaptive window:** when apparent subject size changes substantially, choose
  a square crop side per frame so the subject occupies a declared fraction of
  the crop; smooth both centroid and positive crop scale temporally, then resize
  every crop to one common output size such as \(400\times400\).

For detector-derived crop tracks, prefer a short odd-window median prefilter
(often 5 frames at 30 Hz) on center coordinates and log crop side, followed by
the declared Gaussian smoother. The median stage removes isolated wrong-object
boxes or one-frame coordinate leaps without spreading their magnitude into
neighboring frames; the Gaussian stage then removes ordinary high-frequency
jitter and produces a smooth trajectory. Apply median then Gaussian separately
inside each contiguous subject-derived run, after any explicitly accepted
short-gap interpolation and never across known-absence boundaries. Record both
filter parameters and preserve raw boxes so the trajectory is reproducible.

Do not automatically interpolate missing detections. If missing detections
represent true out-of-frame absence, split smoothing into contiguous valid runs,
record absence/validity explicitly, and use a declared absent-frame output
policy. Interpolation is a separate option only when gaps are established
detector failures and the subject is expected to remain visible.

When an audited detector-failure gap is interpolated, keep its semantic state
as `detector_failure` and record separate `interpolated` and
`has_crop_transform` flags. Downstream consumers must not assume that
`status == visible` is identical to crop validity: a bridged failure can have a
real subject-derived crop even though the detector did not observe it. Define
crop-valid frames as explicit visible frames plus validated interpolated
failures (or equivalently from the validated transform flag); leave unbridged
failures without a transform and preserve known absence as a hard empty/run
boundary. Validate this contract again at the segmentation import and mask
cleaning boundaries.

For calibrated multi-view data, consider rectifying into a declared virtual
pinhole camera before cropping. Transform detector geometry into rectified
coordinates, serialize every frame's crop affine and adjusted intrinsics, and
preserve the inverse mapping. Prefer one composite raw-to-output remap to avoid
two image interpolation passes.

Record the exact occupancy fraction or size quantiles, scale bounds, median and
Gaussian smoothing parameters, gap policy, boundary padding, output codec, and final dimensions. Preserve one
output frame per input frame when later temporal mask smoothing, tracking, or
reconstruction needs full-rate alignment. Write crops into a new
staged/versioned directory, preserve a crop manifest, and publish atomically
after validation. Never overwrite a previously accepted crop set in place.

A prior validated single-animal fixed-center crop workflow used robust
1st/99th-percentile outer size bounds and a smoothed centroid, yielding 400x400
movies. That size is an example, not a universal constant.

### 5. Create a fresh cropped-video segmentation project

After all crop outputs pass count/dimension/frame checks, create the separate
segmentation project and add only the common-output-size cropped movies.
Reconcile its VidSeq video IDs to the crop manifest. Never infer that IDs match
the bbox project; IDs are local to each project and reflect import order.

### 6. Hand off tight-mask annotation

The user creates tight masks in the cropped project, selects training frames,
trains/applies on test videos, visually iterates, and says when training data or
full-cohort application is ready. Resume programmatic work only after that
handoff.

When requested, start segmentation training with selected video IDs and monitor
it. Then apply to a small out-of-training test set for the user's visual check
before cohort-wide inference.

### 7. Validate segmentation output

Use each blocking HTTP response as the authoritative completion signal, then
validate database score coverage. `has_detector_mask=1` counts positive masks
only; it is not a processed-frame counter. During segmentation inference, an
HDF5 sidecar lock identifies the active video, but do not open that HDF5. Inspect
content only after locks are absent. A backend restart resets the in-memory
training progress object to `idle`; model files and database/HDF5 output persist.

When downstream temporal mask smoothing or reconstruction requires full-rate
data, retain the unsmoothed detector result and score state at every source frame
as the immutable first-stage output. Write smoothing/reconstruction products as
separate versioned analysis artifacts after inference and lock-safe validation;
do not overwrite VidSeq's detector masks. A centered, scale-normalized adaptive
crop is a valid subject-centric coordinate system for mask-shape PCA and
temporal score smoothing: clean masks directly in the common output grid. Use
the saved per-frame inverse crop/rectification transforms only when a downstream
consumer needs masks in rectified/native camera coordinates.

### 8. Clean accepted segmentation masks

Only after segmentation coverage passes at the project's declared temporal
rate, the user accepts visual quality, and every `detector_masks.h5.lock` is
absent, create a derived mask lineage. Require full-rate coverage when the
declared downstream smoothing/reconstruction policy needs it. First choose the
per-video visibility/gap policy:

- **Always visible, complete masks:** fit PCA to all accepted masks and smooth
  the continuous score sequence. This is the canonical reference path.
- **Always visible, audited detector failures:** exclude bad masks from the PCA
  fit, project reliable masks, interpolate each PCA score only across declared
  short failure gaps with reliable frames on both sides, then smooth the
  completed sequence. This requires a tested project-local adaptation.
- **True intermittent visibility:** fit PCA only on declared visible/valid
  masks, smooth each contiguous visible/valid score run independently, and
  leave declared absent output empty. Do not bridge view exits and entries.
- **Mixed or unknown:** apply the matching branch per labeled interval. Do not
  infer absence merely from an empty detector mask.

For every branch, select the smallest retained dimension reaching 90%
cumulative explained variance subject to the canonical 200-component cap,
Gaussian-smooth scores with a sigma on the order of 0.1 seconds or smaller
converted using actual FPS, reconstruct, re-binarize with strict `value > 127`,
and retain one centered 8-connected component.

This cleanup PCA is **per video** and is distinct from any later cohort-level
PCA or autoencoder used for behavior modeling. Smoothing is in PCA-score space,
not pixel space. Use declared project knowledge and, when available, the crop
manifest's visibility/validity state. Treat an empty mask as a detector failure
when the animal is known to be visible, as true absence only when independently
established, and otherwise as unknown. The center-component rule is valid
whenever the output crop keeps the animal near the center, including centered
adaptive crops; visually test it before scaling out. Do not add erosion,
dilation, or the later fuzzy rendering ring to this cleaning stage.

If the crop lineage bridges audited detector failures, derive the cleaner's
subject-derived/crop-valid mask from the validated crop transform (explicit
visible **or** interpolated failure), not from detector status alone. Preserve
the original semantic status alongside that mask. A transform-bearing bridged
frame may participate in the same visible run when its segmentation mask is
accepted; an unbridged failure has no crop transform and must break or be
excluded from the run, while known absence remains exact-empty. Fail on
impossible combinations such as an interpolated flag without detector-failure
status, a bridged failure without a transform, or a non-bridged/non-visible
frame with a transform.

Read [mask-cleaning.md](references/mask-cleaning.md) before running it. That
reference gives the exact canonical scripts and arguments, staged-publication
pattern, HDF5 contracts, disk/GPU expectations, and fail-closed validation.

## Monitoring interpretation

- Training: poll `detection/training` for bbox or `detection/seg/training` for
  segmentation. `completed`, `failed`, and `stopped` are terminal. An unexpected
  `idle` after a restart is unknown, not proof of success.
- Training can remain `status=applying` after optimization ends while VidSeq
  reapplies the new model to labeled rows. This is part of the training request,
  not the separate full-video route, and can be much slower per frame when the
  labels are sparse. Wait for `completed` before archiving or restarting.
- Full-video apply: the HTTP call is blocking and no apply-status API exists.
  The helper serializes one request per video. Keep it in a durable terminal,
  tail the backend log, and use `wait-apply`/`inspect-project` for corroboration.
- Within one bbox apply, SQLite does not accumulate; results commit at request
  end. Within segmentation apply, the active HDF5 grows under a sidecar lock but
  detector database fields commit at request end. Never read the active HDF5.
- Post-request completion: compare rows with `detector_score >= 0` to the video's
  `num_frames`, restricted to valid frame indices. Require zero per-row
  score/bbox/mask mismatches and zero active out-of-range rows. A tiny shortfall
  can come from optimistic container frame counts; inspect backend logs and the
  tail frames before accepting it.
- Logs: useful for failures, within-video worker progress, and request lifetime;
  sparse access logs do not mean no work. SQLite counts are a post-request
  verification signal, not a within-request progress signal.
- HDF5 size/mtime and locks: useful as live corroboration for segmentation,
  never as the sole completion test. Old complete results make disk counters
  unable to distinguish a rerun; the new HTTP response/log is authoritative.

## References

- Read [workflow.md](references/workflow.md) for launch topology, concrete API
  commands, human checkpoints, training/application, and recovery rules.
- Read [storage-and-api.md](references/storage-and-api.md) before interpreting a
  project on disk, writing a monitor, or debugging apparent completion.
- Read [mask-cleaning.md](references/mask-cleaning.md) before deriving cleaned
  masks from accepted segmentation output or interpreting those sidecar files.
