# Segmentation-mask cleaning

This is a post-detector analysis stage. It is not a VidSeq API operation and it
is not the cohort-level PCA or autoencoder used later for behavior modeling.
Run it only after segmentation inference is complete, score/mask coverage has
been validated, the user has accepted visual detector quality, and no source
HDF5 writer or sidecar lock remains.

## Canonical implementation

A prior validated 2D single-animal workflow used fixed-center 400x400 crops and
these scripts from the VidSeq example:

```text
/n/groups/datta/john/projects/vidseq/test_projects/
  test2dopenfielddatacropped/scripts/
    pca_reconstruct_detector_masks.py
    smooth_pca_recon.py
    clean_recon_masks.py
```

`test2dopenfielddatacropped` is only the historical example-directory name. It
does not mean this cleanup is limited to an open-field assay. Applicability is
set by the assumptions below, especially one centered subject in a common
subject-centric output grid. A centered, scale-normalized adaptive crop meets
that requirement even though its pixels do not represent fixed camera rays.

Inspect and fingerprint the current copies before every new analysis. They are
example scripts outside the HTTP API and can change independently of this
skill. Do not confuse them with VidSeq's `/pca` endpoint, which fits a different
downstream PCA to aligned masks.

The validated reference recipe is exactly:

```text
detector_masks.h5
  -> per-video PCA reconstruction
  -> detector_masks_pca_recon.h5
  -> temporal smoothing of PCA scores and reconstruction
  -> detector_masks_pca_recon_smoothed.h5
  -> strict threshold and centered connected-component selection
  -> detector_masks_pca_recon_smoothed_cleaned.h5
```

### 1. Per-video PCA reconstruction

For each video independently, the canonical script:

- reads `detector_masks.h5/data` with shape `(T, H, W)` and dtype `uint8`;
- flattens each frame to `(H*W,)` and converts chunks to `float32`;
- incrementally fits VidSeq `GPUPCA` in 2,000-frame batches;
- fits at most 200 components, a hard-coded module constant rather than a CLI
  option;
- chooses the smallest `k` whose cumulative explained variance reaches the
  requested target, 0.90 in the reference recipe, capped at the fitted 200
  components;
- projects and reconstructs in 1,000-frame chunks, clips to `[0, 255]`, casts
  to `uint8`, and writes a gzip-compressed HDF5 dataset named `data`.

It writes these sibling artifacts:

| Artifact | Contract |
|---|---|
| `detector_masks_pca_recon.h5` | `(T,H,W)`, `uint8`, grayscale-valued 0..255, dataset `/data` |
| `detector_pca.pkl` | Trusted pickle containing `pca`, retained `k`, and `variance_target` |
| `detector_pca_scree.png` | Cumulative explained-variance plot and selected `k` |

The source detector mask is `uint8`, but do not require its pixels to be only
0 and 255: resized detector edges can contain intermediate values. Likewise,
the reconstruction is continuous-valued in the image sense even though it is
quantized to `uint8`.

If the first 200 components do not reach the target, the canonical script uses
200 and prints a warning. Treat this as explicit QC, not silent success. In the
32-video reference run, 29 videos selected 200 and 27 remained slightly below
0.90, with cumulative explained variance approximately 0.8943--0.9002. Those
outputs passed visual and downstream QC, but that result is not a universal
acceptance rule. Inspect the scree plot and previews. To change the cap, use a
versioned project copy of the script, test it, and fingerprint it; do not claim
that `--max-components` exists.

### 2. Temporal smoothing in PCA-score space

The smoothing script reloads that video's PCA and retained `k`. It projects the
**original** `detector_masks.h5` again; it does not read or blur
`detector_masks_pca_recon.h5`. It forms a `(T, k)` score matrix and applies a
Gaussian filter along time. The reference invocation used:

```python
scipy.ndimage.gaussian_filter1d(
    scores, sigma=2.0, axis=0, mode="nearest"
)
```

It then reconstructs the smoothed scores, clips to `[0,255]`, casts to `uint8`,
and writes `detector_masks_pca_recon_smoothed.h5` with the same shape and HDF5
layout. The CLI expresses sigma in frames, but frame rate is not a workflow
requirement. Choose a temporal sigma on the order of 0.1 seconds or smaller and
compute `sigma_frames = sigma_seconds * actual_fps` for each video. SciPy accepts
a non-integer sigma. Record the seconds, FPS, and resulting frame value. Never
smooth across video/session boundaries.

This score-space smoothing makes the trajectory through mask-shape space
smooth. It is not a Gaussian blur over each image and it is not pixelwise
temporal filtering.

### Choose the visibility and gap policy

The unchanged canonical scripts fit all frames and smooth one continuous score
array per video. That is correct only when the subject is continuously visible
and accepted masks are complete. Choose and record one of these paths per video
or labeled interval. None is a global default, and a policy selected for one
project must not be carried into another:

#### Continuously visible, complete masks

Use the canonical scripts unchanged: fit every accepted frame and smooth one
continuous score sequence. This is the validated reference path.

#### Continuously visible, short detector failures

Do not mistake an empty detector result for absence. Use a versioned, tested,
fingerprinted project-local adaptation that:

1. fits PCA on accepted nonempty masks from reliable frames;
2. projects those reliable masks into PCA-score space;
3. linearly interpolates each score across only audited failure gaps no longer
   than a declared maximum and bounded by reliable frames on both sides;
4. applies `gaussian_filter1d` to the completed continuous score sequence.

Do not extrapolate leading/trailing failures or bridge a longer/unresolved gap
without a separate declared and validated policy.

#### Genuine intermittent visibility

Use a versioned, tested, fingerprinted project-local adaptation that consumes a
visibility/validity manifest and:

1. fits PCA only on frames declared both subject-visible and crop-valid;
2. requires those frames to have accepted nonempty masks, otherwise marks them
   invalid for audit rather than silently calling them absent;
3. forms maximal contiguous visible/valid runs and applies
   `gaussian_filter1d(..., mode="nearest")` independently within each run;
4. retains full-length output arrays and original frame indices but writes exact
   zeros for frames independently established as absent;
5. does not bridge absence, unknown state, or invalid frames.

Exact-empty output is specific to frames independently established as absent
in this branch. It is not a cleanup rule for an always-visible project; there,
every empty cleaned frame is a detector/reconstruction problem to investigate.

#### Mixed or unknown state

Label the intervals and apply the matching path to each. If the evidence cannot
distinguish failure from absence, keep the state unknown and break the run until
a project-specific fallback is declared; do not silently choose either branch.

For adapted paths, record the state-manifest path/hash, fit-frame count, run or
interpolation bounds, maximum interpolated gap, and excluded-frame counts by
state. If too few reliable frames remain to fit PCA, fail explicitly or use a
declared tested fallback.

#### Crop-valid interpolated detector failures

When the upstream crop trajectory bridges an audited detector-failure gap, the
detector's semantic state and the crop's validity are intentionally different.
Retain `detector_failure` in the state array and require separate
`interpolated_detector_failure` and `has_crop_transform` arrays with this
contract:

```text
has_crop_transform = (status == visible) OR interpolated_detector_failure
interpolated_detector_failure implies status == detector_failure
```

For mask cleaning, use the validated transform-bearing mask as the
subject-derived/crop-valid domain. A bridged failure with an accepted mask may
join its flanking visible run for PCA projection and temporal smoothing even
though its detector status remains `detector_failure`. An unbridged failure has
no transform, is excluded and breaks the run; known absence also has no
transform and retains exact-empty output. Do not collapse those two semantic
states merely because both lack a crop. Fail on any trajectory that violates
the flag/status/transform relationships rather than guessing which field is
authoritative.

### 3. Re-binarization and component selection

For each smoothed reconstructed frame, the cleaner:

1. marks foreground using the strict comparison `value > 127`;
2. finds 8-connected components with OpenCV;
3. defines frame center as `((W-1)/2, (H-1)/2)`;
4. if one or more component bounding boxes pass the script's literal inclusive
   test `x <= cx <= x+w` and `y <= cy <= y+h`, retains the largest of those by
   pixel area;
5. otherwise retains the component whose centroid is nearest the center;
6. leaves an empty thresholded frame empty.

The output is `detector_masks_pca_recon_smoothed_cleaned.h5`, dataset `/data`,
shape `(T,H,W)`, dtype `uint8`, with values exactly `{0,255}`. This is a
center-aware rule, not a global-largest-component rule. It assumes the animal
is centered in the common output grid. This includes a centered,
scale-normalized adaptive crop. If the crop can be off-center or contains
multiple animals, define and validate a different tracking rule rather than
applying this one blindly.

There is no erosion, dilation, hole filling, or spatial Gaussian blur in this
cleanup. A later modeling render used in the reference workflow—crisp pixels
inside the true mask, an 11x11 sigma-2 blurred image in a four-pixel dilation
ring, and zeros beyond—is a separate post-alignment appearance-rendering step.

## Safe staged execution

The canonical scripts unlink an existing derived output of the same name and
write the replacement directly. They are not restart-atomic. They also report a
missing per-video input as `SKIPPED` while the overall command can still exit
zero. Therefore, do not run them directly over an accepted result directory and
do not equate a zero exit code with completion.

Use a private stage for one manifest row/video. The stage may expose the
completed detector mask as a read-only-intent symlink while all derived files
remain local to the stage. The following is the minimum fail-closed shell
skeleton for the three canonical commands; a project-local runner must add the
full content verifier, provenance record, and atomic publication described
below:

```bash
set -euo pipefail
umask 0027

VIDSEQ_ROOT=/n/groups/datta/john/projects/vidseq
CANONICAL="$VIDSEQ_ROOT/test_projects/test2dopenfielddatacropped/scripts"
VIDEO_ID=123
SOURCE_MASK=/absolute/seg-project/array_data/123/detector_masks.h5
WORK_ROOT=/absolute/versioned-work
: "${ATTEMPT_ID:?set ATTEMPT_ID to a unique job/attempt identifier}"
: "${SIGMA_FRAMES:?set to chosen sigma seconds multiplied by verified video FPS}"

SOURCE_MASK="$(readlink -f -- "$SOURCE_MASK")"
[[ -f "$SOURCE_MASK" ]]
[[ ! -e "${SOURCE_MASK}.lock" ]]

mkdir -p "$WORK_ROOT/locks" "$WORK_ROOT/staging" "$WORK_ROOT/final"
exec {video_lock_fd}>"$WORK_ROOT/locks/video_${VIDEO_ID}.lock"
flock -n "$video_lock_fd" || {
  echo "video $VIDEO_ID is already owned by another cleanup task" >&2
  exit 1
}

STAGE_ROOT="$WORK_ROOT/staging/video_${VIDEO_ID}.${ATTEMPT_ID}"
[[ ! -e "$STAGE_ROOT" ]]
mkdir "$STAGE_ROOT"
STAGE_PROJECT="$STAGE_ROOT/project"
SESSION_DIR="$STAGE_PROJECT/array_data/$VIDEO_ID"
mkdir -p "$SESSION_DIR"
ln -s -- "$SOURCE_MASK" "$SESSION_DIR/detector_masks.h5"
[[ "$(readlink -f -- "$SESSION_DIR/detector_masks.h5")" == "$SOURCE_MASK" ]]

assert_source_ready() {
  [[ -f "$SOURCE_MASK" ]]
  [[ ! -e "${SOURCE_MASK}.lock" ]]
}

assert_source_ready
cd "$VIDSEQ_ROOT"
HDF5_USE_FILE_LOCKING=TRUE XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python "$CANONICAL/pca_reconstruct_detector_masks.py" \
    --project "$STAGE_PROJECT" --video-ids "$VIDEO_ID" --variance-target 0.90
assert_source_ready
[[ -s "$SESSION_DIR/detector_masks_pca_recon.h5" ]]
[[ -s "$SESSION_DIR/detector_pca.pkl" ]]
[[ -s "$SESSION_DIR/detector_pca_scree.png" ]]

HDF5_USE_FILE_LOCKING=TRUE XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python "$CANONICAL/smooth_pca_recon.py" \
    --project "$STAGE_PROJECT" --video-ids "$VIDEO_ID" --sigma "$SIGMA_FRAMES"
assert_source_ready
[[ -s "$SESSION_DIR/detector_masks_pca_recon_smoothed.h5" ]]

HDF5_USE_FILE_LOCKING=TRUE \
  uv run python "$CANONICAL/clean_recon_masks.py" \
    --project "$STAGE_PROJECT" --video-ids "$VIDEO_ID" --threshold 127 \
    --src-name detector_masks_pca_recon_smoothed.h5 \
    --out-name detector_masks_pca_recon_smoothed_cleaned.h5
assert_source_ready
[[ -s "$SESSION_DIR/detector_masks_pca_recon_smoothed_cleaned.h5" ]]
```

The cleaner's help text mentions a fallback source, but the current
implementation does not fall back: it skips the video if the requested source
is missing. Always pass both names explicitly and validate every expected file.
The canonical PCA and smoothing scripts have no state-manifest, missing-score
interpolation, or run-boundary arguments. For either adapted path above, replace
the first two commands with the project-local implementation; do not claim the
canonical commands implement that behavior.

Use one private stage and lock per video. Different videos may be restart-safe
array tasks when each task owns isolated output; do not let two tasks process
the same video/stage. PCA and score reconstruction use JAX/GPU. Load
`$using-slurm` before creating any O2 job. Prefer Ampere-or-newer when requested.
The component pass is CPU work but can remain in the same task.

Preflight storage. For a roughly 90,000-frame 400x400 reference video, each
continuous reconstruction was about 1.5--1.7 GB, the PCA pickle about 120--130
MB, and the cleaned mask about 90 MB: roughly 3.4 GB of derived files per video.
The four artifact classes occupied about 98 GiB across the 32-video reference
run. These are planning measurements, not guarantees.

After all validation succeeds, fingerprint the source one final time, remove
only the staged `detector_masks.h5` symlink, and write a completion record in
`SESSION_DIR`. Do not publish a live link to the mutable VidSeq source; retain
its resolved path and fingerprint in provenance. Atomically rename exactly that
symlink-free staged session directory to
`WORK_ROOT/final/VIDEO_ID` (or another manifest-declared versioned destination)
on the same filesystem. Require the destination not to exist. A project-local
runner should use `os.replace` only after confirming source/destination parent
device IDs match. The completion record should include:

- source project/video identity and source mask path, size, mtime, and SHA-256;
- exact parameters: variance target, hard cap, smoothing sigma in seconds,
  source FPS, resulting sigma in frames, threshold, connectivity, and
  center-selection policy;
- declared visibility/gap policy; state-manifest path and SHA-256 when used;
  state definitions, PCA-fit frame count, interpolated-gap or contiguous-run
  bounds, maximum interpolated gap, and excluded-frame counts by state;
- Python/environment identity plus paths and SHA-256 hashes of all scripts;
- output paths recorded relative to the session directory or as the intended
  final absolute paths—not staging absolute paths—plus sizes, mtimes, hashes,
  and HDF5 contracts;
- retained `k`, cumulative explained variance, cap/target warning, empty-frame
  count/rate, and mask-area quantiles;
- creation time and a schema version.

Do not publish a completion sentinel before the full audit. A preempted stage
without a valid sentinel is incomplete even when all filenames exist.

## Validation checklist

Fail closed on structural or provenance errors and make visual quality a user
checkpoint:

1. Before reading, prove segmentation apply is finished and no writer is alive.
   Check for a lock beside the resolved source HDF5 as well as the symlink path;
   never remove a lock merely to proceed.
2. Confirm the source has exactly the expected `/data` dataset, `uint8` dtype,
   frame count, height, and width. Match them to the video/manifest and detector
   coverage. Fingerprint the source before work and again before publication.
   A segmentation reapplication can leave an old positive mask/flag behind when
   the new result is negative; fail if any nonpositive-score row still has a
   detector mask, as required by the per-row consistency audit.
3. Confirm both reconstruction HDF5 files exist and have the same shape and
   dtype, finite/in-range sampled or scanned values, frame-sized chunks, and
   gzip compression.
4. Load `detector_pca.pkl` only when it came from the trusted staged run. Require
   finite explained-variance ratios and `1 <= k <= 200`. Require either target
   attainment or a recorded, reviewed cap warning.
5. Exhaustively scan the cleaned HDF5 in chunks. Require values only 0/255 and
   at most one 8-connected foreground component per nonempty frame. Record
   empty-frame count/rate and area p01/p50/p99 over frames expected visible. For
   a true-absence branch, require known-absent frames to be exactly empty. For an
   always-visible branch, investigate every empty cleaned frame. Investigate
   abrupt area or centroid changes and surprising outliers in either branch.
6. Render an overlay or masked preview spanning representative conditions,
   including beginnings/ends and detector hard cases. Check that reconstruction
   preserves the animal, temporal smoothing removes jitter without erasing real
   motion, and the component selector never jumps to debris.
7. Fingerprint outputs and scripts, write the completion record, atomically
   publish, then independently re-open and validate the published result.

The six-video labeling-subset audit in the reference workflow used an
empty-frame-rate failure threshold of 0.005, and the completed 32-video run
observed zero empty cleaned frames. Treat that as a dataset-specific precedent;
derive acceptance limits from the biology, visibility policy, and crop behavior
of the new project.
