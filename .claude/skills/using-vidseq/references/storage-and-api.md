# VidSeq storage and API reference

This describes the VidSeq checkout examined in July 2026. Verify routes and
models in the current checkout after substantive updates.

## Global registry

VidSeq uses `platformdirs.user_data_dir("vidseq")/registry.db`, normally:

```text
~/.local/share/vidseq/registry.db
```

The `projects` table maps a process-local integer project ID to project name,
absolute project directory, and timestamps. Project IDs are registry IDs, not
portable identifiers. Each project has its own database.

Creating a project through the API creates the target directory, initializes
its database, and adds the registry record. Deleting through the project API
removes the registry record **and recursively deletes the project directory**.

## Project directory

A bbox/segmentation project normally contains:

```text
PROJECT/
├── .vidseqctl-apply.lock        # temporary helper advisory lock, only while applying
├── vidseq.db
├── array_data/
│   └── VIDEO_ID/
│       ├── tracker_masks.h5
│       ├── tracker_logits.h5
│       ├── detector_masks.h5
│       ├── final_masks.h5
│       ├── detector_masks_pca_recon.h5                  # optional derived analysis
│       ├── detector_masks_pca_recon_smoothed.h5         # optional derived analysis
│       ├── detector_masks_pca_recon_smoothed_cleaned.h5 # optional derived analysis
│       ├── detector_pca.pkl                             # optional derived analysis
│       └── detector_pca_scree.png                       # optional derived analysis
└── models/
    ├── detector.pt
    ├── detector_config.json
    ├── yolo_train/ or rtdetr_train/
    ├── seg_detector.pt
    └── seg_train/
```

The PCA reconstruction files shown above are optional external analysis
artifacts, not files created or managed by the VidSeq API. Later pipeline stages
can also add `cropped_masks.h5`, `aligned_masks.h5`,
`alignment_keypoints.h5`, `pca_scores.h5`, generated movies, and other model
artifacts. Keep native and derived provenance distinct.

`.vidseqctl-apply.lock` is not a VidSeq-native file. The bundled helper creates
it atomically across a serialized apply sequence and removes it on normal exit.
If a process crashes, inspect its JSON PID/host/start record and the backend job
before treating it as stale. The GUI does not honor this helper lock.

Video files themselves are not copied into the project. `videos.path` points to
the original absolute file.

## Per-project SQLite

`videos` stores:

- `id`, `name`, `path`;
- `fps`, `height`, `width`, `num_frames`;
- `segmentation_status` and confidence summaries;
- associated-video metadata.

`segmentation_status=NULL` does not mean detector application is incomplete.
Detector-only bbox/segmentation application updates detector fields/masks but
does not set that video status.

`frame_data` has one optional row per `(video_id, frame_idx)` and a unique index
on that pair. Important columns:

- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: generic/manual bbox fields. In the
  current tracker-mask training flow these can remain NULL; do not use them to
  decide whether training labels exist;
- `frame_type`: `train`, `apply`, or NULL;
- `score`: tracker/segmentation confidence; -1 means not computed;
- `detector_bbox_x1`, `detector_bbox_y1`, `detector_bbox_x2`,
  `detector_bbox_y2`: bbox detector or segmentation-detector result;
- `detector_score`: -1 means not processed, 0 means processed without a
  positive detection, and >0 means a positive detection;
- `has_tracker_mask`, `has_detector_mask`, `has_final_mask`: presence flags,
  not processed-frame counters;
- additional OBB and pose fields.

`conditioning_frames` records `(video_id, frame_idx)` frames with user prompts.
`pose_labels` stores later pose/flip labeling state.

Detector training gathers exactly the rows whose `frame_type='train'` and reads
their labels from `tracker_masks.h5`: bbox training computes a tight axis-aligned
box at training time, while segmentation training uses the largest-contour
polygon. The `bbox_*` columns are not the source of truth for this flow.

For a fresh bbox or segmentation project, use in-range frame counts plus
per-row consistency checks. Do not let out-of-range rows inflate coverage or
let aggregate bbox/mask totals hide mismatches that cancel each other:

```sql
SELECT v.id, v.name, v.num_frames,
  SUM(CASE WHEN f.frame_idx >= 0 AND f.frame_idx < v.num_frames
                 AND f.detector_score >= 0 THEN 1 ELSE 0 END) AS scored,
  SUM(CASE WHEN f.frame_idx >= 0 AND f.frame_idx < v.num_frames
                 AND f.detector_score > 0 THEN 1 ELSE 0 END) AS positive,
  SUM(CASE WHEN f.frame_idx >= 0 AND f.frame_idx < v.num_frames
                 AND f.detector_score > 0
                 AND (f.detector_bbox_x1 IS NULL OR f.detector_bbox_y1 IS NULL
                      OR f.detector_bbox_x2 IS NULL OR f.detector_bbox_y2 IS NULL)
           THEN 1 ELSE 0 END) AS positive_without_complete_bbox,
  SUM(CASE WHEN f.frame_idx >= 0 AND f.frame_idx < v.num_frames
                 AND f.detector_score <= 0
                 AND (f.detector_bbox_x1 IS NOT NULL OR f.detector_bbox_y1 IS NOT NULL
                      OR f.detector_bbox_x2 IS NOT NULL OR f.detector_bbox_y2 IS NOT NULL)
           THEN 1 ELSE 0 END) AS nonpositive_with_any_bbox,
  SUM(CASE WHEN (f.frame_idx < 0 OR f.frame_idx >= v.num_frames)
                 AND (f.detector_score >= 0 OR f.detector_bbox_x1 IS NOT NULL
                      OR f.detector_bbox_y1 IS NOT NULL OR f.detector_bbox_x2 IS NOT NULL
                      OR f.detector_bbox_y2 IS NOT NULL OR f.has_detector_mask=1)
           THEN 1 ELSE 0 END) AS active_out_of_range
FROM videos v
LEFT JOIN frame_data f ON f.video_id=v.id
GROUP BY v.id
ORDER BY v.id;
```

Compare `scored` to `num_frames` after the synchronous request returns and
require all mismatch/out-of-range counters to be zero. For segmentation also
require zero positive-score rows without `has_detector_mask=1` and zero
nonpositive-score rows with that flag. The helper implements these checks.
Bbox results and segmentation detector metadata do not accumulate in SQLite
within the request. Because bbox and seg detectors share detector fields, this
interpretation is cleanest when each stage uses its own fresh project.

## HDF5 arrays

Every HDF5 file uses dataset name `data`.

| File | Shape | dtype | Meaning |
|---|---|---|---|
| `tracker_masks.h5` | `(frames, height, width)` | `uint8` | User/SAM tracker masks |
| `tracker_logits.h5` | `(frames, 256, 256)` | `float32` | Low-resolution tracker logits |
| `detector_masks.h5` | `(frames, height, width)` | `uint8` | Segmentation-detector masks; gzip compressed |
| `final_masks.h5` | `(frames, height, width)` | `uint8` | Combined/final masks |

The four base arrays are created at video import. HDF5 allocation is sparse, so
an untouched large logical dataset can occupy very little disk. Neither file
existence nor logical shape proves that frames were populated.

VidSeq disables HDF5's own file locking and uses a write sidecar such as
`detector_masks.h5.lock`. Its normal read path does not reject that sidecar, but
external inspectors should: do not open a file while its sidecar exists. The
helper intentionally reports metadata/locks without importing `h5py`.
Internal masks are normally stored as `uint8` 0/255. VidSeq's detector-mask
export binarizes them to 0/1; do not assume an exported file has identical value
encoding to the project HDF5.

### Derived segmentation-cleaning arrays

The optional post-detector cleanup described in this skill writes these
sidecars. They all use dataset name `data`, retain the native
`(frames, height, width)` shape, use `uint8`, frame-sized chunks
`(1,height,width)`, and gzip compression:

| File | Values | Meaning |
|---|---|---|
| `detector_masks_pca_recon.h5` | 0..255 | Per-video PCA reconstruction before temporal smoothing |
| `detector_masks_pca_recon_smoothed.h5` | 0..255 | Reconstruction after Gaussian smoothing in PCA-score space |
| `detector_masks_pca_recon_smoothed_cleaned.h5` | exactly 0 or 255 | Strictly thresholded, single centered 8-connected component |

The first two are grayscale-valued reconstructions despite their `uint8` dtype.
The source `detector_masks.h5` can also contain intermediate edge values after
resizing, so do not require it to be exactly binary. Only the final cleaned
sidecar has the `{0,255}` contract.

The associated trusted `detector_pca.pkl` stores the per-video PCA object,
retained component count, and variance target; `detector_pca_scree.png` records
the selection visually. Pickle is executable serialization: load it only from
a trusted, fingerprinted pipeline run.

These sidecars have no API completion state. The canonical example scripts can
unlink/rewrite them and are non-atomic. Treat a versioned completion record plus
independent content validation as authoritative, not filename existence. Read
[mask-cleaning.md](mask-cleaning.md) for the exact lineage and safety rules.

## Core HTTP endpoints

All paths are below `/api`.

### Projects and videos

| Method | Endpoint | Body/result |
|---|---|---|
| GET | `/projects` | Registered projects |
| GET | `/projects/{pid}` | One registered project and its absolute directory |
| POST | `/projects` | `{"name": NAME, "path": PARENT}` |
| GET | `/projects/{pid}/videos` | Main-video metadata plus training-frame counts; excludes associated rows |
| POST | `/projects/{pid}/videos` | `{"paths": [ABSOLUTE_PATHS...]}` |
| DELETE | `/projects/{pid}` | Recursively deletes project; dangerous |

The add-video API also accepts a UTF-8 text file in `paths`; each nonempty line
must be an absolute movie path. The helper expands such lists itself to perform
deduplication against already imported videos.

Because the list endpoint filters out `is_associated=1`, use a read-only query of
the complete SQLite `videos` table when proving an exact project-wide count or
path bijection. Reconcile that all-row result with the API-visible main videos.

### Bbox detector

| Method | Endpoint | Meaning |
|---|---|---|
| GET | `/projects/{pid}/detection/status` | Model/training/type status |
| PUT | `/projects/{pid}/detection/config` | Set `yolo` or `rtdetr` |
| POST | `/projects/{pid}/detection/training` | Start background training |
| GET | `/projects/{pid}/detection/training` | Process-local progress |
| GET | `/projects/{pid}/detection/training/stream` | SSE progress |
| DELETE | `/projects/{pid}/detection/training` | Request graceful stop |
| POST | `/projects/{pid}/videos/detection` | Blocking full-video apply |

### Segmentation detector

| Method | Endpoint | Meaning |
|---|---|---|
| GET | `/projects/{pid}/detection/seg/status` | Model/training status |
| POST | `/projects/{pid}/detection/seg/training` | Start background training |
| GET | `/projects/{pid}/detection/seg/training` | Process-local progress |
| GET | `/projects/{pid}/detection/seg/training/stream` | SSE progress |
| DELETE | `/projects/{pid}/detection/seg/training` | Request graceful stop |
| POST | `/projects/{pid}/videos/seg-detection` | Blocking full-video apply |
| GET | `/projects/{pid}/videos/{vid}/seg-detector-masks/exists` | Any positive mask exists; not completion |

Training bodies use:

```json
{
  "video_ids": [1, 2],
  "max_epochs": 100,
  "batch_size": 4,
  "lr": 0.0001,
  "early_stop_patience": 20,
  "from_checkpoint": false
}
```

Both detector-apply bodies use `{"video_ids": [1, 2]}`. Training starts a
background thread and returns immediately. Full-video application holds the
HTTP request open until completion and has no status endpoint. Prefer one video
per request, serialized. Streamed worker progress is printed to backend stdout;
bbox output reaches SQLite at request end, while segmentation writes the active
HDF5 under its sidecar lock and commits database metadata at request end.
