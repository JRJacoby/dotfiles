---
name: sam3-bbox-tracking
description: Run SAM3's PVS (Promptable Visual Segmentation) tracker from a bbox-only prompt on long videos, with a Dash bbox-drawing widget and a per-bbox role-labeling convention. Use when you need to segment one or more objects in a video (or extract a static-object mask from a single frame) using SAM3 without text prompts.
---

# SAM3 bbox-tracking

Box-prompted, no-text PVS segmentation with SAM3, packaged so it scales to long
videos (memory-bounded streaming + sliding-window eviction) and supports
multi-object identity via per-bbox role labels.

## When to use

- You have a video and need to segment one or more objects, identified by
  drawing a rectangle around them on some frame.
- You want SAM3's tracker without its text / concept-detection head — the
  text-prompted path tends to spawn spurious tracks for incidental features
  (parts of the subject, arena fixtures, etc.).
- Your video is long (10k+ frames) and the stock
  `Sam3TrackerPredictor.init_state(video_path=...)` would OOM by preloading
  every frame into a single tensor.
- You need to assign a stable identity ("familiar"/"novel", "pig"/"trough",
  "object_a"/"object_b") to each tracked object, recoverable from the output.

## When NOT to use

- You want to track everything matching a semantic concept ("all the cars").
  Use SAM3's text-prompted path via HuggingFace `transformers.Sam3VideoModel`
  instead.
- Your video is short and fits in memory — the stock SAM3 tracker is fine.
- You don't have an interactive session to draw boxes. The widget assumes a
  user is in the loop.

## The two-phase pattern

```
user                       widget                       driver
────                       ──────                       ──────
"draw boxes on session" -> open Dash UI               ->
                           draw rectangles
                           label each with a role
                           save boxes.json           ->  read boxes.json
                                                          build SAM3 model
                                                          adopt streaming subclass
                                                          frame 0: prompt with each box
                                                          stream forward through video
                                                          evict old memory each step
                                                          write per-frame masks
```

The widget and driver communicate exclusively through `boxes.json` per video.

## Setup

### SAM3 install

The canonical editable install lives at `/n/groups/datta/john/repos/sam3/`
(checked out from `facebookresearch/sam3`). Install into your project's env:

```bash
uv pip install -e /n/groups/datta/john/repos/sam3/
```

The package's declared deps under-specify what's actually needed at runtime.
Install these manually in addition:

```bash
uv pip install einops decord pycocotools hydra-core omegaconf pandas scipy psutil
```

### Other Python deps

| package        | minimum version | role                                     |
|----------------|-----------------|------------------------------------------|
| torch          | 2.11+cu128      | sam3 backend                             |
| torchvision    | 0.20+           | image transforms                         |
| transformers   | 5.9+            | (only for the text-prompted alt path)    |
| opencv-python  | 4.10+           | video I/O, resize                        |
| h5py           | 3.16+           | per-frame mask storage (if used)         |
| dash           | 4.0+            | bbox widget UI                           |
| plotly         | 6.0+            | bbox widget canvas                       |
| numpy          | 1.24+           |                                          |

CUDA versions matter — torch must match the system CUDA. Don't co-install JAX
in the same env unless you've verified CUDA versions agree; they almost never
do. Make a dedicated SAM3 env if needed.

### Model checkpoint

`facebook/sam3` on HuggingFace Hub (gated — accept the license once per HF
account in the browser, then `huggingface-cli login`). First inference call
downloads `sam3.pt` (3.3 GB) to `~/.cache/huggingface/hub/`.

### GPU memory budget

- Model on GPU: ~3 GB
- Active streaming state with `keep=32`: ~2 GB
- Per-frame transient: ~6 MB at 1008² × 3 × bfloat16
- Working total: **~5 GB GPU, ~2–4 GB host**, flat over 18k+ frame videos

Throughput is ~6 fps on an A6000 (memory-bound forward pass).

## Streaming pattern

The stock `Sam3TrackerPredictor.init_state(video_path=...)` preloads every
frame into `state["images"]` as one tensor. For 18 000 frames at 1008² × 3 ×
bfloat16 that's ~220 GB — won't fit.

The fix is a method-only subclass that:

1. Replaces `init_state` with `init_stream_state(H, W)` — no `state["images"]`.
2. Adds `seed_features(state, frame_idx, image)` — writes one frame's backbone
   features into `state["cached_features"]` so the predictor's internal
   `_get_image_feature` hits the cache instead of indexing the absent
   `state["images"]`.
3. Adds `stream_step(state, frame_idx, image)` — replicates the per-frame body
   of `propagate_in_video`, returns `(frame_idx, obj_ids, low_res_masks,
   video_res_masks, object_score_logits)`.
4. Adds `evict(state, current_frame, keep=32)` — deletes non-conditioning
   memory older than `current_frame - keep`. Conditioning frames (where the
   user-prompted boxes live) are never touched.

The class is **methods-only, no instance state**, so it retrofits onto an
already-built predictor via `obj.__class__ = StreamingSam3Tracker`. This
matters because `build_sam3_video_model()` returns a wrapping model whose
`.tracker` is the predictor — you don't get to construct the tracker yourself.

### Why eviction is safe

SAM3's tracker reads non-conditioning memory only within a recent causal
window: `num_maskmem=7` spatial-memory frames back, `max_obj_ptrs_in_encoder=16`
object-pointer frames back. `keep=32` is 2× the longest read distance — safe.
The frame-0 conditioning entries (where the box prompts live) are needed
forever and are preserved.

### `apply_temporal_disambiguation=False` is mandatory

When building the model, pass `apply_temporal_disambiguation=False`. The
`True` setting activates `use_memory_selection=True`, which may read frames
older than the eviction cutoff and crash the streaming run. With `False` the
tracker uses the standard causal recent-window memory, which `keep=32` covers.

### Canonical setup

```python
from sam3.model_builder import build_sam3_video_model
from streaming_pvs import StreamingSam3Tracker

sam3_model = build_sam3_video_model(apply_temporal_disambiguation=False)
predictor = sam3_model.tracker
predictor.__class__ = StreamingSam3Tracker        # method-only mixin via __class__ swap
predictor.backbone = sam3_model.detector.backbone  # streaming code uses the detector's backbone
```

The `backbone =` line is required: `seed_features` calls `self.forward_image`
which uses `self.backbone`, but the tracker is built without it set.

## Box prompts: polarity + role

Two orthogonal axes per box:

- **`label`**: `"positive"` or `"negative"` — polarity. Positive boxes seed
  an object's mask; negative boxes refine it ("the object is in here, but
  not in there").
- **`role`**: arbitrary string — object identity. Multiple positive boxes
  with different roles = multiple objects tracked in parallel. Positive +
  negative boxes that share a role apply to the same `obj_id`.

The driver maps `{unique role}` → `{1, 2, 3, ...}` for `obj_id` assignments.
This is the only thing the consumer needs to do to track multiple objects: a
positive box per object, each with a distinct role string.

### boxes.json schema

```json
{
  "session": "<session_id>",
  "video": "<relative_or_absolute_path_to_video>",
  "frame_idx": 0,
  "boxes": [
    {"label": "positive", "role": "<role_a>", "xyxy": [x0, y0, x1, y1]},
    {"label": "negative", "role": "<role_a>", "xyxy": [x0, y0, x1, y1]},
    {"label": "positive", "role": "<role_b>", "xyxy": [x0, y0, x1, y1]}
  ]
}
```

- `xyxy` is `[x_min, y_min, x_max, y_max]` in **native video pixels**, integer.
  When passed to `add_new_points_or_box`, divide by `(W, H)` to get the
  normalized `[0,1]` coords the API expects (with `rel_coordinates=True`,
  the default).
- `frame_idx` is informational — the frame the user was viewing when they
  drew. The driver conventionally prompts at frame 0; if the consumer needs
  to prompt elsewhere, that's a driver-side change.

### Multi-session widget

The widget supports a `--sessions` argument with multiple session dirs (one
video per session, or a session structure with multiple sub-videos). Switching
sessions in the UI does not lose pending edits — state lives in a
`dcc.Store`. Save writes only the currently-selected session's slice to disk.

## Driver pattern (skeleton)

```python
sam3_model = build_sam3_video_model(apply_temporal_disambiguation=False)
predictor = sam3_model.tracker
predictor.__class__ = StreamingSam3Tracker
predictor.backbone = sam3_model.detector.backbone

state = predictor.init_stream_state(H, W)
cap = cv2.VideoCapture(str(video_path))
ok, bgr = cap.read(); assert ok
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Frame 0: seed cache, then prompt every positive box.
img0 = predictor.preprocess_frame(rgb)
predictor.seed_features(state, 0, img0)

role_to_obj_id = {}            # stable mapping
for box in boxes:
    role = box["role"]
    if role not in role_to_obj_id:
        role_to_obj_id[role] = len(role_to_obj_id) + 1
    obj_id = role_to_obj_id[role]
    rel = np.array([[box["xyxy"][0]/W, box["xyxy"][1]/H,
                     box["xyxy"][2]/W, box["xyxy"][3]/H]], dtype=np.float32)
    # SAM3 distinguishes label via points api; box-only calls treat each as
    # positive. To use polarity, pass via `points`+`labels` instead. See the
    # template driver for the full implementation.
    res0 = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=obj_id, box=rel,
    )

predictor.propagate_in_video_preflight(state)
# res0[3] is now (n_obj, 1, H, W) at native resolution; first-frame masks.

f = 1
while True:
    ok, bgr = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = predictor.preprocess_frame(rgb)
    _, obj_ids, _, video_res, scores = predictor.stream_step(state, f, img)
    # video_res is (n_obj, 1, H, W) at native resolution.
    # Save / aggregate per-frame masks here.
    predictor.evict(state, f, keep=32)
    f += 1
cap.release()
```

## Templates

The `templates/` directory contains reference implementations:

- **`streaming_pvs.py`** — the `StreamingSam3Tracker` class. Copy verbatim;
  no project-specific changes needed.
- **`bbox_widget.py`** — Dash + Plotly bbox-drawing UI. Generic over sessions,
  with multi-bbox support and per-box `(label, role)` editing. Adapt the
  session-discovery callable for your project's directory layout.
- **`run_sam3_pvs.py`** — the multi-object streaming driver template. Handles
  role → obj_id mapping, multi-bbox prompting, per-frame mask emission.
  Adapt the I/O glue (output formats, where masks land) for your needs.

## Gotchas

- **`predictor.backbone = sam3_model.detector.backbone`** is required after
  the class swap. Without it `seed_features` errors with
  `AttributeError: 'StreamingSam3Tracker' object has no attribute 'backbone'`.
- **`apply_temporal_disambiguation=False`** is mandatory. With `True`, the
  tracker may read evicted frames and crash mid-stream.
- **Box coords are normalized [0,1]** when passed to `add_new_points_or_box`.
  Pass absolute pixel xyxy and you'll get an off-screen prompt at the model's
  internal 1008² resolution.
- **The image canvas in the widget must use `xaxis.range=[0, W]` and
  `yaxis.range=[H, 0]`** (y-inverted). Plotly then reports drawn-shape coords
  in native pixels.
- **The frame-0-only prompt assumes the object is visible at t=0.** For
  prompting on a later frame, the driver needs to seek to that frame, seed
  the cache there, prompt, then `propagate_in_video_preflight`, then stream
  forward from that frame. The streaming class supports it; the template
  driver doesn't, by default.
- **Streaming is forward-only.** No backward propagation. Frames before the
  prompt frame are not segmented.
- **No checkpoint/resume.** If a run is interrupted, restart from frame 0.
  The session state is purely in-process.
- **SAM3's tracker enforces non-overlapping masks across objects** via
  `non_overlap_masks_for_output=True` (default). If your objects touch or
  overlap in the image plane, the output mask edges may behave non-obviously
  — inspect the first session visually before scaling.

## Output formats (consumer's choice)

The driver returns per-frame `(n_obj, H, W)` masks plus per-object score
logits. How you persist these depends on downstream use:

- **Per-frame packed-bit HDF5** (pigseq's choice): one h5 file per video,
  group per `obj_id`, dataset per frame with packed bits. Compact, supports
  random access. Use for cases that need full-video tracking.
- **Single indexed PNG** (use for static objects with a single representative
  mask): take frame 0 (or a temporal consensus — mode/intersection across
  many frames) and write a `uint8` PNG where pixel value = obj_id, plus a
  sidecar JSON mapping `{1: role_a, 2: role_b, ...}`. Compact, single-file
  per video.
- **Multi-channel PNG**: one channel per role. Verbose, no sidecar needed.
- **Overlay MP4**: the rendered video with masks drawn on it. Always useful
  for visual QA — write this in addition to the structured output.
