---
name: visualize-depth-data
description: Render depth/height frames (h5 or numpy arrays) as cubehelix MP4 videos. Use when the user asks to visualize depth data, render an h5 to video, or create a cubehelix MP4.
---

# Depth Data Visualization

Standard method for rendering depth/height frame sequences as cubehelix-colored MP4 videos. This is the convention used across the sidb-ratseq project and should be applied consistently whenever depth data needs visual output.

## The Recipe

### 1. Percentile normalization (p1/p99)

Sample ~200 evenly-spaced frames from the dataset. For each sampled frame, collect all **nonzero** pixel values. Concatenate and compute the 1st and 99th percentiles. These become the normalization bounds — they're robust to outliers and empty-frame artifacts.

```python
stride = max(1, N // 200)
samples = []
for i in range(0, N, stride):
    frame = dset[i]
    nz = frame[frame > 0]
    if nz.size:
        samples.append(nz)
pvs = np.concatenate(samples).astype(np.float32)
p1 = float(np.percentile(pvs, 1))
p99 = float(np.percentile(pvs, 99))
```

**Why nonzero**: zero pixels are background/outside-mask. Including them would pull p1 toward zero and waste contrast range on the background.

**Why p1/p99 not min/max**: outlier depth values (noise spikes, rare frames) would crush the useful contrast range into a narrow band.

### 2. Per-frame normalization + cubehelix LUT

```python
import matplotlib
cmap = matplotlib.colormaps["cubehelix"]
lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)  # (256, 3) RGB

denom = max(p99 - p1, 1.0)
for each frame:
    norm = np.clip((frame - p1) / denom, 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
```

**Why cubehelix**: it's perceptually uniform in luminance, prints well in grayscale, and distinguishes depth gradients better than viridis/plasma for smooth surfaces like animal bodies.

### 3. Upscale + ffmpeg pipe

```python
import cv2, subprocess

# Adaptive upscale: target smallest dimension == 600 px (fractional scale OK).
# Output stays readable on typical monitors without ffmpeg cost growing
# needlessly. If the source is already >= 600 on its smallest side, skip the
# resize entirely. libx264 also requires even W and H, so round up to even.
TARGET_MIN_DIM = 600
scale = max(1.0, TARGET_MIN_DIM / min(H, W))
up_h = ((round(H * scale) + 1) // 2) * 2
up_w = ((round(W * scale) + 1) // 2) * 2

if scale == 1.0 and (H % 2 == 0) and (W % 2 == 0):
    idx_up = idx
else:
    idx_up = cv2.resize(idx, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
frame_rgb = lut[idx_up]  # (up_h, up_w, 3) uint8 via LUT indexing
proc.stdin.write(frame_rgb.tobytes())
```

**Why bilinear (INTER_LINEAR)**: depth surfaces (animal bodies, objects, floor) are smooth, so bilinear gives a visually natural upscale at any fractional scale. With nearest-neighbor at non-integer scales, some source pixels duplicate into N+1 output pixels and others into N — a "wobbly grid" pattern that's invisible at small scales (1.04× from 576) but distracting at large scales (7.5× from 80×80). Bilinear removes the pattern entirely.

**When to override to INTER_NEAREST**: only if you're inspecting *pixel-level* artifacts (mask boundary correctness, single-pixel noise, single-frame anomalies). For those, the source pixel grid is the data you want to see, and bilinear blur destroys it. For everything else (general QA, draw-bboxes-on-this, "show me the recording"), bilinear is the better default.

**Why target 600 px on the smallest side**: 600 px is comfortable for human review on most monitors without ffmpeg encoding cost and file size growing needlessly. Examples:
- 640×576 raw Azure Kinect depth → smallest dim 576 → scale 1.042× → 667×600.
- 80×80 size-normed → smallest dim 80 → scale 7.5× → 600×600.
- 1920×1080 already-large render → smallest dim 1080 → scale 1.0× (no resize).

### Full ffmpeg invocation

```python
proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{up_w}x{up_h}", "-r", str(fps),
        "-pix_fmt", "rgb24", "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
```

`-crf 23` is a good default for QA videos — visually lossless at reasonable file sizes. Drop to 18 for publication-quality, raise to 28 for smaller files. `+faststart` moves the moov atom to the front of the file at encode-finish so browsers can stream and seek without downloading the whole file first.

### Typical parameters

| Parameter | Default | Notes |
|---|---|---|
| FPS | 30 | Matches Azure Kinect depth frame rate |
| Upscale target | smallest dim >= 600 px | `upscale = max(1, ceil(600 / min(H, W)))`; nearest-neighbor; no resize if already >=600 |
| Percentile sampling | 200 frames | Enough for stable estimates, avoids loading full dataset |
| CRF | 23 | Good QA quality. Lower = better quality, bigger file |
| Colormap | cubehelix | Project standard. Don't change without reason |

### Reading from h5

The depth data is typically stored as:
- `h5::frames` or `h5::clean` or `h5::recon` — the dataset name varies by pipeline stage
- dtype: uint16 (raw depth in mm) or float32 (cleaned/processed)
- Shape: `(N, H, W)` — N frames, H×W spatial

Stream frames one at a time from h5 to avoid loading the full dataset:
```python
with h5py.File(h5_path, "r") as f:
    dset = f["dataset_name"]
    N, H, W = dset.shape
    for i in range(N):
        frame = dset[i].astype(np.float32)
        # normalize, LUT, upscale, write...
```

## When to use

- Any time the user says "render", "visualize", "make a video of", or "show me" depth/height data
- When creating QA videos of pipeline outputs
- When writing a new script that produces depth h5 and needs a visualization step

## When NOT to use

- For non-depth data (pose tracks, scalar time series, etc.)
- When the user explicitly asks for a different colormap or normalization
