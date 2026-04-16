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

up = H * UPSCALE  # typically UPSCALE = 3
idx_up = cv2.resize(idx, (up, up), interpolation=cv2.INTER_NEAREST)
frame_rgb = lut[idx_up]  # (up, up, 3) uint8 via LUT indexing
proc.stdin.write(frame_rgb.tobytes())
```

**Why nearest-neighbor**: preserves pixel-level detail without interpolation blur. At 3× the individual depth pixels are visible, which is useful for QA (you can see mask edges, noise, artifacts).

### Full ffmpeg invocation

```python
proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{up}x{up}", "-r", str(fps),
        "-pix_fmt", "rgb24", "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
```

`-crf 23` is a good default for QA videos — visually lossless at reasonable file sizes. Drop to 18 for publication-quality, raise to 28 for smaller files.

### Typical parameters

| Parameter | Default | Notes |
|---|---|---|
| FPS | 30 | Matches Azure Kinect depth frame rate |
| UPSCALE | 3 | 3× nearest-neighbor. Use 1 for native resolution |
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
