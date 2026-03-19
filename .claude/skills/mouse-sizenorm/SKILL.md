---
name: mouse-sizenorm
description: Apply mouse size normalization to MoSeq2-extracted depth frames. Use when the user explicitly asks for size normalization, typically for developmental studies or when comparing across ages/sizes.
---

# Mouse Size Normalization

Size normalization transforms MoSeq2-extracted depth frames so that all mice
appear a canonical size, removing body size as a confound in behavioral analysis.
It uses a pre-trained neural network (autoencoder) to reconstruct each 80x80
depth frame at a standardized size.

**When to apply:** Only when the user explicitly requests it. Developmental
studies (comparing across ages) almost always use it. Multi-genotype or multi-sex
studies are case-by-case — ask before applying.

## Package

The `aging` package provides the size normalization model and inference code.

- **Repository:** https://github.com/dattalab/ontogeny-of-behavior
- **Local clone:** `/home/joj144/projects/keypoint-moseq/ontogeny-of-behavior/`
- **Package name:** `aging` (installed as `aging[torch]`)
- **Authors:** Winthrop Gillis, Dana Rubi Levy

### Installation

Requires Python 3.9+. Recommended to use a separate environment:

```bash
# From GitHub (no clone needed)
pip install "aging[torch] @ git+https://github.com/dattalab/ontogeny-of-behavior.git"

# From local clone
pip install -e /home/joj144/projects/keypoint-moseq/ontogeny-of-behavior[torch]
```

## Required Files

Two files are needed beyond the package itself:

| File | Path on cluster | Purpose |
|------|----------------|---------|
| `model.pt` | `/n/groups/datta/win/longtogeny/size_norm/models/freeze_decoder_00/stage_09/7b96ec7e-f894-4391-8c39-f0cb8d7dd516/model.pt` | Pre-trained TorchScript autoencoder. **Always use this model** — it is the current best ("new SNN"). An older model at `bottleneck_optimization_00/stage_06/` exists but over-smooths. |
| `median_template_poses.npy` | `/n/groups/datta/win/longtogeny/data/median_template_poses.npy` | Reference depth templates for computing a height offset that corrects for camera-to-floor distance variation across rigs. |

A zip of both files is also at:
`/n/groups/datta/john/projects/evan-schema/resources/2026-03-18_size_normalization/size_norm_resources.zip`

## How It Works

### Pipeline

```
Extracted frames (80x80 uint8, "frames" key in H5)
    |
    v
Height offset — compare session median frame to template poses,
    compute depth shift to match training baseline
    |
    v
Clip to 0-200mm range
    |
    v
Neural network inference — TorchScript autoencoder, batch_size=512
    |
    v
Clip to 0-255, round to uint8
    |
    v
Save as "size_norm_frames" dataset in same H5 file
```

### Height Offset

Different rigs may position the depth camera at slightly different heights above
the arena. The model was trained at a particular baseline depth. The height offset
corrects for this by:

1. Computing the session's median frame
2. Comparing it to each template pose in `median_template_poses.npy`
3. Taking the mean of the median depth differences
4. Adding this offset to all frames before inference

### Key Parameters

- `rescale` and `clean_noise` options exist in `predict_and_save()` but should
  be left at their defaults (`False`). We don't use them.
- `batch_size=512` for GPU inference
- GPU is strongly recommended; CPU works but is much slower

## Usage

### Using `predict_and_save` (simple case, on our cluster)

The `aging` package provides `predict_and_save` which handles everything, but it
has hardcoded cluster paths for `median_template_poses.npy`. Only works if running
on a machine with access to `/n/groups/datta/win/longtogeny/`:

```python
import torch
from aging.size_norm.apply import predict_and_save

model = torch.jit.load("/path/to/model.pt")
predict_and_save("session.h5", model, "size_norm_frames")
```

### Inlining the logic (portable, recommended for sharing)

When sharing with collaborators who don't have access to our cluster paths,
inline the height offset logic. See the tutorial script for a complete example:

`/n/groups/datta/john/projects/evan-schema/scripts/2026-03-18_size_normalization/apply_size_norm.py`

The key functions to inline are `compute_height_offset()` and `size_norm_session()`,
which load `model.pt` and `median_template_poses.npy` from configurable paths
instead of the hardcoded cluster paths in the `aging` package.

## Downstream MoSeq2 Integration

Size normalization creates a new HDF5 dataset (`size_norm_frames`) alongside the
original `frames`. Downstream MoSeq2 steps must be told which dataset to use:

| Step | Tool | Flag | Value |
|------|------|------|-------|
| Train PCA | moseq2-pca | `--h5-path` | `/size_norm_frames` |
| Apply PCA | moseq2-pca | `--h5-path` | `/size_norm_frames` |
| Compute changepoints | moseq2-pca | `--h5-path` | `/size_norm_frames` |
| Train model | moseq2-model | (none) | Uses PCA scores, already reflects normalization |
| Crowd movies | moseq2-viz | (none) | Use original `frames` for actual mouse appearance |
| Notebook GUI | moseq2-app | (none) | Use original `frames` |

Note: `--h5-path` needs a leading slash (HDF5 internal path).

## SLURM Considerations

- Size norm needs a GPU: use `gpu_quad,gpu` partition with `--gres=gpu:1` and
  `--qos=gpuquad_qos`
- 1 CPU, 5GB RAM, 3 hours is typically sufficient per session
- For batch processing, use SLURM array jobs with a worker script that takes
  an H5 path as a CLI argument (see `_size_norm_worker.py` in the evan-schema
  project for an example)

## Example Project

The SCHEMA MoSeq project has a complete working pipeline:

```
/n/groups/datta/john/projects/evan-schema/scripts/2026-03-18_size_normalization/
  apply_size_norm.py       # Tutorial/example script (portable, for sharing)
  _size_norm_worker.py     # SLURM GPU worker (takes H5 path as CLI arg)
  03_size_norm_sample.py   # Submit sample size norm jobs
  04_size_norm_all.py      # Submit all size norm jobs
  _shared.py               # Shared helpers (paths, SLURM submission)
```
