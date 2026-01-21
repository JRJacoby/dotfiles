---
name: keypoint-moseq
description: Fit Keypoint-MoSeq models to animal pose tracking data for unsupervised behavioral segmentation. Use when working with SLEAP, DeepLabCut, or other keypoint tracking data to discover behavioral syllables.
---

# Keypoint-MoSeq

Keypoint-MoSeq is a machine learning method for unsupervised segmentation of animal behavior from pose tracking data. It identifies discrete behavioral "syllables" and their temporal structure using an autoregressive hidden Markov model (AR-HMM) on pose dynamics.

## Installation

**Important: Always use Python 3.10 for keypoint-moseq projects.** Later versions may have compatibility issues with JAX and other dependencies.

```bash
# Create environment with Python 3.10
uv init --python 3.10
# or
conda create -n kpms python=3.10

pip install keypoint-moseq
# or
uv add keypoint-moseq
```

Requires JAX. For GPU support, install JAX with CUDA first.

## Core Pipeline

The standard Keypoint-MoSeq workflow:

1. **Load keypoints** - Import from SLEAP, DeepLabCut, or other formats
2. **Project setup** - Create config.yml with bodyparts, skeleton, fps
3. **Outlier removal** - Identify and interpolate outlier keypoints
4. **Noise calibration** (optional) - Estimate observation noise from stationary frames
5. **Format data** - Batch recordings into segments for model fitting
6. **Fit PCA** - Reduce dimensionality of pose space
7. **Initialize model** - Set initial AR-HMM parameters
8. **Fit AR-HMM** - Fit autoregressive model (fast, ~minutes)
9. **Fit full model** - Joint fitting with keypoint observations (slow, ~hours)
10. **Reindex syllables** - Renumber so syllable 0 = most frequent (IMPORTANT!)
11. **Extract results** - Get syllable labels, centroids, headings

## Key Functions

### Project Setup

```python
import keypoint_moseq as kpms

kpms.setup_project(
    project_dir='project/',
    bodyparts=['nose', 'left_ear', 'right_ear', 'neck', 'tail_base'],
    skeleton=[['nose', 'neck'], ['neck', 'tail_base']],  # Optional connections
    fps=30,
    anterior_bodyparts=['nose'],      # For heading calculation
    posterior_bodyparts=['tail_base'],
    use_bodyparts=None,               # Subset to use (default: all)
    overwrite=False,
)
```

Creates `project/config.yml` with all parameters.

### Loading Keypoints

```python
# From SLEAP
coordinates, confidences, bodyparts = kpms.load_keypoints(
    filepath_pattern='predictions/*.slp',
    format='sleap',
)

# From DeepLabCut
coordinates, confidences, bodyparts = kpms.load_keypoints(
    filepath_pattern='videos/*DLC*.h5',
    format='deeplabcut',
)

# From NWB
coordinates, confidences, bodyparts = kpms.load_keypoints(
    filepath_pattern='data/*.nwb',
    format='nwb',
)
```

**Returns:**
- `coordinates`: dict mapping session_id to array (n_frames, n_keypoints, 2)
- `confidences`: dict mapping session_id to array (n_frames, n_keypoints)
- `bodyparts`: list of bodypart names

### Outlier Removal

```python
coordinates, confidences = kpms.outlier_removal(
    coordinates=coordinates,
    confidences=confidences,
    project_dir='project/',
    bodyparts=bodyparts,
    outlier_scale_factor=6.0,  # Higher = fewer outliers removed
    overwrite=False,
)
```

Identifies keypoints too far from the animal's medoid and interpolates them. Generates QA plots in `project/QA/plots/`.

### Noise Calibration (Optional)

```python
kpms.noise_calibration(
    project_dir='project/',
    coordinates=coordinates,
    confidences=confidences,
    bodyparts=bodyparts,
    # Automatically detects low-movement frames
)
```

Estimates observation noise from stationary periods. Can skip if using default noise estimates.

### Format Data

```python
data, metadata = kpms.format_data(
    coordinates=coordinates,
    confidences=confidences,
    bodyparts=bodyparts,
    use_bodyparts=None,  # Or subset list
    seg_length=10000,    # Segment length for batching
)
```

**Returns:**
- `data`: dict with keys 'Y' (coordinates), 'conf' (confidences), 'mask' (padding)
- `metadata`: tuple (keys, bounds) for mapping segments back to recordings

### Fit PCA

```python
pca = kpms.fit_pca(
    Y=data['Y'],
    mask=data['mask'],
    conf=data['conf'],
    anterior_idxs=config['anterior_idxs'],
    posterior_idxs=config['posterior_idxs'],
    conf_threshold=0.5,
    PCA_fitting_num_frames=1000000,
    verbose=True,
)

kpms.save_pca(pca, 'project/')
```

PCA is fit on egocentric-aligned, centered keypoints.

**Visualizations:**
```python
kpms.plot_scree(pca, project_dir='project/', savefig=True)
kpms.plot_pcs(pca, use_bodyparts=bodyparts, project_dir='project/', savefig=True)
```

### Initialize Model

```python
model = kpms.init_model(
    data=data,
    pca=pca,
    project_dir='project/',
)
```

### Fit AR-HMM

```python
model, history = kpms.fit_model(
    model=model,
    data=data,
    metadata=metadata,
    project_dir='project/',
    ar_only=True,           # Only fit AR-HMM (fast)
    num_iters=50,
)
```

Fast fitting (~minutes) using only latent pose dynamics.

### Fit Full Model

```python
model, history = kpms.fit_model(
    model=model,
    data=data,
    metadata=metadata,
    project_dir='project/',
    ar_only=False,          # Full model with keypoint observations
    num_iters=500,
)

kpms.save_checkpoint(model, 'project/')
```

Joint fitting (~hours) for final model. Checkpoints saved automatically.

### Extract Results

```python
results = kpms.extract_results(
    model=model,
    metadata=metadata,
    project_dir='project/',
)
```

**Returns dict with:**
- `syllables`: dict mapping session_id to syllable labels per frame
- `centroids`: dict mapping session_id to (n_frames, 2) centroid positions
- `headings`: dict mapping session_id to (n_frames,) heading angles

### Visualization

```python
# Syllable dendrogram (hierarchical clustering)
kpms.plot_syllable_dendrogram(model, project_dir='project/')

# Grid movies showing each syllable
kpms.generate_grid_movies(
    results=results,
    project_dir='project/',
    coordinates=coordinates,
    video_dir='videos/',
    output_dir='project/grid_movies/',
)

# Syllable frequencies
kpms.plot_syllable_frequencies(results, project_dir='project/')
```

## Config File (config.yml)

```yaml
# Bodypart configuration
bodyparts: [nose, left_ear, right_ear, neck, tail_base]
use_bodyparts: [nose, left_ear, right_ear, neck, tail_base]
skeleton: [[nose, neck], [neck, tail_base]]
anterior_bodyparts: [nose]
posterior_bodyparts: [tail_base]

# Recording parameters
fps: 30

# Preprocessing
conf_threshold: 0.5
outlier_scale_factor: 6.0

# Model hyperparameters
latent_dim: 10              # PCA dimensions to use
ar_hypparams:
  latent_dim: 10
  num_states: 100           # Max syllables (unused states pruned)
  kappa: 1000000            # State stickiness
```

### Loading/Updating Config

```python
config = kpms.load_config('project/', build_indexes=True)

# Update parameters
kpms.update_config('project/', latent_dim=8, num_states=50)
```

## Applying PCA Manually

For computing PC scores outside the main pipeline:

```python
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca

# Load PCA and config
pca = kpms.load_pca('project/')
config = kpms.load_config('project/', build_indexes=True)

# Preprocess: egocentric alignment + center embedding + flatten
Y_flat, centroid, heading = preprocess_for_pca(
    Y=coordinates,                      # (n_frames, n_keypoints, 2)
    anterior_idxs=config['anterior_idxs'],
    posterior_idxs=config['posterior_idxs'],
    conf=confidences,
    conf_threshold=config['conf_threshold'],
    verbose=False,
)

# Project onto PCs
import numpy as np
Y_flat_np = np.array(Y_flat)
pc_scores = pca.transform(Y_flat_np)    # (n_frames, n_components)
```

## Success Criteria

Good Keypoint-MoSeq fits typically show:
- **Median syllable duration**: 300-400ms (18-24 frames at 60fps, 9-12 frames at 30fps)
- **Grid movies**: Cohesive, interpretable behaviors within each syllable
- **Syllable frequencies**: Power-law-like distribution (few common, many rare)

If syllables are too short/long, adjust `kappa` (stickiness) in config.

## Common Issues

### Too few/many syllables
- Adjust `num_states` in config (model prunes unused states)
- Check `kappa` parameter for state switching rate

### PCA explains variance in few components
- Check for outliers or tracking failures
- Verify egocentric alignment (anterior/posterior bodyparts correct)
- Look for artifacts (human in frame, object occlusion)

### Slow fitting
- Reduce `num_iters` for initial exploration
- Use `ar_only=True` for fast iteration
- Reduce data size for prototyping

## Important Gotchas

### Always use Python 3.10
Keypoint-moseq and its dependencies (especially JAX and jax-moseq) are tested with Python 3.10. Using newer Python versions (3.11+) can cause compatibility issues, installation failures, or subtle runtime bugs. Always create your environment with Python 3.10:
```bash
uv init --python 3.10
# or in pyproject.toml: requires-python = ">=3.10,<3.11"
```

### Kappa range can be much higher than typical
The default kappa is `1e6` and documentation suggests `1e4` to `1e8`, but some datasets require kappa values up to `1e15` or higher to achieve target syllable durations. Don't assume the typical range will work - always tune kappa empirically.

### init_model requires config hyperparameters
The simple call `kpms.init_model(data=data, pca=pca, project_dir=project_dir)` will fail with a ValueError about missing hypparams. You must load config and pass it:
```python
config = kpms.load_config(project_dir)
model = kpms.init_model(data=data, pca=pca, **config)
```

### update_config corrupts YAML with numpy types
If you pass a numpy scalar to `kpms.update_config()`, it writes a numpy object to the YAML that can't be parsed back. Always convert to Python types:
```python
# Wrong - will corrupt config.yml
kpms.update_config(project_dir, kappa=np.float64(1e15))

# Correct
kpms.update_config(project_dir, kappa=float(1e15))
```

### get_durations() is in jax_moseq.utils
To compute syllable duration statistics, import from jax_moseq:
```python
from jax_moseq.utils import get_durations
durations = get_durations(model['states']['z'], mask=data['mask'])
median_duration = np.median(durations)
```

### PCA is sensitive to sessions with high NaN rates
Sessions with very high percentages of NaN frames (e.g., 70%+) can cause PCA variance to collapse into the first component. Check for and exclude problematic sessions before fitting PCA.

### GPU memory estimation and troubleshooting
Full model fitting requires approximately **3MB of VRAM per 100 frames**. For large datasets:
```
1M frames → ~30GB VRAM
10M frames → ~300GB VRAM
```

If your dataset exceeds available GPU memory, use these options:

**1. `set_mixed_map_iters(N)`** - Splits data into N batches processed serially, reducing memory ~N-fold:
```python
from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(8)  # Process in 8 batches: 300GB / 8 = ~37GB

# Then fit as normal
model, model_name = kpms.fit_model(...)
```

**2. `parallel_message_passing=False`** - Reduces memory in Kalman smoother (4-6x reduction):
```python
model, model_name = kpms.fit_model(
    model=model,
    data=data,
    metadata=metadata,
    project_dir=project_dir,
    ar_only=False,
    num_iters=50,
    parallel_message_passing=False,  # Lower memory usage
)
```

Combine both for maximum memory reduction.

**Important:** Call `set_mixed_map_iters()` immediately after imports, before any other JAX operations. JAX compiles functions on first use, so the setting must be configured before any model initialization or fitting begins.

### Always reindex syllables before extract_results
After fitting the full model, syllable labels are arbitrary model state indices (e.g., syllable 52 might be the most frequent). You **must** run `reindex_syllables_in_checkpoint` before `extract_results` to renumber syllables by frequency:
```python
# After fitting, before extracting results:
kpms.reindex_syllables_in_checkpoint(project_dir=project_dir, model_name=model_name)

# Now extract results - syllable 0 = most frequent, 1 = second, etc.
results = kpms.extract_results(...)
```

Without this step:
- Syllable labels are meaningless numbers (internal HMM state indices)
- The frequency plot x-axis ("syllable rank") won't match syllable labels
- Grid movie filenames won't correspond to frequency order

This is an **in-place operation** that modifies the checkpoint file. If you've already run extract_results without reindexing, you need to:
1. Run `reindex_syllables_in_checkpoint`
2. Re-run `extract_results`
3. Re-generate any downstream outputs (grid movies, plots, etc.)

### Use binary search for kappa tuning
When tuning kappa to achieve target syllable duration, use binary search in log-space rather than linear adjustment. This converges much faster across the wide range of possible kappa values:
```python
log_kappa_min, log_kappa_max = 3, 18  # 1e3 to 1e18
while not in_target:
    log_kappa = (log_kappa_min + log_kappa_max) / 2
    kappa = 10 ** log_kappa
    # fit model, check duration
    if duration < target_min:
        log_kappa_min = log_kappa  # need higher kappa
    else:
        log_kappa_max = log_kappa  # need lower kappa
```

**Important:** Kappa tuning must be done twice - once for AR-HMM fitting and again for full model fitting, as the optimal kappa differs between them. Start the full model binary search using the final kappa value from AR-HMM as the initial guess.

### Checkpoints are saved automatically by fit_model
There is **no `save_checkpoint()` function**. Checkpoints are saved automatically by `fit_model()`:
- Each `fit_model()` call creates a new checkpoint directory with auto-generated timestamp name (e.g., `2024_01_15-10_30_45`)
- Returns `(model, model_name)` where `model_name` identifies the checkpoint
- Checkpoint stored at: `{project_dir}/{model_name}/checkpoint.h5`
- Load with: `kpms.load_checkpoint(project_dir=..., model_name=...)`

To track checkpoints for resumability, store the `model_name` returned by `fit_model()` in a state file.

### Warm-start full model from AR-HMM checkpoint
The proper fitting flow uses the AR-HMM model as a warm start for full model fitting:

1. **AR-HMM tuning**: Binary search with ~25 iters per kappa attempt
2. **AR-HMM completion**: When correct kappa found, continue to 50 total iters (checkpoint auto-saved)
3. **Full model tuning**: Load AR-HMM checkpoint by model_name, binary search with ~25 iters per attempt
4. **Full model completion**: When correct kappa found, continue to 200 total iters

```python
# AR-HMM fit - checkpoint saved automatically, track model_name
ar_model, arhmm_model_name = kpms.fit_model(
    model=ar_model, data=data, metadata=metadata,
    project_dir=project_dir, ar_only=True,
    num_iters=25,
)
# Store arhmm_model_name in your state file for later use!

# For full model, load AR-HMM checkpoint by model_name and update kappa
model, _, _, _ = kpms.load_checkpoint(
    project_dir=project_dir,
    model_name=arhmm_model_name,
)
model['hypparams']['trans_hypparams']['kappa'] = float(new_kappa)

# Continue fitting with ar_only=False
model, model_name = kpms.fit_model(
    model=model, data=data, metadata=metadata,
    project_dir=project_dir, ar_only=False,
    num_iters=25,
)
```

This warm-start approach is much faster than initializing the full model from scratch each time.

### Cleanup intermediate checkpoints
Each `fit_model()` call creates a new checkpoint directory. After a long pipeline, clean up intermediate checkpoints:
```python
import shutil

def cleanup_intermediate_checkpoints(project_dir, keep_model_name):
    """Remove all checkpoint directories except the final one."""
    for item in project_dir.iterdir():
        if item.is_dir() and item.name != keep_model_name:
            if (item / "checkpoint.h5").exists():
                shutil.rmtree(item)
                print(f"Removed: {item.name}")

# At end of pipeline, keep only the final model
cleanup_intermediate_checkpoints(PROJECT_DIR, final_model_name)
```

### extract_results requires model_name parameter
When calling `extract_results`, you must pass `model_name` (returned by `fit_model`) or it will fail:
```python
# Wrong - will raise AssertionError about model_name
results = kpms.extract_results(model=model, metadata=metadata, project_dir=project_dir)

# Correct
results = kpms.extract_results(
    model=model,
    metadata=metadata,
    project_dir=project_dir,
    model_name=model_name,  # Required!
)
```

### Auto-detect fps from video files
Don't hardcode fps - detect it from video files using OpenCV:
```python
import cv2
import random
from pathlib import Path

def detect_fps_from_video(video_dir: Path) -> float:
    """Detect fps from a random video in the directory."""
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    video_path = random.choice(videos)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
```

### Resumable pipelines with conditional execution
For long-running pipelines, add conditional checks to skip completed steps:
```python
# Skip outlier removal if QA plots already exist
if (project_dir / "QA" / "plots").exists():
    print("Skipping outlier removal (already done)")
else:
    coordinates, confidences = kpms.outlier_removal(...)

# Skip PCA if pca.p exists
if (project_dir / "pca.p").exists():
    pca = kpms.load_pca(project_dir)
else:
    pca = kpms.fit_pca(...)
    kpms.save_pca(pca, project_dir)

# Skip model fitting if checkpoint exists (check by model_name from state file)
saved_model_name = state.get("model_name")
if saved_model_name and (project_dir / saved_model_name / "checkpoint.h5").exists():
    model, _, _, _ = kpms.load_checkpoint(
        project_dir=project_dir,
        model_name=saved_model_name,
    )
else:
    model, model_name = kpms.fit_model(...)
    state["model_name"] = model_name  # Save for resumability
```

### Resumable kappa search with persistent state
For binary search kappa tuning that can resume after interruption, persist the search bounds and checkpoint names:
```python
import json

KAPPA_STATE_FILE = project_dir / "kappa_state.json"

def load_kappa_state():
    if KAPPA_STATE_FILE.exists():
        with open(KAPPA_STATE_FILE) as f:
            return json.load(f)
    return {
        "arhmm": {
            "log_min": 3, "log_max": 18,
            "completed": False, "final_kappa": None,
            "model_name": None,  # Track checkpoint name!
        },
        "full": {
            "log_min": None, "log_max": None,
            "completed": False, "final_kappa": None,
            "model_name": None,
        },
    }

def save_kappa_state(state):
    with open(KAPPA_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# In binary search loop:
state = load_kappa_state()
if state["arhmm"]["completed"] and state["arhmm"]["model_name"]:
    # Resume from saved checkpoint
    model, _, _, _ = kpms.load_checkpoint(
        project_dir=project_dir,
        model_name=state["arhmm"]["model_name"],
    )
else:
    # ... do binary search iteration ...
    model, model_name = kpms.fit_model(...)
    state["arhmm"]["log_min"] = log_min
    state["arhmm"]["log_max"] = log_max
    state["arhmm"]["model_name"] = model_name  # Save checkpoint name
    save_kappa_state(state)
```

## Tips

1. **Start with AR-HMM**: Use `ar_only=True` for fast iteration before full model fitting.

2. **Check PCA first**: Scree plot and PC visualizations reveal data quality issues before slow fitting.

3. **Use QA plots**: Outlier removal generates useful diagnostics in `project/QA/plots/`.

4. **Consistent bodyparts**: Ensure all recordings use the same bodypart names and ordering.

5. **Appropriate fps**: Set fps correctly in config - affects syllable duration interpretation.

6. **Save checkpoints**: Full model fitting is slow; checkpoints allow resuming.

7. **Grid movies for validation**: The ultimate test is whether syllables look behaviorally meaningful.
