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

---

## Canonical Pipeline (Gold Standard)

**This is the refined, production-ready pipeline that handles all known edge cases.**

This pipeline incorporates:
- Automatic latent_dim selection (90% variance explained)
- Binary search kappa tuning for both AR-HMM and full model
- Full resumability (can be interrupted and restarted at any point)
- Proper conditional execution (skips completed steps)
- Automatic cleanup of intermediate checkpoints
- Auto-detection of fps from video files

When adapting for your own projects, this should be your starting point. Modify paths and
bodypart configurations as needed, but preserve the overall structure and flow.

```python
#!/usr/bin/env python
"""
Canonical Keypoint-MoSeq Pipeline
=================================

This is the gold-standard pipeline for fitting Keypoint-MoSeq models. It handles:
- Automatic latent_dim selection based on variance explained
- Binary search kappa tuning (separately for AR-HMM and full model)
- Full resumability via persistent state files
- Conditional execution (skips already-completed steps)
- Automatic fps detection from video files
- Cleanup of intermediate checkpoints

IMPORTANT: This pipeline has been refined through extensive testing and handles
many edge cases in the keypoint-moseq API. When starting a new project, copy this
file and modify the Configuration section - do not rewrite the pipeline logic
from scratch unless you have a specific reason to do so.

Key design decisions:
1. Binary search in log-space for kappa (range can span 1e3 to 1e18)
2. Separate kappa tuning for AR-HMM and full model (optimal values differ)
3. State persistence via JSON for resumability across interruptions
4. Skip logic based on state file, not checkpoint existence (intermediate
   checkpoints get cleaned up, but state file persists)
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import keypoint_moseq as kpms
from jax_moseq.utils import get_durations

# vidio is a keypoint-moseq dependency - no need to install separately
# Using vidio instead of cv2 directly avoids dependency conflicts
from vidio.read import OpenCVReader


# =============================================================================
# FPS Detection
# =============================================================================
# Don't hardcode fps - detect it from actual video files. This ensures the
# syllable duration calculations are accurate for your specific recordings.

def detect_fps_from_video(video_dir: Path) -> float:
    """
    Detect fps from a random video in the directory.

    Uses vidio (a keypoint-moseq dependency) rather than importing cv2 directly.
    This avoids potential dependency conflicts when installing opencv-python.
    """
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    video_path = random.choice(videos)
    reader = OpenCVReader(str(video_path))
    fps = reader.fps

    print(f"  Detected fps={fps} from {video_path.name}")
    return fps


# =============================================================================
# Configuration
# =============================================================================
# MODIFY THIS SECTION for your project. The rest of the pipeline should work
# without changes for most use cases.

# Paths - adjust these for your project structure
REPO_ROOT = Path(__file__).parent.parent
DLC_CONFIG = REPO_ROOT / "dlc_project" / "config.yaml"  # DeepLabCut config
DLC_DATA_PATH = REPO_ROOT / "dlc_project" / "videos"    # Directory with .h5 files
PROJECT_DIR = REPO_ROOT / "fitted_model"                 # Output directory

# Bodypart configuration - adjust for your tracking setup
ANTERIOR_BODYPARTS = ["nose"]           # Front of animal (for heading calculation)
POSTERIOR_BODYPARTS = ["spine4"]        # Back of animal (for heading calculation)
USE_BODYPARTS = [                       # Bodyparts to include in model
    "spine4", "spine3", "spine2", "spine1",
    "head", "nose", "right ear", "left ear"
]

# Preprocessing
OUTLIER_SCALE_FACTOR = 6.0  # Higher = fewer outliers removed (default: 6.0)
VARIANCE_THRESHOLD = 0.90   # PCA components to explain this much variance

# Kappa tuning - target syllable duration
# Good fits typically have median duration 300-400ms
# Adjust TARGET_DURATION_MIN/MAX based on your fps
TARGET_DURATION_MIN = 9   # frames (300ms at 30fps, 18 at 60fps)
TARGET_DURATION_MAX = 12  # frames (400ms at 30fps, 24 at 60fps)

# Kappa search bounds (in log10 space)
# These are very wide bounds - binary search will narrow quickly
KAPPA_LOG_MIN = 3         # 1e3 - very low stickiness
KAPPA_LOG_MAX = 18        # 1e18 - very high stickiness
MAX_KAPPA_SEARCH_ITERS = 15  # Usually converges in <10 iterations

# Fitting iterations
KAPPA_SEARCH_ITERS = 25    # Iterations per kappa attempt during search
ARHMM_TOTAL_ITERS = 50     # Total AR-HMM iterations after finding kappa
FULL_TOTAL_ITERS = 200     # Total full model iterations (this is slow!)


# =============================================================================
# Helper Functions
# =============================================================================

def get_latent_dim_for_variance(pca, threshold=0.90):
    """
    Find the number of PCA components needed to explain `threshold` variance.

    This automates latent_dim selection rather than picking an arbitrary number.
    90% variance is a good default - adjust if you need more/fewer dimensions.
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.searchsorted(cumvar, threshold) + 1)
    print(f"PCA: {latent_dim} components explain {cumvar[latent_dim-1]*100:.1f}% variance (target: {threshold*100}%)")
    return latent_dim


def compute_median_duration(model, mask):
    """
    Compute median syllable duration from model states.

    This is the key metric for evaluating kappa tuning.
    Target: 300-400ms (9-12 frames at 30fps, 18-24 at 60fps).
    """
    durations = get_durations(model['states']['z'], mask=mask)
    return float(np.median(durations))


def cleanup_intermediate_checkpoints(project_dir: Path, keep_model_name: str) -> int:
    """
    Remove all checkpoint directories except the final one.

    Each fit_model() call creates a new checkpoint. After a full pipeline run,
    you may have 20+ intermediate checkpoints taking up disk space. This cleans
    them up while preserving the final model.

    Returns the number of checkpoints removed.
    """
    removed = 0
    for item in project_dir.iterdir():
        if item.is_dir() and item.name != keep_model_name:
            # Check if it's a checkpoint directory (has checkpoint.h5)
            if (item / "checkpoint.h5").exists():
                shutil.rmtree(item)
                print(f"    Removed: {item.name}")
                removed += 1
    return removed


# =============================================================================
# Kappa State Persistence
# =============================================================================
# This is the key to resumability. The state file tracks:
# - Binary search bounds (log_min, log_max) for each phase
# - Whether each phase completed successfully
# - The final kappa and duration found
# - The checkpoint name (model_name) for loading later
#
# IMPORTANT: Skip logic for AR-HMM uses state["completed"], NOT checkpoint
# existence - because cleanup removes intermediate checkpoints!

KAPPA_STATE_FILE = PROJECT_DIR / "kappa_state.json"


def load_kappa_state() -> dict:
    """Load kappa tuning state from disk, or return initial state."""
    if KAPPA_STATE_FILE.exists():
        with open(KAPPA_STATE_FILE) as f:
            return json.load(f)
    return {
        "arhmm": {
            "log_min": KAPPA_LOG_MIN,
            "log_max": KAPPA_LOG_MAX,
            "completed": False,
            "final_kappa": None,
            "final_duration": None,
            "model_name": None,  # Checkpoint name after AR-HMM fit
        },
        "full": {
            "log_min": None,  # Initialized from AR-HMM result
            "log_max": None,
            "completed": False,
            "final_kappa": None,
            "final_duration": None,
            "model_name": None,  # Checkpoint name after full model fit
        },
    }


def save_kappa_state(state: dict) -> None:
    """Save kappa tuning state to disk for resumability."""
    with open(KAPPA_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def binary_search_kappa(
    fit_func,
    mask,
    state_key: str,
    fps: float,
    target_min=TARGET_DURATION_MIN,
    target_max=TARGET_DURATION_MAX,
    log_min=KAPPA_LOG_MIN,
    log_max=KAPPA_LOG_MAX,
    max_iters=MAX_KAPPA_SEARCH_ITERS,
):
    """
    Binary search for kappa that produces target syllable duration.

    WHY BINARY SEARCH IN LOG SPACE?
    Kappa can range from 1e3 to 1e18 - a factor of 10^15! Linear search would
    take forever. Binary search in log space converges in ~10 iterations.

    WHY RESUMABLE?
    Each iteration takes ~1 minute for AR-HMM, ~5 minutes for full model.
    A full search could take an hour. If interrupted, we want to resume from
    the narrowed bounds, not start over.

    Args:
        fit_func: Function that takes kappa and returns (model, model_name)
        mask: Data mask for duration calculation
        state_key: Key in state file ("arhmm" or "full")
        fps: Frames per second (for ms conversion in output)
        target_min: Minimum acceptable median duration (frames)
        target_max: Maximum acceptable median duration (frames)
        log_min: Log10 of minimum kappa (used if no saved state)
        log_max: Log10 of maximum kappa (used if no saved state)
        max_iters: Maximum binary search iterations

    Returns:
        (kappa, model, model_name, median_duration)
    """
    # Load persistent state - may have narrowed bounds from previous run
    state = load_kappa_state()
    key_state = state[state_key]

    # Use saved bounds if available, otherwise use provided defaults
    if key_state["log_min"] is not None:
        log_min = key_state["log_min"]
    if key_state["log_max"] is not None:
        log_max = key_state["log_max"]

    print(f"  Starting bounds: log_min={log_min:.2f}, log_max={log_max:.2f}")

    # Track best result (in case we hit max_iters without finding target)
    best_kappa: float = 10 ** ((log_min + log_max) / 2)
    best_model: dict = {}
    best_model_name: str = ""
    best_duration: float = 0.0

    for i in range(max_iters):
        # Binary search: try the midpoint
        log_kappa = (log_min + log_max) / 2
        kappa = 10 ** log_kappa

        print(f"  Iteration {i+1}/{max_iters}: kappa = {kappa:.2e} (log={log_kappa:.2f})")

        # Fit model with this kappa
        model, model_name = fit_func(kappa)
        median_dur = compute_median_duration(model, mask)

        print(f"    Median duration: {median_dur:.1f} frames ({median_dur/fps*1000:.0f}ms)")

        # Track best result
        best_kappa = kappa
        best_model = model
        best_model_name = model_name
        best_duration = median_dur

        # Check if in target range
        if target_min <= median_dur <= target_max:
            print(f"    Found target duration!")
            # Mark as completed and save final results
            state[state_key]["completed"] = True
            state[state_key]["final_kappa"] = float(best_kappa)
            state[state_key]["final_duration"] = float(best_duration)
            save_kappa_state(state)
            break
        elif median_dur < target_min:
            # Syllables too short -> states switching too fast -> need MORE stickiness
            log_min = log_kappa
            print(f"    Too short, increasing kappa...")
        else:
            # Syllables too long -> states too sticky -> need LESS stickiness
            log_max = log_kappa
            print(f"    Too long, decreasing kappa...")

        # Save updated bounds after EVERY iteration (for resumability)
        state[state_key]["log_min"] = float(log_min)
        state[state_key]["log_max"] = float(log_max)
        save_kappa_state(state)
    else:
        # Loop completed without finding target - save best result anyway
        state[state_key]["completed"] = True
        state[state_key]["final_kappa"] = float(best_kappa)
        state[state_key]["final_duration"] = float(best_duration)
        save_kappa_state(state)
        print(f"    Max iterations reached, using best kappa found")

    return best_kappa, best_model, best_model_name, best_duration


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Keypoint-MoSeq Pipeline")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 0: Detect fps from video
    # -------------------------------------------------------------------------
    # Don't hardcode fps! Different rigs record at different rates, and fps
    # affects all duration calculations.
    print("\n[0/9] Detecting fps from video...")
    fps = detect_fps_from_video(DLC_DATA_PATH)

    # -------------------------------------------------------------------------
    # Step 1: Setup project
    # -------------------------------------------------------------------------
    # Creates config.yml with all hyperparameters. If project already exists,
    # just update the config with current settings.
    print("\n[1/9] Setting up project...")

    if PROJECT_DIR.exists():
        print(f"  Project directory exists: {PROJECT_DIR}")
    else:
        kpms.setup_project(
            str(PROJECT_DIR),
            deeplabcut_config=str(DLC_CONFIG),
        )
        print(f"  Created project: {PROJECT_DIR}")

    # IMPORTANT: Always convert values to Python types (not numpy) when calling
    # update_config, or it will corrupt the YAML file!
    kpms.update_config(
        str(PROJECT_DIR),
        video_dir=str(DLC_DATA_PATH),
        anterior_bodyparts=ANTERIOR_BODYPARTS,
        posterior_bodyparts=POSTERIOR_BODYPARTS,
        use_bodyparts=USE_BODYPARTS,
        fps=fps,
        outlier_scale_factor=OUTLIER_SCALE_FACTOR,
    )
    print("  Updated config")

    # -------------------------------------------------------------------------
    # Step 2: Load keypoints
    # -------------------------------------------------------------------------
    # Loads pose tracking data. Format can be "deeplabcut", "sleap", "nwb", etc.
    print("\n[2/9] Loading keypoints...")

    coordinates, confidences, bodyparts = kpms.load_keypoints(
        str(DLC_DATA_PATH),
        "deeplabcut",  # Change this for other formats
    )
    print(f"  Loaded {len(coordinates)} sessions")
    print(f"  Bodyparts: {bodyparts}")

    # -------------------------------------------------------------------------
    # Step 3: Outlier removal
    # -------------------------------------------------------------------------
    # Identifies keypoints too far from animal medoid and interpolates them.
    # Skip if already done (QA plots exist).
    print("\n[3/9] Removing outliers...")

    outlier_plots_dir = PROJECT_DIR / "QA" / "plots"
    if outlier_plots_dir.exists():
        print(f"  Skipping (plots exist): {outlier_plots_dir}")
    else:
        config = kpms.load_config(str(PROJECT_DIR))
        coordinates, confidences = kpms.outlier_removal(
            coordinates,
            confidences,
            str(PROJECT_DIR),
            overwrite=True,
            **config,  # Pass config to get bodyparts, etc.
        )
        print("  Outliers removed")

    # -------------------------------------------------------------------------
    # Step 4: Format data
    # -------------------------------------------------------------------------
    # Batches recordings into segments for model fitting.
    # Must be done every run (depends on coordinates which may be modified).
    print("\n[4/9] Formatting data...")

    config = kpms.load_config(str(PROJECT_DIR))
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    print(f"  Data shape: Y={data['Y'].shape}")

    # -------------------------------------------------------------------------
    # Step 5: Fit PCA and select latent_dim
    # -------------------------------------------------------------------------
    # PCA reduces pose dimensions. We auto-select latent_dim to explain 90%
    # of variance rather than picking an arbitrary number.
    print("\n[5/9] Fitting PCA...")

    pca_file = PROJECT_DIR / "pca.p"
    if pca_file.exists():
        print(f"  Skipping (file exists): {pca_file}")
        pca = kpms.load_pca(str(PROJECT_DIR))
    else:
        config = kpms.load_config(str(PROJECT_DIR))
        pca = kpms.fit_pca(
            Y=data['Y'],
            mask=data['mask'],
            conf=data['conf'],
            anterior_idxs=config['anterior_idxs'],
            posterior_idxs=config['posterior_idxs'],
            conf_threshold=config.get('conf_threshold', 0.5),
        )
        kpms.save_pca(pca, str(PROJECT_DIR))

    # Auto-select latent_dim for target variance
    latent_dim = get_latent_dim_for_variance(pca, VARIANCE_THRESHOLD)

    # IMPORTANT: Convert to Python int! Numpy types corrupt the YAML.
    kpms.update_config(str(PROJECT_DIR), latent_dim=int(latent_dim))
    print(f"  Set latent_dim = {latent_dim}")

    # -------------------------------------------------------------------------
    # Step 6: Tune kappa for AR-HMM and fit to 50 iterations
    # -------------------------------------------------------------------------
    # AR-HMM fitting is fast (~1 min/iteration). We binary search for the
    # kappa that produces our target syllable duration.
    #
    # SKIP LOGIC: Check state["completed"], NOT checkpoint existence!
    # The AR-HMM checkpoint gets deleted by cleanup, but we still have the
    # kappa value we need in the state file.
    print("\n[6/9] Tuning kappa for AR-HMM...")

    kappa_state = load_kappa_state()

    if kappa_state["arhmm"]["completed"]:
        # Already done - just load the values we need
        ar_kappa = kappa_state["arhmm"]["final_kappa"]
        ar_duration = kappa_state["arhmm"]["final_duration"]
        arhmm_model_name = kappa_state["arhmm"].get("model_name")
        print(f"  Skipping (already completed)")
        print(f"    Kappa: {ar_kappa:.2e}")
        print(f"    Median duration: {ar_duration:.1f} frames ({ar_duration/fps*1000:.0f}ms)")
    else:
        # Binary search for kappa (resumes from saved bounds if interrupted)
        def fit_ar_only(kappa):
            """Fit AR-HMM with given kappa value."""
            # IMPORTANT: Convert kappa to float! Numpy types corrupt YAML.
            kpms.update_config(str(PROJECT_DIR), kappa=float(kappa))
            config = kpms.load_config(str(PROJECT_DIR))

            # IMPORTANT: Must pass **config to init_model or it fails!
            model = kpms.init_model(data=data, pca=pca, **config)

            model, model_name = kpms.fit_model(
                model=model, data=data, metadata=metadata,
                project_dir=str(PROJECT_DIR), ar_only=True,
                num_iters=KAPPA_SEARCH_ITERS,
            )
            return model, model_name

        ar_kappa, ar_model, _, ar_duration = binary_search_kappa(
            fit_ar_only, data['mask'], state_key="arhmm", fps=fps,
        )
        print(f"\n  AR-HMM kappa tuning complete:")
        print(f"    Kappa: {ar_kappa:.2e}")
        print(f"    Median duration: {ar_duration:.1f} frames ({ar_duration/fps*1000:.0f}ms)")

        # Continue fitting to 50 total iterations with the final kappa
        remaining_iters = ARHMM_TOTAL_ITERS - KAPPA_SEARCH_ITERS
        print(f"\n  Continuing AR-HMM fit for {remaining_iters} more iterations (to {ARHMM_TOTAL_ITERS} total)...")
        ar_model, arhmm_model_name = kpms.fit_model(
            model=ar_model, data=data, metadata=metadata,
            project_dir=str(PROJECT_DIR), ar_only=True,
            num_iters=remaining_iters,
        )
        ar_duration = compute_median_duration(ar_model, data['mask'])
        print(f"    Final median duration: {ar_duration:.1f} frames ({ar_duration/fps*1000:.0f}ms)")

        # Save the checkpoint name for later use
        kappa_state = load_kappa_state()
        kappa_state["arhmm"]["model_name"] = arhmm_model_name
        save_kappa_state(kappa_state)
        print(f"  AR-HMM checkpoint saved: {arhmm_model_name}")

    # -------------------------------------------------------------------------
    # Step 7: Tune kappa for full model and fit to 200 iterations
    # -------------------------------------------------------------------------
    # Full model fitting is SLOW (~5 min/iteration). We warm-start from the
    # AR-HMM checkpoint and binary search for kappa again (optimal value
    # differs between AR-HMM and full model).
    #
    # SKIP LOGIC: Check state["completed"] AND checkpoint existence.
    # Unlike AR-HMM, we need the full model checkpoint for step 8, so we
    # verify it still exists.
    print("\n[7/9] Tuning kappa for full model...")

    kappa_state = load_kappa_state()
    final_model_name = kappa_state["full"].get("model_name")

    if kappa_state["full"]["completed"] and final_model_name and (PROJECT_DIR / final_model_name / "checkpoint.h5").exists():
        # Already done and checkpoint exists - just load it
        full_kappa = kappa_state["full"]["final_kappa"]
        full_duration = kappa_state["full"]["final_duration"]
        print(f"  Skipping (already completed)")
        print(f"    Kappa: {full_kappa:.2e}")
        print(f"    Median duration: {full_duration:.1f} frames ({full_duration/fps*1000:.0f}ms)")

        # Load the final model for step 8
        model, _, _, _ = kpms.load_checkpoint(
            project_dir=str(PROJECT_DIR),
            model_name=final_model_name,
        )
        model_name = final_model_name
        final_duration = compute_median_duration(model, data['mask'])
    else:
        # Need to fit - load the AR-HMM checkpoint as warm start
        print(f"  Loading AR-HMM checkpoint: {arhmm_model_name}")
        arhmm_model, _, _, _ = kpms.load_checkpoint(
            project_dir=str(PROJECT_DIR),
            model_name=arhmm_model_name,
        )

        # Initialize full model search bounds from AR-HMM result
        # Start with a narrower range centered on AR-HMM kappa
        ar_log_kappa = np.log10(ar_kappa)
        if kappa_state["full"]["log_min"] is None:
            kappa_state["full"]["log_min"] = float(max(KAPPA_LOG_MIN, ar_log_kappa - 3))
        if kappa_state["full"]["log_max"] is None:
            kappa_state["full"]["log_max"] = float(min(KAPPA_LOG_MAX, ar_log_kappa + 3))
        save_kappa_state(kappa_state)

        def fit_full_model(kappa):
            """Fit full model with given kappa, warm-starting from AR-HMM."""
            # Reload AR-HMM checkpoint fresh each iteration
            # (previous iteration's model has wrong kappa baked in)
            model, _, _, _ = kpms.load_checkpoint(
                project_dir=str(PROJECT_DIR),
                model_name=arhmm_model_name,
            )

            # Update kappa in config AND in model's hyperparams
            kpms.update_config(str(PROJECT_DIR), kappa=float(kappa))
            model['hypparams']['trans_hypparams']['kappa'] = float(kappa)

            # Fit full model (ar_only=False)
            model, model_name = kpms.fit_model(
                model=model, data=data, metadata=metadata,
                project_dir=str(PROJECT_DIR), ar_only=False,
                num_iters=KAPPA_SEARCH_ITERS,
            )
            return model, model_name

        full_kappa, full_model, _, full_duration = binary_search_kappa(
            fit_full_model, data['mask'], state_key="full", fps=fps,
        )
        print(f"\n  Full model kappa tuning complete:")
        print(f"    Kappa: {full_kappa:.2e}")
        print(f"    Median duration: {full_duration:.1f} frames ({full_duration/fps*1000:.0f}ms)")

        # Continue fitting to 200 total iterations
        # Currently at 75 iters (50 AR-HMM + 25 full), need 125 more
        current_iters = ARHMM_TOTAL_ITERS + KAPPA_SEARCH_ITERS  # 75
        remaining_iters = FULL_TOTAL_ITERS - current_iters      # 125
        print(f"\n  Continuing full model fit for {remaining_iters} more iterations (to {FULL_TOTAL_ITERS} total)...")

        model, model_name = kpms.fit_model(
            model=full_model, data=data, metadata=metadata,
            project_dir=str(PROJECT_DIR), ar_only=False,
            num_iters=remaining_iters,
        )

        final_duration = compute_median_duration(model, data['mask'])
        print(f"  Final median duration: {final_duration:.1f} frames ({final_duration/fps*1000:.0f}ms)")

        # Save final model name to state
        kappa_state = load_kappa_state()
        kappa_state["full"]["model_name"] = model_name
        save_kappa_state(kappa_state)

    # -------------------------------------------------------------------------
    # Step 8: Reindex and extract results
    # -------------------------------------------------------------------------
    # IMPORTANT: Must reindex syllables BEFORE extracting results!
    # After fitting, syllable labels are arbitrary internal state indices.
    # Reindexing renumbers them by frequency (0 = most common).
    #
    # Skip if results.h5 already exists (avoids "already exists" error).
    print("\n[8/9] Extracting results...")

    results_path = PROJECT_DIR / model_name / "results.h5"
    if results_path.exists():
        print(f"  Skipping (results already exist): {results_path}")
        # Load existing results for summary
        results = kpms.load_results(str(PROJECT_DIR), model_name=model_name)
    else:
        # Reindex syllables by frequency (syllable 0 = most common)
        kpms.reindex_syllables_in_checkpoint(
            project_dir=str(PROJECT_DIR),
            model_name=model_name,
        )
        print("  Reindexed syllables by frequency")

        # Extract results - MUST pass model_name or it will fail!
        results = kpms.extract_results(
            model=model,
            metadata=metadata,
            project_dir=str(PROJECT_DIR),
            model_name=model_name,
        )
        print(f"  Extracted results for {len(results)} sessions")

    # -------------------------------------------------------------------------
    # Step 9: Cleanup intermediate checkpoints
    # -------------------------------------------------------------------------
    # Each fit_model() call creates a new checkpoint directory. After kappa
    # tuning, we may have 20+ intermediate checkpoints. Clean them up to
    # save disk space, keeping only the final model.
    print("\n[9/9] Cleaning up intermediate checkpoints...")
    n_removed = cleanup_intermediate_checkpoints(PROJECT_DIR, model_name)
    print(f"  Removed {n_removed} intermediate checkpoint(s), kept: {model_name}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {PROJECT_DIR}")
    print(f"Model checkpoint: {model_name}")
    print(f"\nSettings:")
    print(f"  latent_dim: {latent_dim} ({VARIANCE_THRESHOLD*100:.0f}% variance)")
    print(f"  AR-HMM kappa: {ar_kappa:.2e}")
    print(f"  Full model kappa: {full_kappa:.2e}")
    print(f"\nFinal metrics:")
    print(f"  Median syllable duration: {final_duration:.1f} frames ({final_duration/fps*1000:.0f}ms)")

    # Count unique syllables
    # Results is dict of {session_name: {'syllable': array, 'centroid': array, ...}}
    all_syllables = np.concatenate([results[key]['syllable'] for key in results])
    n_syllables = len(np.unique(all_syllables))
    print(f"  Number of syllables: {n_syllables}")

    return results


if __name__ == "__main__":
    main()
```
