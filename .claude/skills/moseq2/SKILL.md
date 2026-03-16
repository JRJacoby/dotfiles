---
name: moseq2
description: Work with the MoSeq2 depth-video behavioral segmentation pipeline. Use when dealing with depth camera recordings, 3D mouse tracking, AR-HMM modeling, or any of the moseq2-* packages.
---

# MoSeq2

MoSeq2 (Motion Sequencing) is a suite of five Python packages for unsupervised behavioral segmentation from **depth camera** recordings of mice. It discovers discrete behavioral "syllables" (sub-second motifs) and their sequential structure ("grammar") by fitting autoregressive hidden Markov models (AR-HMMs) to 3D depth video data.

This is the depth-video counterpart to **keypoint-moseq**, which operates on 2D pose tracking data (SLEAP, DeepLabCut). MoSeq2 works directly on raw depth frames rather than tracked keypoints.

## Repository Locations

All repositories live under `/home/joj144/projects/keypoint-moseq/`:

| Repository | Package | Purpose |
|------------|---------|---------|
| `moseq2-extract/` | `moseq2_extract` | Depth video processing and mouse extraction |
| `moseq2-pca/` | `moseq2_pca` | PCA on extracted frames + changepoint detection |
| `moseq2-model/` | `moseq2_model` | AR-HMM training for syllable discovery |
| `moseq2-viz/` | `moseq2_viz` | Visualization, crowd movies, statistics |
| `moseq2-app/` | `moseq2_app` | Interactive Jupyter notebook UI |

## Pipeline Overview

```
Raw Depth Video (.dat/.avi/.mkv)
        |
        v
  [moseq2-extract]  ──>  Extracted frames + scalars (HDF5)
        |
        v
    [moseq2-pca]    ──>  PCA model + PC scores + changepoints (HDF5)
        |
        v
   [moseq2-model]   ──>  Trained AR-HMM + syllable labels (pickle)
        |
        v
    [moseq2-viz]    ──>  Crowd movies, transition graphs, stats (MP4/PNG/CSV)
        |
        v
    [moseq2-app]    ──>  Interactive Jupyter exploration
```

## Common Infrastructure

All five packages share:
- **Python 3.6-3.7** (legacy pinned versions; strict due to dependency locks)
- **Click-based CLIs** with a `moseq2-<name>` console entry point
- **HDF5 + YAML** data interchange format
- **Index file** (`moseq2-index.yaml`) as the central manifest linking sessions to metadata
- **Datta Lab** (Harvard) authorship
- **Version ~1.2-1.3** across the suite

---

## Package Details

### moseq2-extract

**Purpose:** Process raw depth video from Kinect/Azure/RealSense cameras to detect, track, and extract a cropped, orientation-normalized mouse from each frame.

**Input:** Raw depth recordings (`.dat`, `.avi`, `.mkv`), 16-bit depth frames at ~30 fps.

**Output:** HDF5 files containing:
- `frames`: Cropped, rotated depth frames (n_frames, 80, 80)
- `frames_mask`: Confidence/validity masks
- `scalars/`: Per-frame features (centroid, velocity, area, orientation, depth, etc.)
- `metadata/extraction/`: ROI mask, background image, extraction parameters

**Key processing steps:**
1. Background computation (median across frames)
2. ROI detection via RANSAC plane fitting to arena floor
3. Height thresholding (keep pixels 10-120mm above floor)
4. Spatial/temporal denoising
5. Optional EM tracking for robust mouse detection
6. Centroid + orientation extraction (PCA on silhouette)
7. Crop and rotate to standard orientation
8. Optional flip correction via pre-trained classifier
9. Scalar feature computation (14+ features per frame)

**CLI:** `moseq2-extract [extract|batch-extract|find-roi|generate-config|...]`

**Module layout:**
```
moseq2_extract/
├── cli.py              # Click CLI (9 commands)
├── extract/
│   ├── extract.py      # extract_chunk() - core per-batch extraction
│   ├── proc.py         # Image processing, ROI, features, scalars
│   ├── roi.py          # RANSAC plane fitting
│   └── track.py        # EM tracking algorithm
├── helpers/
│   ├── wrappers.py     # High-level extract_wrapper(), get_roi_wrapper()
│   └── data.py         # HDF5 writing, metadata, indexing
└── io/
    └── video.py        # Raw/AVI depth video reading
```

---

### moseq2-pca

**Purpose:** Fit PCA on extracted depth frames to reduce dimensionality, then project all sessions into PC space. Optionally detect model-free behavioral changepoints.

**Input:** Extracted HDF5 files (frames + masks) from moseq2-extract.

**Output:**
- `pca.h5`: PCA components, singular values, explained variance, mean frame
- `pca_scores.h5`: Per-session PC scores (n_frames, n_components)
- `changepoints.h5`: Detected syllable boundary times (model-free)

**Key processing steps:**
1. Load frames from all sessions into Dask arrays
2. Apply spatial/temporal Gaussian filtering
3. Compute SVD (supports iterative PCA for missing data)
4. Project all frames onto learned components
5. Optional: detect changepoints via random projections + peak detection

**CLI:** `moseq2-pca [train-pca|apply-pca|compute-changepoints|clip-scores]`

**Notable features:**
- Distributed computing via Dask (local or SLURM clusters)
- Missing data handling for cable-contaminated frames (e.g., electrophysiology setups)
- Changepoint detection is model-free (no AR-HMM needed)

**Module layout:**
```
moseq2_pca/
├── cli.py              # Click CLI (4 commands)
├── util.py             # Frame filtering, timestamps, Dask setup
├── viz.py              # Scree plots, component visualization
├── pca/
│   └── util.py         # train_pca_dask(), apply_pca_dask(), compute_svd()
└── helpers/
    ├── wrappers.py     # train_pca_wrapper(), apply_pca_wrapper()
    └── data.py         # H5 loading helpers
```

---

### moseq2-model

**Purpose:** Train Bayesian AR-HMM models on PCA scores to discover behavioral syllables. Each syllable corresponds to a latent state with its own autoregressive dynamics.

**Input:** PCA scores (HDF5, pickle, or MATLAB format) from moseq2-pca.

**Output:** Model file (pickle) containing:
- Trained AR-HMM object (via `pyhsmm` / `autoregressive` libraries)
- Syllable labels per frame per session
- Log-likelihoods (training + validation)
- Whitening parameters (for applying model to new data)
- Model hyperparameters

**Key processing steps:**
1. Load PC scores, organize by session UUID
2. Whiten data (all sessions combined, per-session, or none)
3. Initialize AR-HMM with hyperparameters (kappa, gamma, alpha, nlags, max_states)
4. Train via Gibbs sampling with optional checkpointing
5. Extract syllable labels and expected states

**CLI:** `moseq2-model [learn-model|apply-model|kappa-scan|count-frames]`

**Key hyperparameters:**
- `kappa`: Controls syllable duration (higher = longer syllables)
- `nlags`: Autoregressive lag order (typically 3)
- `max_states`: Maximum possible syllables (unused states are pruned)
- `gamma`, `alpha`: Hierarchical Dirichlet Process priors
- `robust`: Use Student-t AR model for noise tolerance

**Notable features:**
- `kappa-scan` generates batch scripts for parallel hyperparameter sweeps
- Supports separate transition matrices per experimental group
- Empirical Bayes initialization of noise covariance
- Graceful keyboard interrupt handling during training

**Module layout:**
```
moseq2_model/
├── cli.py              # Click CLI (4 commands)
├── util.py             # I/O, model loading/saving, multi-format support
├── train/
│   ├── models.py       # ARHMM() initialization and configuration
│   └── util.py         # train_model() loop, run_e_step(), checkpointing
└── helpers/
    ├── data.py         # load_pcs(), prepare_model_metadata(), whitening
    └── wrappers.py     # learn_model_wrapper(), apply_model_wrapper()
```

**Core dependency:** `pyhsmm` + `pybasicbayes` + `autoregressive` (Matthew Johnson's Bayesian HMM libraries)

---

### moseq2-viz

**Purpose:** Post-model visualization and analysis. Generates crowd movies, transition graphs, statistical comparisons, position heatmaps, and exportable dataframes.

**Input:** Trained model file (pickle) + index file (YAML) + extracted HDF5 files.

**Output:**
- Crowd movies (MP4): Composite videos showing exemplar instances of each syllable
- Statistical plots (PNG/PDF): Usage, duration, velocity bar charts with error bars
- Transition graphs (PNG/PDF): Network visualizations of syllable sequences
- Position heatmaps (PNG/PDF): Spatial occupancy per group/session
- DataFrames (CSV/Parquet): Structured data for downstream statistical analysis

**CLI:** `moseq2-viz [make-crowd-movies|plot-stats|plot-transition-graph|plot-group-position-heatmaps|make-df|add-group|get-best-model|...]`

**Key capabilities:**
- Syllable relabeling by usage frequency
- Group-aware comparisons (control vs. treatment)
- Multiple sorting modes: usage, duration, velocity, group difference
- Transition matrix computation and network graph rendering (via NetworkX)
- Scalar aggregation per syllable (velocity, height, area, distance to center)
- Multi-processing for crowd movie generation

**Module layout:**
```
moseq2_viz/
├── cli.py              # Click CLI (10 commands)
├── viz.py              # Plotting functions (bar charts, heatmaps, matrices)
├── util.py             # YAML/HDF5 I/O, index parsing
├── model/
│   ├── util.py         # parse_model_results(), syllable statistics, relabeling
│   ├── trans_graph.py  # Transition matrices, network graph operations
│   ├── stat.py         # Statistical computations
│   ├── dist.py         # Distance metrics
│   └── embed.py        # Embedding operations
├── scalars/
│   └── util.py         # scalars_to_dataframe(), position PDFs, px→mm conversion
└── io/
    └── video.py        # write_crowd_movies(), video parameter validation
```

---

### moseq2-app

**Purpose:** Interactive Jupyter notebook frontend that wraps and coordinates all other packages. Provides widgets for exploration, validation, labeling, and comparison.

**Input:** All data products from the other four packages.

**Output:** Interactive Jupyter widgets + saved figures, crowd movies, and metadata.

**Key interactive tools (all in `moseq2_app.main`):**
1. `preview_extractions()` / `validate_extractions()` - Video playback and anomaly detection
2. `interactive_group_setting()` - Editable table for assigning experimental groups
3. `interactive_scalar_summary()` - Violin plots of behavioral measurements
4. `label_syllables()` - Preview crowd movies and assign syllable names/descriptions
5. `show_dendrogram()` - Hierarchical clustering of syllable similarity
6. `interactive_syllable_stats()` - Bar charts with statistical tests (Kruskal-Wallis, Dunn's, Mann-Whitney, etc.)
7. `interactive_crowd_movie_comparison()` - Side-by-side group comparisons
8. `interactive_transition_graph()` - Network visualization with interactive thresholds

**Architecture:** Model-View-Controller pattern adapted for Jupyter:
- Each feature area (`roi/`, `scalars/`, `stat/`, `viz/`, `flip/`) has its own `controller.py`, `widgets.py`, and optionally `view.py`
- Uses Bokeh for interactive server-side plots, Plotly for client-side, ipywidgets for controls

**Module layout:**
```
moseq2_app/
├── main.py             # 10 public API functions (primary entry point)
├── util.py             # Config, YAML, dataframe utilities
├── gui/
│   ├── progress.py     # Session discovery, path management
│   ├── widgets.py      # GroupSettingWidgets (QGrid table)
│   └── wrappers.py     # High-level workflow wrappers
├── roi/                # Extraction validation and viewing
├── scalars/            # Scalar visualization (violin plots)
├── stat/               # Syllable statistics + transition graphs
├── viz/                # Syllable labeling + crowd movie comparison
└── flip/               # Flip classifier training
```

---

## Key Data Files

| File | Created By | Consumed By | Format |
|------|-----------|-------------|--------|
| `results_00.h5` | extract | pca, viz, app | HDF5 (frames, masks, scalars, metadata) |
| `results_00.yaml` | extract | app | YAML (extraction status) |
| `moseq2-index.yaml` | extract | model, viz, app | YAML (session manifest with metadata + groups). See [moseq2-index-yaml-lifecycle.md](moseq2-index-yaml-lifecycle.md) for full lifecycle: schema, all producers/consumers, source-of-truth analysis, and recovery procedures. See [group-data-flow.md](group-data-flow.md) for how group labels propagate between the index file and model pickle, and how conflicts are resolved at visualization time. |
| `pca.h5` | pca | model | HDF5 (components, singular values, variance) |
| `pca_scores.h5` | pca | model | HDF5 (PC scores per session) |
| `changepoints.h5` | pca | viz | HDF5 (model-free syllable boundaries) |
| Model file (`.p`) | model | viz, app | Pickle (AR-HMM, labels, whitening params) |
| `syll_info.yaml` | app | app | YAML (syllable labels and descriptions) |
| `syll_df.parquet` | app | app | Parquet (cached syllable statistics) |

## Relationship to Keypoint-MoSeq

| | MoSeq2 (depth) | Keypoint-MoSeq (keypoints) |
|---|---|---|
| **Input** | Raw depth video (Kinect/Azure/RealSense) | Pose tracking (SLEAP/DLC) keypoint coordinates |
| **Preprocessing** | Background subtraction, ROI, crop+rotate | Outlier removal, egocentric alignment |
| **Dimensionality reduction** | PCA on depth frames (separate package) | PCA on pose vectors (built-in) |
| **Model** | AR-HMM via pyhsmm (Gibbs sampling) | AR-HMM via JAX (gradient-based) |
| **Architecture** | 5 separate packages + CLI | Single package + Python API |
| **Fitting** | Bayesian Gibbs sampling | JAX-based optimization |
| **GPU** | Not required | JAX with optional GPU |
| **Python** | 3.6-3.7 (legacy) | 3.10 |

## Working Rules

- **Never commit `docs/plans/`** to any moseq2 repo. Design docs and plans should stay local or in the skills/memory system, not in the repositories.
