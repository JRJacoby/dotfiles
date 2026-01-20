---
name: snub
description: Create and configure SNUB (Systems Neuro Browser) projects for visualizing synchronized video, neural data, pose tracking, and behavioral annotations. Use when working with multi-modal neuroscience data visualization.
---

# SNUB (Systems Neuro Browser)

SNUB is a visual interface for systems neuroscience that enables exploration of relationships between multiple synchronized data streams including video, pose tracking, behavioral annotations, and neural recordings.

## User Preferences

- **Always use `copy=True`** when adding videos (creates portable, self-contained projects)
- **Always use `add_traceplot=True`** when adding heatmaps (enables interactive row selection)

## Installation

```bash
pip install systems-neuro-browser
# or
uv add systems-neuro-browser
```

## Core Concepts

- **Project**: A directory containing `config.json` and associated data files
- **Timeline**: Interactive timeline synchronizing all data views to a common time reference (in seconds)
- **Tracks**: Time-series visualizations (heatmaps, traceplots) stacked vertically
- **Panels**: Non-time-series views (videos, scatter plots) displayed alongside tracks
- **Data Views**: Individual visualizations synced to the timeline

## Running SNUB

```bash
# Command line
snub /path/to/project

# Python
import snub
snub.open('/path/to/project')
```

---

## Python API (`snub.io`)

### Creating a Project

```python
import snub.io

snub.io.create_project(
    project_directory,        # Path where project will be created
    duration=None,            # Timeline duration in seconds (or use end_time)
    end_time=None,            # Timeline end in seconds (alternative to duration)
    start_time=0,             # Timeline start in seconds
    layout_mode='columns',    # 'columns' (panels left, tracks right) or 'rows'
    min_step=1/30,            # Time discretization (typically 1/fps)
    overwrite=False,          # Overwrite existing config.json
)
```

### Adding Video

```python
snub.io.add_video(
    project_directory,
    videopath,                # Path to video file (must be 8-bit RGB)
    name=None,                # Display name (defaults to filename stem)
    copy=True,                # PREFERRED: copy video into project directory
    fps=None,                 # Framerate (inferred from video if None)
    start_time=0,             # Video start time in seconds
    timestamps=None,          # Array or path to .npy/.txt of frame timestamps
    order=0,                  # Placement order in panel stack
    initial_visibility=True,  # Whether visible when project opens
)
```

**Notes:**
- Videos must be 8-bit RGB. Use `snub.io.video.transform_azure_ir_stream()` for 16-bit IR conversion.
- If `timestamps=None`, timestamps are generated from `fps` and `start_time`.
- **Always use `copy=True`** for portable, self-contained projects.

### Adding Heatmap

```python
snub.io.add_heatmap(
    project_directory,
    name,                     # Display name
    data,                     # 2D array: shape (n_rows, n_time_bins)

    # Time specification (choose one approach):
    binsize=None,             # Time per column in seconds (e.g., 1/60 for 60fps)
    start_time=None,          # Start time of first column in seconds
    # OR:
    time_intervals=None,      # (N, 2) array of [start, end] for each column

    # Row configuration
    labels=None,              # List of row labels (defaults to row indices as strings)
    sort_method=None,         # 'rastermap', array of indices, or None (keep order)

    # Appearance
    colormap='viridis',       # Any matplotlib colormap name
    vmin=None,                # Colormap floor (default: 1st percentile)
    vmax=None,                # Colormap ceiling (default: 99th percentile)

    # Label display
    initial_show_labels=True,
    label_color=(255, 255, 255),
    label_font_size=12,
    max_label_width=300,

    # Traceplot integration
    add_traceplot=True,       # PREFERRED: add interactive traceplot for selected rows
    heatmap_height_ratio=2,   # Relative height of heatmap portion
    trace_height_ratio=1,     # Relative height of traceplot portion

    # Layout
    height_ratio=1,           # Relative height in track stack
    order=0,                  # Placement order
    initial_visibility=True,
)
```

**Data format:** Rows are variables (neurons, PCs, etc.), columns are time bins. Each column corresponds to a time interval.

**Colormap suggestions:**
- `'viridis'` - sequential data (default)
- `'coolwarm'` - diverging data with positive/negative values
- `'magma'`, `'plasma'` - other sequential options
- `'RdBu'`, `'seismic'` - other diverging options

### Adding Traceplot (Standalone)

```python
snub.io.add_traceplot(
    project_directory,
    name,
    traces,                   # Dict: {trace_name: (N, 2) array of [time, value]}
    linewidth=1,
    trace_colors={},          # Dict: {trace_name: (R, G, B) tuple}
    height_ratio=1,
    order=0,
    initial_visibility=True,
)
```

### Adding Scatter Plot

```python
snub.io.add_scatter(
    project_directory,
    name,
    data,                     # (N, 2) array of [x, y] coordinates
    times=None,               # (N,) array of timestamps for each point
    colormap='viridis',
    size=3,
    order=0,
)
```

Useful for UMAP/t-SNE embeddings where points can be colored by time or selected by timeline position.

### Adding Spike Raster / Rate Heatmap

```python
snub.io.add_spikeplot(
    project_directory,
    name,
    spike_times,              # List of arrays, one per neuron
    spike_labels,             # List of neuron labels
    window_size=0.5,          # Sliding window for rate calculation
    window_step=0.05,         # Step size for sliding window
    colormap='magma',
)
```

### Adding 3D Pose

```python
snub.io.add_pose3D(
    project_directory,
    name,
    poses,                    # (n_frames, n_keypoints, 3) array
    links=None,               # List of (i, j) tuples for skeleton connections
    fps=30,
    start_time=0,
)
```

### Adding Annotator

```python
snub.io.add_annotator(
    project_directory,
    name,
    labels,                   # List of annotation label names
    colors=None,              # List of (R, G, B) tuples for each label
)
```

Creates an interactive annotation widget for labeling time intervals.

### Adding ROI Plot (Video Overlay)

```python
snub.io.add_roiplot(
    project_directory,
    name,
    rois,                     # Dict: {roi_name: binary mask array}
    video_name,               # Name of video to overlay on
    colors=None,
)
```

---

## Modifying Projects

### Edit Configuration

```python
# Edit global settings
snub.io.edit_global_config(project_directory, duration=2000, layout_mode='rows')

# Edit specific data view
snub.io.edit_dataview_properties(
    project_directory,
    dataview_type='heatmap',  # 'video', 'heatmap', 'traceplot', etc.
    name='my_heatmap',
    colormap='plasma',
    vmin=-2,
    vmax=2,
)
```

### Remove Data View

```python
snub.io.remove_dataview(
    project_directory,
    dataview_type='heatmap',
    name='my_heatmap',
    delete_data=True,         # Also delete associated data files
)
```

### Add Timeline Markers

```python
snub.io.set_markers(
    project_directory,
    times=[10.5, 25.0, 100.2],           # Event times in seconds
    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # RGB colors
)
```

---

## Data Processing Utilities

### Bin Data

```python
binned = snub.io.bin_data(
    data,                     # (n_variables, n_timepoints) array
    timestamps,               # (n_timepoints,) array
    binsize=0.1,              # Bin width in seconds
    agg_func=np.mean,         # Aggregation function
)
```

### Sort Neurons (Rastermap)

```python
row_order = snub.io.sort(data, method='rastermap')
```

### UMAP Embedding

```python
embedding = snub.io.umap_embedding(
    data,                     # (n_variables, n_timepoints) array
    n_pcs=50,                 # PCA components before UMAP
    n_neighbors=15,
    min_dist=0.1,
)
```

### Generate Video Timestamps

```python
from snub.io.video import generate_video_timestamps

timestamps = generate_video_timestamps(videopath, fps=None, start_time=0)
```

---

## Project File Structure

A SNUB project directory contains:

```
project_directory/
├── config.json                        # Project configuration
├── video_name.timestamps.npy          # Video frame timestamps
├── heatmap_name.heatmap_data.npy      # Heatmap data array
├── heatmap_name.heatmap_intervals.npy # Time intervals for columns
├── heatmap_name.heatmap_labels.txt    # Row labels (one per line)
├── heatmap_name.heatmap_row_order.npy # Row ordering indices
├── traceplot_name.trace_data.npy      # Traceplot data
└── ...
```

---

## Common Patterns

### Synchronized Video + Neural Heatmap

```python
import snub.io
import numpy as np

fps = 30
n_frames = 54000  # 30 minutes at 30fps
duration = n_frames / fps

# Create project
snub.io.create_project('my_project', duration=duration, min_step=1/fps)

# Add video (with copy=True for portability)
snub.io.add_video('my_project', 'video.mp4', fps=fps, copy=True)

# Add neural data heatmap (n_neurons x n_frames)
snub.io.add_heatmap(
    'my_project',
    name='neural_activity',
    data=neural_data,  # shape: (n_neurons, n_frames)
    binsize=1/fps,
    start_time=0,
    labels=[f'neuron_{i}' for i in range(n_neurons)],
    sort_method='rastermap',
    colormap='magma',
    add_traceplot=True,  # Always include traceplot
)
```

### Behavioral Embedding Visualization

```python
# Add UMAP scatter colored by time
snub.io.add_scatter(
    'my_project',
    name='behavior_umap',
    data=umap_coords,  # (n_frames, 2)
    times=np.arange(n_frames) / fps,
)
```

### Multiple Aligned Videos

```python
# Same timestamps for synchronized cameras
timestamps = np.arange(n_frames) / fps

snub.io.add_video('my_project', 'camera1.mp4', name='front', timestamps=timestamps, copy=True)
snub.io.add_video('my_project', 'camera2.mp4', name='side', timestamps=timestamps, copy=True)
```

---

## Tips

1. **Time alignment**: Ensure `binsize=1/fps` and matching `start_time` for synchronized video and heatmap.

2. **Colormaps for different data types**:
   - Neural firing rates: `'magma'` or `'viridis'`
   - Z-scored data: `'coolwarm'` or `'RdBu'`
   - Binary/categorical: `'Set1'` or custom

3. **Large datasets**: For very long recordings, SNUB handles data efficiently, but consider downsampling for initial exploration.

4. **Row sorting**: Use `sort_method='rastermap'` for neural data to cluster correlated neurons. For ordered data (like PCs), keep original order with `sort_method=None`.

5. **NaN handling**: SNUB displays NaN values as transparent in heatmaps. Consider interpolating or replacing with 0 depending on your needs.

6. **Portable projects**: Always use `copy=True` when adding videos.

7. **Interactive traces**: With `add_traceplot=True` on heatmaps, click rows in SNUB to add/remove traces from the plot.
