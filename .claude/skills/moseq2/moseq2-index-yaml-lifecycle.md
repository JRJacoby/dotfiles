# moseq2-index.yaml Lifecycle

Central manifest file linking all extracted sessions to their metadata, file paths, group assignments, and PCA scores path. Acts as the glue between pipeline stages.

## Schema

```yaml
files:                                    # list — one entry per extracted session
  - uuid: "xxxxxxxx-xxxx-..."             # str — session UUID (generated at extraction time)
    group: "default"                      # str — experimental group label ("default" at creation)
    path:                                 # list[str, str] — relative to index file location
      - "relative/path/to/results_00.h5"  #   [0] extraction HDF5
      - "relative/path/to/results_00.yaml"#   [1] extraction YAML
    metadata:                             # dict — acquisition metadata from HDF5
      SessionName: "session_2020..."
      SubjectName: "mouse1"
      StartTime: "..."
      # ... any other keys from /metadata/acquisition in h5
pca_path: ""                              # str — path to pca_scores.h5 (empty at creation)
```

## Origin

The index is created **after** extraction. Extraction happens first (per-session), producing `results_00.h5` + `results_00.yaml` pairs each with a `uuid.uuid4()` baked in. The index is then generated as a summary/registry of what's already been extracted.

UUIDs are assigned once during extraction at `moseq2_extract/helpers/wrappers.py:361`:
```python
status_dict = {
    "complete": False,
    "skip": False,
    "uuid": str(uuid.uuid4()),  # generated once, stored in both .yaml and .h5
    "metadata": "",
    "parameters": deepcopy(config_data),
}
```

The UUID is stored in two places per session: the per-session `results_00.yaml` (top-level `uuid` key) and the HDF5 at `/metadata/uuid`. The index reads from the YAML files — it never creates UUIDs itself.

**Regenerating the index produces identical UUIDs and file paths** because they're read from the already-existing per-session files. Only group assignments (reset to `"default"`) and `pca_path` (reset to `""`) are lost.

## Write Operations (Producers)

Every writer does a **full file overwrite** — there is no incremental/append logic anywhere. The entire YAML is read into a dict, modified in memory, and written back whole.

### Initial Creation (moseq2-extract)

| Function | File | Trigger |
|----------|------|---------|
| `generate_index_wrapper()` | `moseq2_extract/helpers/wrappers.py:84` | CLI: `moseq2-extract generate-index`, GUI: `generate_index_command()` |
| `generate_index_from_agg_res_wrapper()` | `moseq2_extract/helpers/wrappers.py:181` | CLI: `moseq2-extract agg-to-index` |
| `aggregate_extract_results_wrapper()` | `moseq2_extract/helpers/wrappers.py:129` | CLI: `moseq2-extract aggregate-results` (delegates to `generate_index_wrapper`) |

**How `generate_index_wrapper` works:**
1. `recursive_find_h5s(input_dir)` — globs for all `.h5` files, filters to those that have a matching `.yaml` AND contain a `frames` dataset. Sessions missing either file are **silently skipped** (no warning).
2. Ensures metadata exists in each session's YAML via `copy_h5_metadata_to_yaml_wrapper()` (pulls from HDF5 if missing)
3. `build_index_dict()` — constructs the dict, deduplicates by UUID, sets `group: "default"` and `pca_path: ""`
4. Writes with `open(output_file, "w")` + `yaml.safe_dump()` (full overwrite)

### Group Assignment Updates

| Function | Package | File | Trigger |
|----------|---------|------|---------|
| `add_group_wrapper()` | moseq2-viz | `moseq2_viz/helpers/wrappers.py:87` | CLI: `moseq2-viz add-group`, GUI: `add_group()` |
| `GroupSettingWidgets.update_clicked()` | moseq2-app | `moseq2_app/gui/widgets.py:81` | User clicks "Update Index File" in notebook QGrid widget |

`add_group_wrapper` is the only writer that uses **atomic writes** (writes `_update.yaml` then `shutil.move`). All others use direct `open("w")`.

### PCA Path Updates

| Function | Package | File | Trigger |
|----------|---------|------|---------|
| `apply_pca_command()` | moseq2-pca | `moseq2_pca/gui.py:49` | After PCA scores computed (notebook GUI only) |
| `find_progress()` | moseq2-app | `moseq2_app/gui/progress.py:267` | Automatic during notebook progress discovery |

These only modify the `pca_path` field, but still rewrite the entire file.

## Read Operations (Consumers)

The file is read **on-demand** each time a function needs it — never cached globally. Parsed results may be stored as instance variables on controller objects for the lifetime of a widget.

### Core Parsing Chain (moseq2-viz)

`parse_index()` at `moseq2_viz/util.py:217` is the primary parser used across viz and app. It:
1. Loads the YAML
2. Groups `files` by UUID into a dict (`{uuid: {group, path, metadata}}`)
3. Joins relative paths with the index file's directory to make them absolute
4. Returns both raw index and UUID-sorted index

`get_sorted_index()` at `moseq2_viz/util.py:265` is a thin wrapper returning just the sorted dict.

### By Pipeline Stage

| Stage | Package | Function | What It Reads |
|-------|---------|----------|--------------|
| Group editing | app | `index_to_dataframe()` (`moseq2_app/util.py:79`) | `files`, `metadata` → DataFrame for QGrid table |
| Scalar summary | app+viz | `scalars_to_dataframe()` (`moseq2_viz/scalars/util.py:393`) | `path[0]`, `group`, UUID, `metadata` keys |
| Syllable labeling | app+viz | `SyllableLabeler.__init__` (`moseq2_app/viz/controller.py:66`) | Sorted index for UUID matching + crowd movies |
| Syllable stats | app+viz | `InteractiveSyllableStats` (`moseq2_app/stat/controller.py:138`) | UUIDs, groups for model-index matching |
| Transition graphs | app+viz | `InteractiveTransitionGraph` (`moseq2_app/stat/controller.py:313`) | UUIDs, groups |
| Dendrogram | app+viz | `plot_dendrogram()` (`moseq2_app/stat/view.py:1306`) | Sorted index |
| Crowd movies | viz | `make_crowd_movies_wrapper()` (`moseq2_viz/helpers/wrappers.py:461`) | `path[0]` (h5 files), `pca_path` |
| Video params | viz | `check_video_parameters()` (`moseq2_viz/io/video.py:22`) | `path[1]` (extraction yamls) |
| Model training | model | `process_indexfile()` (`moseq2_model/helpers/data.py:20`) | `uuid`, `group` (optionally `SubjectName`/`SessionName` for display) |
| Session selection | model | `select_data_to_model()` (`moseq2_model/helpers/data.py:74`) | `uuid`, `group` |
| Progress tracking | app | `find_progress()` (`moseq2_app/gui/progress.py:269`) | `pca_path` |

## Source of Truth Analysis

| Data | Sole Source of Truth? | Notes |
|------|-----------------------|-------|
| Session registry | No | Derived from HDF5+YAML files on disk via `recursive_find_h5s()` |
| Session UUID | No | Stored in per-session `results_00.yaml` and `results_00.h5` |
| Session metadata | No | Stored in HDF5 `/metadata/acquisition` |
| File paths | No | Discovered from filesystem |
| **Group assignments** | **YES** | Only stored here. Lost if deleted. A snapshot is also saved in the model pickle at training time — see [group-data-flow.md](group-data-flow.md) for how conflicts between the index and model are resolved. |
| PCA scores path | Weak | Discoverable from filesystem (`_pca/pca_scores.h5`) via `find_progress()` |
| Processing status | No | Not tracked here — tracked in per-session `results_00.yaml` (`complete`/`skip` flags) and `progress.yaml` |

The file does **not** act as a processing status registry. It does not track which sessions have been "extracted", "PCA'd", or "modeled".

## Recovery

If the file is accidentally deleted:

```bash
# Step 1: Regenerate from per-session extraction files
# Recovers: UUIDs, paths, metadata. Sets all groups to "default".
cd /path/to/project
moseq2-extract generate-index -i . -o moseq2-index.yaml

# Step 2: Re-assign groups (MANUAL — only irreplaceable data)
# Option A: CLI
moseq2-viz add-group moseq2-index.yaml --key SubjectName --value "mouse1" --group control
moseq2-viz add-group moseq2-index.yaml --key SubjectName --value "mouse2" --group treatment

# Option B: Notebook
# Run interactive_group_setting() and edit the QGrid table

# Step 3: Update PCA path
# Option A: Automatic — run any moseq2-app notebook cell; find_progress() auto-detects
# Option B: Manual — edit YAML, set pca_path to relative path to pca_scores.h5
```

**Permanently lost:** Group assignments (the only truly irreplaceable data).
**Automatically recoverable:** Session list, UUIDs, file paths, metadata, PCA path.

## Missing Index Behavior by Package

| Package | Context | Behavior |
|---------|---------|----------|
| moseq2-viz CLI | `click.argument(..., exists=True)` | Click rejects before function runs |
| moseq2-viz code | `parse_index()` calls `open()` | `FileNotFoundError` propagates |
| moseq2-model CLI | Default `--index ""` | Falls back to h5 groups, each session gets unique integer group |
| moseq2-model GUI | `assert exists(index)` | `AssertionError` |
| moseq2-pca GUI | `read_yaml()` catches `IOError` | Prints warning, continues without updating |
| moseq2-app | `find_progress()` | `index_file` stays as `""`, downstream functions print error and return |

## Data Flow Diagram

```
Raw Depth Videos
       |
       v
 [moseq2-extract]  ── per session ──>  results_00.h5 + results_00.yaml
       |                                     (uuid baked in at extraction)
       |
       | generate-index (scans all sessions)
       v
 moseq2-index.yaml  <-- CREATED (uuid, paths, metadata from per-session files; group="default"; pca_path="")
       |
       +-- [moseq2-app]  -------- WRITES group assignments (QGrid widget)
       +-- [moseq2-viz]  -------- WRITES group assignments (add-group CLI)
       +-- [moseq2-pca]  -------- WRITES pca_path (after PCA scores computed)
       +-- [moseq2-app]  -------- WRITES pca_path (find_progress auto-discovery)
       |
       +-- [moseq2-model] ------- READS uuid + group (session selection, training)
       +-- [moseq2-viz]  -------- READS everything (crowd movies, stats, graphs)
       +-- [moseq2-app]  -------- READS everything (all interactive widgets)
```
