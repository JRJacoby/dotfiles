# Group Data Flow Across the MoSeq2 Pipeline

How experimental group labels propagate through training, application, and visualization — and what happens when group assignments change after model training.

## Where Groups Are Stored

| Location | Format | When Written |
|----------|--------|--------------|
| `moseq2-index.yaml` | `files[].group` (str) | Group assignment via `moseq2-viz add-group` or notebook QGrid widget |
| Model pickle (`.p`) | `metadata.groups` (dict: `{uuid: group}`) | Snapshot at `learn-model` / `apply-model` time |

The index file is the **editable source of truth**. The model pickle contains a **frozen snapshot** from training time.

## How Groups Enter the Model Pickle

### learn-model

At `moseq2_model/helpers/wrappers.py:77-102`:
1. PC scores loaded from `pca_scores.h5`
2. `process_indexfile()` reads `uuid` + `group` from the index file into `data_metadata["groups"]`
3. Groups are packed into `data_metadata` and saved in the export dict at line 196: `"metadata": data_metadata`

Groups are also copied into `model_parameters["groups"]` **only if `--separate-trans` is used** (`data.py:200-201`). Without that flag, `model_parameters` has no group info.

**Effect on training:** Groups do NOT affect the AR-HMM fitting unless `--separate-trans` is passed (off by default). Without it, all sessions share one transition matrix regardless of group labels. The groups in `metadata` are purely bookkeeping.

### apply-model

At `moseq2_model/helpers/wrappers.py:252-295`:
1. Loads the pre-trained model pickle
2. Loads new PC scores
3. Reads groups from the index file via `process_indexfile()` into `data_metadata`
4. Saves the applied model with `"metadata": data_metadata` (line 286) — this captures the **current** index file groups, not the original training groups
5. Copies `model_parameters` from the original model (line 291-292) — if `separate_trans` was used, this still contains the **training-time** groups

## How Groups Are Resolved at Visualization Time

The critical merge happens in `scalars_to_dataframe()` at `moseq2_viz/scalars/util.py:393`.

This function builds the analysis DataFrame by combining:
- **Model data** via `prepare_model_dataframe()` — reads `mdl['metadata']['groups'][k]` (line 881 of `model/util.py`)
- **Index data** via the sorted index — reads `v['group']` (line 452 of `scalars/util.py`)

### Conflict Resolution (lines 473-483)

When the group label for a UUID differs between the model pickle and the index file:

| `--separate-trans` used? | Winner | Behavior |
|--------------------------|--------|----------|
| **No** (default) | **Index file** | Prints notice: "Overwriting group label with those from the index.yaml file." Drops the model's group column. |
| **Yes** | **Model pickle** | Warns: "Group labels from index.yaml and model results do not match! Setting group labels to ones used in the model." Drops the index's group column. |

### Why `separate-trans` differs

When `--separate-trans` is on, the model has structurally different transition matrices per group. The syllable labels were computed using group-specific transition dynamics, so the group assignments are mathematically coupled to the labels. Changing groups post-hoc would make the labels inconsistent with the model structure.

## Practical Implications

### Reassigning groups after training (default case, no `--separate-trans`)

**Safe to do.** Just edit the index file and re-run visualization. The code will:
1. Print a notice about the mismatch for each affected session
2. Use the index file's (new) group assignments
3. All downstream stats, crowd movies, transition graphs will use the new groups

No retraining needed. The syllable labels are group-independent.

### Reassigning groups after training (with `--separate-trans`)

**Requires retraining.** The model's transition matrices are group-specific, and the viz code will ignore your index file changes in favor of the model's groups.

### What the printed notices look like

For each session whose group changed, you'll see:
```
Group name for UUID <uuid> in the index.yaml file does not match the group name in the model file.
Overwriting group label with those from the index.yaml file.
```
This is expected and harmless when intentionally reassigning groups.
