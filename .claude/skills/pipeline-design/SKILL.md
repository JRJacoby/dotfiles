---
name: pipeline-design
description: Design and implement data analysis pipelines. Use when creating scripts that process data through sequential steps, batch-process sessions/files, or build multi-stage analysis workflows.
---

# Pipeline Design Conventions

These are the user's requirements for how analysis pipeline scripts should be structured. Follow them exactly.

## Style Consistency

Match existing project conventions for visual style (colors, fonts, line widths, etc.) when they exist. Search the codebase for established patterns before introducing new ones. When no convention exists, define one explicitly as a module-level constant so future scripts can reuse it.

## Script Structure

Every pipeline script follows this layout, in order:

1. **Imports** at the top
2. **Constants and options** (paths, parameters, configuration)
3. **Helper/reusable function definitions**
4. **Pipeline step functions** (one function per logical step)
5. **`main()` function** that calls all step functions in sequence
6. **`if __name__ == "__main__"` block** that calls `main()`

## Code Reuse Between Steps

When multiple steps share the same core logic (e.g., the same plot with different filters), extract the shared logic into a **helper function** with parameters for what varies. Steps stay argument-free and call the helper internally.

```python
# Helper (in the helpers section)
def _plot_usage_bar(genotypes: list[str], output_path: Path):
    """Shared logic for usage bar charts."""
    ...

# Steps (no arguments, no return)
def plot_usage_all():
    _plot_usage_bar(["WT", "HET", "KO"], OUTPUT_DIR / "usage_all.png")

def plot_usage_wt_ko():
    _plot_usage_bar(["WT", "KO"], OUTPUT_DIR / "usage_wt_ko.png")
```

## Step Function Design

### No arguments, no return values

Each step function takes **no arguments** and **returns nothing**. Steps communicate exclusively through files on disk — a step loads its own inputs (raw data or outputs from previous steps) and saves its own outputs.

```python
# CORRECT
def extract_features():
    sessions = load_csv(INPUT_TABLE)
    results = process(sessions)
    results.to_csv(FEATURES_CSV)

# WRONG - steps do not pass data through parameters
def extract_features(sessions):
    return process(sessions)
```

### No step numbers in function names

Name functions by what they do, not their position in the pipeline. Step order changes; names shouldn't.

```python
# CORRECT
def count_samples():
def extract_features():
def fit_model():

# WRONG
def step1_count_samples():
def step2_extract_features():
```

## Docstrings

Every function — helpers and pipeline steps alike — must have a docstring. Docstrings serve two audiences: (1) someone reading the function signature who needs the technical contract (inputs, outputs, types, side effects), and (2) a coworker coming to the project later who needs to understand **why this function exists** in the context of the project's scientific or analytical goals.

### Required content

1. **Purpose and motivation** (first line or short paragraph): What does this function accomplish, and why does the project need it? Connect it to the analytical goal — not just "processes sessions" but *why* those sessions need this processing.
2. **Parameters**: Name, type, and meaning of each parameter. For complex types (DataFrames, dicts), describe the expected schema or keys.
3. **Returns**: Type and meaning of the return value. For step functions that return nothing, this can be omitted.
4. **Side effects**: Any files written to disk (with path patterns), logging, or state changes. For step functions this is the primary output mechanism, so be explicit about what gets saved and where.
5. **Inputs read from disk** (for step functions): Which module-level constants or files the function reads, since steps take no arguments.

### When the agent lacks context

If the agent writing the pipeline does not have enough context to write the "why" portion of a docstring — e.g., it doesn't know the scientific motivation behind a step — it must **stop and ask the user** rather than writing a vague or generic placeholder. A docstring that says "Process the data" is worse than useless; it gives the illusion of documentation while communicating nothing.

### Examples

```python
# Helper
def _plot_usage_bar(genotypes: list[str], output_path: Path):
    """Plot syllable usage frequencies as a bar chart for the given genotypes.

    We compare usage distributions across genotypes to identify syllables
    whose frequency differs between groups — the primary behavioral readout
    from the MoSeq model.

    Parameters
    ----------
    genotypes : list[str]
        Genotype labels to include (e.g., ["WT", "KO"]). Must match values
        in the 'genotype' column of the session table.
    output_path : Path
        Where to save the PNG figure.

    Side effects
    ------------
    Saves a PNG plot to `output_path`.
    Saves a stats sidecar JSON to `output_path` with suffix `_stats.json`.
    """
    ...


# Step function
def fit_model():
    """Fit the AR-HMM to aligned pose sequences and save the model checkpoint.

    This is the core unsupervised segmentation step: the AR-HMM discovers
    recurring movement patterns (syllables) from continuous pose trajectories.
    Downstream steps use the learned model to label frames and compare
    syllable usage across genotypes.

    Reads
    -----
    ALIGNED_SEQUENCES_H5 : Path
        HDF5 file of aligned pose sequences produced by `align_sequences()`.
    MODEL_CONFIG : Path
        YAML config specifying kappa, gamma, and number of AR lags.

    Side effects
    ------------
    Saves the fitted model to `MODEL_DIR / "model.p"`.
    Saves per-frame state labels to `MODEL_DIR / "labels.h5"`.
    """
    ...
```

## Idempotency

The combination of caching, resumability, and atomic writes makes every pipeline run idempotent. If a run is interrupted and restarted, it picks up where it left off and produces the same result as a clean run.

### Output caching (skip-if-fresh)

Every step checks whether its outputs already exist **AND are newer than all of its inputs**. If so, skip the step entirely.

```python
def some_step():
    inputs = [INPUT_TABLE, CONFIG_FILE]
    outputs = [OUTPUT_CSV]
    if all_fresh(outputs, inputs):
        print("some_step: outputs are fresh, skipping")
        return
    # ... do work ...
```

Write a shared `all_fresh(outputs, inputs)` helper that checks:
- All output files exist
- All output files are newer than all input files

Note: `all_fresh` does **not** check against the pipeline script itself. If the user changes pipeline code and wants to re-run, they delete stale outputs manually. The caching system only tracks data dependencies, not code dependencies.

### Resumable loops

If a step loops over items (e.g., processing each session), each item's output gets its own freshness check. If a long step is interrupted, only unfinished items re-run on restart.

```python
def process_sessions():
    sessions = load_csv(INPUT_TABLE)
    for _, row in sessions.iterrows():
        output_path = OUTPUT_DIR / f"{row['session_id']}.h5"
        if all_fresh([output_path], [row['input_path']]):
            continue
        # ... process this session ...
```

### Atomic writes

All outputs must be written atomically. Write to a temporary file, then rename to the final path in one operation. This prevents partial/corrupt outputs from being treated as complete.

```python
tmp = output_path.with_name("_tmp_" + output_path.name)
# Clean up any leftover tmp from a previous interrupted run
if tmp.exists():
    tmp.unlink()
write_data(tmp)
tmp.rename(output_path)  # atomic on same filesystem
```

For atomic writes, the pattern is:
1. At the start of processing an item, delete any existing `.tmp` file for that item
2. Write to the `.tmp` path (preserving the original extension)
3. Rename `.tmp` to the final path as the last operation

**Important — matplotlib and file extensions:** matplotlib infers the output
format from the file extension. The temp file MUST keep the same extension as
the final file. Use `output_path.with_name("_tmp_" + output_path.name)` which
prepends `_tmp_` to the filename while preserving the extension (e.g.,
`plots/foo.png` → `plots/_tmp_foo.png`). Do NOT use:
- `with_suffix(".tmp")` — changes `foo.png` to `foo.tmp` (matplotlib error)
- `with_suffix(".png.tmp")` — changes `foo.png` to `foo.png.tmp` (same error)
- `with_name(name + ".tmp")` — gives `foo.png.tmp` (same error)

Freshness checks use only the final output path, never the `.tmp` path.

## Logging

All log messages must include **timestamps** and be written to a **log file**. Each pipeline run creates a new log file named with the **timestamp of when the run started** (e.g., `2026-03-18_14-30-05.log`). Configure logging once at the top of `main()` and use it throughout.

**Flush on every write.** Python buffers stdout when output is redirected (e.g., background execution, piped to a file). This makes it impossible to monitor progress in real time. Always ensure log output is unbuffered:
- If using `print()`: pass `flush=True` on every call
- If using `logging`: add `stream=sys.stdout` to `basicConfig` and call `sys.stdout.flush()` after each handler, or use a handler with `flush` support
- Alternatively, run with `PYTHONUNBUFFERED=1` environment variable

```python
from datetime import datetime
import logging

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)
    ...
```

Within loops (at per-item granularity, e.g., per-session — not per-frame), log each item with a **current/total counter**:

```python
for i, row in enumerate(sessions):
    log.info(f"process_sessions: [{i+1}/{len(sessions)}] {row['session_id']}")
```

When processing large data in chunks (e.g., streaming video frames in batches), log each chunk with a chunk/total counter:

```python
for chunk_idx in range(n_chunks):
    frames = read_chunk(...)
    process(frames)
    log.info(f"process_sessions: [{i+1}/{n}] {session_id} chunk [{chunk_idx+1}/{n_chunks}]")
```

Log when a step starts, when it skips (outputs fresh), and when items within a loop are processed or skipped.

## Output Directories

Steps create their own output directories as needed (`makedirs(exist_ok=True)`). Do not require directories to exist beforehand.

## Plot Outputs

Every plot writes three artifacts side by side, sharing the plot's basename:

```
plots/
  state_frequencies_by_genotype.pdf
  state_frequencies_by_genotype_stats.json
  state_frequencies_by_genotype_data.csv
```

All three are tracked by the freshness check — deleting any one triggers a rebuild — and are written atomically (per the atomic-write rule above).

### File format

Plots are saved as **PDF** (vector). PDF preserves text, line widths, and colors at any zoom and is the assumed input for downstream editing (Illustrator, Figma, etc.); a rasterized format alone loses that.

### Stats sidecar (`_stats.json`)

The stats sidecar captures the numerical content the plot depicts in machine-readable form. The intent: anything one might want to ask about the plot later — exact mean, CI, sample size, fitted parameter, threshold, boundary, p-value — should be readable from the sidecar without re-running the pipeline or eyeballing the figure. At minimum it should capture every quantity the plot actually visualizes, plus the sample sizes underlying each summary statistic.

What goes in depends on the plot type:

#### Group-comparison plots

For plots that compare groups (e.g., bar charts with multiple categories), include:

- All pairwise comparisons with test statistic, raw p-value, corrected p-value (if any correction was applied), and significance flag.
- The statistical test used and the correction method.
- Sample sizes per group.
- Summary statistics (means, medians, CIs) per group.

```json
{
  "test": "bootstrap_permutation",
  "correction": "FDR",
  "alpha": 0.05,
  "n_bootstrap": 10000,
  "comparisons": [
    {
      "groups": ["WT", "HET"],
      "n": [10, 10],
      "means": [0.32, 0.28],
      "ci_95": [[0.28, 0.36], [0.24, 0.32]],
      "statistic": 0.04,
      "p_value": 0.023,
      "p_corrected": 0.069,
      "significant": false
    }
  ]
}
```

#### Diagnostic / model-fit plots

For plots showing a fitted model, distribution, or threshold (no comparison being made), include:

- The fitted parameters (e.g., per-component means / variances / weights for a mixture model, regression coefficients for a fit line, kernel bandwidth for a KDE).
- Any thresholds or boundaries drawn on the plot, with their numeric values.
- Sample size and basic summary statistics of the input data (n, mean, std, percentiles).

```json
{
  "n_input": 247,
  "input_summary": {"mean": 12.4, "std": 5.1, "min": 1.0, "max": 38.7},
  "model": {
    "name": "GaussianMixture",
    "n_components": 2,
    "components": [
      {"label": "Slow", "mean": 7.8, "std": 2.1, "weight": 0.62},
      {"label": "Fast", "mean": 18.5, "std": 3.4, "weight": 0.38}
    ]
  },
  "boundary": {"value": 13.15, "rule": "midpoint of component means"}
}
```

#### Single-series plots (histograms, time series)

For plots with no model fit and no group comparison, the sidecar still records:

- Sample size.
- Bin edges + counts (for histograms) or summary statistics (mean, std, percentiles) for the series.
- Any reference lines, thresholds, or annotations and their numeric values.

### Data CSV (`_data.csv`)

Alongside the stats sidecar, every plot writes a CSV with **one row per data point that drove the plot**, at the natural input-level granularity. Pick the level by what the plot shows:

- **Bar / box / violin chart of group summaries**: rows are per-(individual, group) values — the inputs to the per-group aggregate, not the aggregate itself.
- **Scatter or line plot**: rows are the (x, y) pairs that get plotted, plus any grouping column.
- **Histogram or density**: rows are the individual values being binned.
- **Diagnostic plot fit to a population**: rows are the input population (one per subject/event/whatever the population element is), plus any per-row label or assignment derived in the step (e.g., the cluster the row was assigned to).

The intent: a collaborator with the CSV alone can run independent analyses, regenerate the plot in another tool, or sanity-check the aggregation chain.

**Exception — very high-sample-size data**. When the natural per-row level is something like per-frame across hours of video, per-pixel of an image stack, or any case where the CSV would be hundreds of MB or more, write the next-coarser meaningful aggregation instead (e.g., per-bout, per-subject) and document the chosen level in the step's docstring. Don't write multi-GB CSVs that no one can open.

These conventions ensure every figure is reproducible, queryable from text alone, and editable downstream without re-running anything.

## Data Provenance

Every data artifact a step writes — CSV, HDF5, plot sidecar, array dump, anything — carries its own provenance, embedded as close to the data as the format allows. A data file copied away from the pipeline should still answer on its own: what produced it, from what inputs, with what parameters, when, and at what code version. (This is about data outputs; scripts don't carry it, their outputs do.)

Record, for every output artifact:

- **Script** — the module/entrypoint that wrote the file.
- **Inputs** — every input consumed, as `path` + `size` + `mtime`, plus a content hash for small or critical inputs. Full-hashing large files is too expensive; size+mtime is the cheap staleness proxy, a hash is worth it where correctness hinges on the exact bytes.
- **Parameters** — the config/params in effect at production time, serialized (e.g. JSON of the config object or the parsed CLI args).
- **Created-at** — ISO-8601 datetime with timezone.
- **Git** — the commit SHA at run time, **and a dirty flag** for uncommitted changes. A SHA is meaningless if the working tree was dirty, so always record both.
- **Environment** — the interpreter/venv and the versions of the libraries the result actually depends on.
- **Invocation** — the exact command line / argv.

Embed it as close to the data as the format allows:

- **CSV** — leading comment lines before the header row (`# key: value`, or one `# provenance: {json}` line).
- **HDF5** — a top-level `/provenance` group (fields as scalar datasets or group attrs); dataset-specific provenance as attrs on that dataset.
- **Plots** — a `provenance` block inside the `_stats.json` sidecar.
- **Parquet** — the schema's key-value metadata.
- **npy / npz** — a sibling `<name>.prov.json` (or a `provenance` entry inside the npz).

For **any other output format, devise the analogous "embed it as closely as possible" strategy** — a native metadata slot if the format has one, otherwise a sibling `<name>.prov.json`. Every data artifact gets provenance; no format is exempt.

Conventions:

- Build the common record (git SHA + dirty, datetime, script, invocation, environment) once in a shared helper; thin per-format writers embed it. Don't hand-roll provenance in each step.
- Write provenance **atomically with the output** (same tmp→rename), so no artifact ever exists without it.
- On skip-if-fresh reuse, **keep the existing provenance** — it describes the real production run; don't overwrite it with a no-op re-stamp.
- Provenance is descriptive metadata, **never load-bearing** — pipeline logic must not depend on reading it back.

## HDF5 Performance

Never use compression (`compression="gzip"`, etc.) on HDF5 datasets. Disk space is cheap; compression adds significant CPU overhead on both writes and reads, often dominating pipeline runtime. Always create datasets with no compression:

```python
# CORRECT
f.create_dataset("data", shape=(...), dtype=np.uint16, chunks=(1, H, W))

# WRONG — gzip adds ~3x overhead on write and ~2x on read
f.create_dataset("data", shape=(...), dtype=np.uint16, chunks=(1, H, W),
                  compression="gzip", compression_opts=4)
```

## Fail Fast, Fail Hard

### Validate early

Perform quick validation checks as early as possible in each step — before doing expensive work. Check that input files exist, required columns are present, values are in expected ranges, etc.

### No silent errors

Never swallow exceptions. Never use bare `try/except/pass`. Never invent workarounds for missing data. If something prevents a step from doing its job (missing file, unexpected value, failed operation), raise an exception immediately.

```python
# CORRECT
if not input_path.exists():
    raise FileNotFoundError(f"Required input missing: {input_path}")

# WRONG - silently skipping broken data
try:
    data = load(path)
except Exception:
    pass  # skip this one
```

If a session is missing data that the step needs, that is an error in the pipeline inputs, not something to work around. Raise and fix upstream.

```python
# WRONG - converting a hard error into a warning to keep the pipeline running
missing = [r for r in sessions if not r["input_path"].exists()]
if missing:
    for r in missing:
        log(f"WARNING — input not found for {r['id']}, skipping")
    sessions = [r for r in sessions if r["input_path"].exists()]

# CORRECT - crash immediately, fix the upstream step that produced bad inputs
for r in sessions:
    if not r["input_path"].exists():
        raise FileNotFoundError(f"Required input missing: {r['input_path']}")
```

### Agent behavior on pipeline crashes

When an agent is running a pipeline and hits one of these hard crashes, it must **stop immediately and alert the human user**. Do NOT:
- Try to fix the crash to keep the pipeline running
- Convert the error into a warning or skip
- Modify validation logic to be more lenient

The crash exists to surface a real problem. The human decides how to fix it.
