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
import tempfile

tmp = output_path.with_suffix(output_path.suffix + ".tmp")
# Clean up any leftover tmp from a previous interrupted run
if tmp.exists():
    tmp.unlink()
write_data(tmp)
tmp.rename(output_path)  # atomic on same filesystem
```

For atomic writes, the pattern is:
1. At the start of processing an item, delete any existing `.tmp` file for that item
2. Write to the `.tmp` path
3. Rename `.tmp` to the final path as the last operation

Freshness checks use only the final output path, never the `.tmp` path.

## Logging

All log messages must include **timestamps** and be written to a **log file**. Each pipeline run creates a new log file named with the **timestamp of when the run started** (e.g., `2026-03-18_14-30-05.log`). Configure logging once at the top of `main()` and use it throughout.

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

## Plot Sidecar Files

Any time a plot compares groups (genotypes, conditions, etc.), it must have a **sidecar file** saved alongside it containing the statistical details for every comparison. The sidecar file shares the plot's basename with a `_stats.json` suffix.

```
plots/
  state_frequencies_by_genotype.png
  state_frequencies_by_genotype_stats.json
```

The sidecar file is structured JSON containing:
- All pairwise comparisons with test statistic, raw p-value, corrected p-value, and significance flag
- The statistical test used and any correction method
- Sample sizes per group
- Summary statistics (means, medians, CIs) per group

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

This ensures all statistical results are machine-readable and reproducible, not just visually indicated on the plot.

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
