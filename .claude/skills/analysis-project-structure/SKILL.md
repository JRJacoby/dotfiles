---
name: analysis-project-structure
description: Enforce the standard directory layout for analysis projects. Use when creating new files, directories, scripts, or outputs within an analysis project — ensures everything goes in the right place.
---

# Analysis Project Structure

These are the user's requirements for how analysis projects are organized on disk. Follow them exactly when creating files, directories, or referencing paths.

## Project Initialization

Use **git** and **uv** for project management unless otherwise specified.
Initialize with:

```bash
git init
uv init --python 3.10 --bare  # --bare avoids creating unnecessary main.py and README.md
```

Pin the Python version in `pyproject.toml` as appropriate for the project's
dependencies (e.g., `requires-python = ">=3.10,<3.11"` for keypoint-moseq).

## Top-Level Layout

An analysis project may have some of the following top-level directories:

```
<project>/
  scripts/       # Analysis scripts and pipeline code
  tables/        # CSV and tabular outputs (summaries, master tables, etc.)
  plots/         # Generated figures and visualizations
  docs/          # Plans, notes, and other documentation
  logs/          # Pipeline run logs
  <data_type>/   # Raw or processed data (named for the data type, e.g. moseq_sessions/)
```

- **Do not create other top-level directories** without explicit user approval.
- The data directory name should describe the data type it holds (e.g., `moseq_sessions/`, `pose_tracks/`, `neural_recordings/`). It is not always called `data/`.

## Unit-of-Work Subdirectories

Below each top-level directory, all content is organized into **dated, descriptively-named subdirectories** representing a logical unit of work:

```
scripts/<YYYY-MM-DD>_<descriptive_name>/
tables/<YYYY-MM-DD>_<descriptive_name>/
docs/<YYYY-MM-DD>_<descriptive_name>/
plots/<YYYY-MM-DD>_<descriptive_name>/
logs/<YYYY-MM-DD>_<descriptive_name>/
<data_type>/<YYYY-MM-DD>_<descriptive_name>/
```

**This pattern applies to every top-level directory, not just the ones shown in examples elsewhere.** `docs/`, `logs/`, and `plots/` follow the same unit-of-work convention as `scripts/` and `tables/`.

### Rules

- The date is the date the work was initiated, in `YYYY-MM-DD` format.
- The descriptive name is a short, lowercase, underscore-separated label (e.g., `initial_dataset_inventory`, `stress_vs_naive_comparison`).
- **Related work shares the same subdirectory name across top-level directories.** If `scripts/2026-03-12_initial_dataset_inventory/` produces CSV outputs, those go in `tables/2026-03-12_initial_dataset_inventory/`.
- Do not put files directly in a top-level directory — always use a unit-of-work subdirectory.
- **If it is unclear whether a new task belongs in an existing unit of work or warrants a new one, ask the user before creating directories.**

## Path References in Scripts

- Scripts derive the project root from their own location using `__file__` and relative traversal (e.g., `Path(__file__).resolve().parent.parent.parent`).
- Scripts build paths to `tables/` and the data directory from the project root.
- Scripts use `SCRIPT_DIR.name` (which gives the `YYYY-MM-DD_<name>` portion) to find their corresponding data and output directories, keeping the mapping automatic.
- **Do not hardcode absolute paths** in reusable scripts. One-off exploratory scripts may use absolute paths.

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "moseq_sessions" / SCRIPT_DIR.name
OUTPUT_DIR = PROJECT_ROOT / "tables" / SCRIPT_DIR.name
```

## Tables Are Generated Outputs

Anything in `tables/` should be reproducible by running the corresponding script in `scripts/`. Do not manually create or edit files in `tables/`.
