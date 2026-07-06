---
name: shmoseq
description: Fit state-moseq (shMoSeq) hierarchical HMM models to discover high-level behavioral states from syllable sequences. Use when working with state-moseq, hierarchical behavioral states, kappa scans, or any analysis built on top of keypoint-MoSeq or MoSeq2 syllables.
---

# State-MoSeq (shMoSeq)

State-moseq discovers **high-level behavioral states** from frame-level syllable sequences using a hierarchical hidden Markov model (HHMM). It works with syllables from any source: keypoint-MoSeq, depth MoSeq, or other segmentation methods.

**Paper:** Weinreb, Thamarai Kannan, Newman-Boulle, Sainburg, Gillis, Plotnikoff, Makowska, Pearl, Osman, Linderman, Datta. "Spontaneous behavior is a succession of self-directed tasks." *Neuron* 114(5):922-937.e12, 2026.

**Repository:** `/n/groups/datta/john/repos/state-moseq`
**Docs:** https://state-moseq.readthedocs.io/en/latest/
**Install:** `pip install state-moseq` (requires JAX)

## What it does

MoSeq discovers ~40-100 sub-second behavioral syllables. But syllable sequences have structure at longer timescales that a first-order Markov chain cannot capture. shMoSeq fits a two-level hierarchical HMM:

- **Level 1 (hidden):** A Markov chain over K high-level states (typically 3-5). Each frame is assigned one hidden state.
- **Level 2 (observed):** Given the current state, syllable-to-syllable transitions follow a state-specific transition matrix. States don't use exclusive syllables -- they bias the relative frequencies and transition patterns.

The key diagnostic that motivates this model is **lagged mutual information**: real syllable data shows higher MI at long lags than an equivalent Markov chain. The timescale where real MI diverges from Markov MI identifies the target duration for shMoSeq states.

The paper shows that prefrontal cortex (PFC) neural activity preferentially encodes the identity of these high-level states rather than individual syllables, and that PFC shapes which states are expressed in different contexts.

## Package API

### Imports

```python
import state_moseq as sm
from state_moseq import hhmm_efficient   # Data-efficient HHMM (use this one)
from state_moseq import hhmm_standard    # Full-rank HHMM (pedagogical)
from state_moseq.util import (
    batch, unbatch,
    get_durations, get_frequencies, get_adjusted_rand,
    compare_states, sample_instances,
    lagged_mutual_information, count_transitions,
)
from state_moseq.hhmm_efficient import (
    resample_params, resample_states, log_params_prob,
    fit_gibbs, predicted_states, smoothed_states, filtered_states,
    simulate, random_params, initialize_params,
    marginal_loglik, log_joint_prob,
    get_syllable_trans_probs,
)
from state_moseq.viz import generate_grid_movies, grid_movie, plot_sankey
```

### Data Structures

**Input data** (`syllables_dict`): `Dict[str, np.ndarray]` mapping recording names to 1D integer arrays of syllable labels (one per frame).

**Batched data** (for model fitting):
```python
data = {
    "syllables": jnp.array,  # (n_sequences, n_timesteps) int
    "mask": jnp.array,       # (n_sequences, n_timesteps) binary (1=valid, 0=padding)
}
```

**Model parameters** (efficient version):
```python
params = {
    "emission_base": (n_syllables, n_syllables-1),     # Shared baseline syllable transitions (low-rank)
    "emission_biases": (n_states-1, n_syllables-1),    # Per-state transition biases (low-rank)
    "trans_probs": (n_states, n_states),                # Hidden state transitions (rows sum to 1)
}
```

The low-rank parameterization is critical: with ~50 syllables, the standard model needs ~50x50x5 = 12,500 emission params per state. The efficient version shares a baseline and only adds `(n_states-1) * (n_syllables-1)` bias parameters.

**Hyperparameters:**
```python
hypparams = {
    "n_states": int,              # Number of hidden states (typically 3-5)
    "n_syllables": int,           # Number of unique syllables (auto from data)
    "emission_base_sigma": float, # Prior std for baseline transitions (default 1)
    "emission_biases_sigma": float, # Prior std for state biases (default 1)
    "trans_beta": float,          # Dirichlet baseline concentration (default 1)
    "trans_kappa": float,         # Stickiness -- self-transition boost (10^4 to 10^6)
    "emission_gd_iters": int,     # GD iterations for Laplace approx (default 1000)
    "emission_gd_lr": float,      # GD learning rate (default 5e-3)
}
```

### Key Functions

#### Data handling

| Function | Signature | Purpose |
|----------|-----------|---------|
| `sm.batch()` | `(data_dict, keys=None, seg_length=None, seg_overlap=30)` -> `(data, mask, (keys, bounds))` | Stack variable-length sequences into fixed-size array with padding |
| `sm.unbatch()` | `(data, keys, bounds)` -> `data_dict` | Reconstruct original variable-length sequences |

#### Inference

| Function | Purpose |
|----------|---------|
| `fit_gibbs(data, hypparams, init_params, init_states=None, seed, num_iters, parallel=False)` | Main inference: Gibbs sampling. Returns `(params, states, log_joints)` |
| `resample_params(seed, data, states, hypparams, params=None)` | Resample emission + transition params. Returns `(params, gd_losses)` |
| `resample_states(seed, data, params, parallel=False)` | Sample hidden states from posterior. Returns `(states, marginal_loglik)` |
| `predicted_states(data, params)` | Viterbi (MAP) state sequence |
| `smoothed_states(data, params, parallel=False)` | Marginal posteriors via forward-backward. Returns `(n_seq, n_time, n_states)` |
| `filtered_states(data, params, parallel=False)` | Forward-pass only marginals |
| `simulate(seed, params, n_timesteps, n_sequences)` | Generate synthetic data from model |

#### Analysis

| Function | Purpose |
|----------|---------|
| `sm.get_durations(states_dict)` | Duration of each state run (in frames) |
| `sm.get_frequencies(states_dict, num_states, runlength=False)` | Proportion of time in each state |
| `sm.compare_states(states1, states2, n_states)` | Hungarian-aligned confusion matrix + accuracy |
| `get_adjusted_rand(states_dict1, states_dict2, downsample=10)` | Adjusted Rand index between two models |
| `sm.lagged_mutual_information(sequences, mask, lags)` | Returns `(real_mi, markov_mi, shuff_mi)` |
| `count_transitions(states, mask, n_states)` | Frame-level transition counts (includes self-loops; see "Transition matrices" rule below) |
| `sample_instances(states_dict, num_instances)` | Random instances of each state: `{state: [(key, start, end), ...]}` |

#### Visualization

| Function | Purpose |
|----------|---------|
| `sm.generate_grid_movies(states_dict, video_paths, centroids, output_dir, rows, cols, ...)` | Montage videos per state |
| `plot_sankey(states_dict1, states_dict2)` | Plotly Sankey diagram comparing two models |

### Gibbs Sampling Loop (Manual)

When parallelizing across SLURM (the typical production use), write your own loop rather than using `fit_gibbs`:

```python
import jax.random as jr
import jax.numpy as jnp
from state_moseq.hhmm_efficient import resample_params, resample_states, log_params_prob

seed = jr.PRNGKey(seed_idx)
states = jr.randint(seed, data["syllables"].shape, 0, n_states)

log_probs = []
params = None
for _ in tqdm.trange(100, desc="Gibbs sampling"):
    seed, subseed = jr.split(seed)
    params, losses = resample_params(subseed, data, states, hypparams, params)
    states, marginal_loglik = resample_states(seed, data, params, parallel=False)
    log_probs.append(log_params_prob(params, hypparams).item() + marginal_loglik.item())

    if np.isnan(log_probs[-1]):
        raise RuntimeError("NaNs during fitting")
```

### SLURM Parallelization with submitit

For production use, parallelize the kappa scan across GPU nodes using `submitit`. Each
(kappa, seed) combination runs as an independent SLURM job.

The pattern: define a picklable wrapper function that loads data and calls `fit_single_model`,
then use `executor.map_array` to fan out all jobs. Run the script once to submit, then
again after jobs complete to generate plots and downstream analyses.

```python
import submitit
import shmoseq_utils as utils

# These must be module-level (picklable) for submitit
results_h5_path = Path("path/to/results.h5")
output_dir = Path("path/to/output")
n_states = 5
fps = 25

def fit_single_model_wrapper(kappa, seed_idx):
    """Wrapper that loads data inside the SLURM job and fits one model."""
    # IMPORTANT: submitit pickles this function and unpickles it in a fresh
    # Python process on the compute node. sys.path modifications from the
    # main script do NOT carry over. You must add the path to shmoseq_utils
    # inside the wrapper, and import it here -- not at module level.
    import sys as _sys
    _scripts_dir = str(Path(__file__).resolve().parent.parent)
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    import shmoseq_utils as _utils
    syllables_dict = _utils.load_syllables_dict(results_h5_path)
    scan_dir = output_dir / "kappa_scan"
    return _utils.fit_single_model(
        kappa, seed_idx, syllables_dict, scan_dir, n_states, fps
    )

def run_kappa_scan(kappas, random_seeds):
    """Submit kappa scan jobs via submitit. Skip already-completed models."""
    scan_dir = output_dir / "kappa_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "submitit_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Collect only jobs that haven't been run yet
    jobs_to_submit = []
    for kappa in kappas:
        for seed_idx in random_seeds:
            file_prefix = scan_dir / f"kappa={kappa}_nstates={n_states}_seed={seed_idx}"
            seq_path = Path(f"{file_prefix}-state_sequences.p")
            info_path = Path(f"{file_prefix}-additional_info.p")
            if not (seq_path.exists() and info_path.exists()):
                jobs_to_submit.append((kappa, int(seed_idx)))

    if not jobs_to_submit:
        print("All kappa scan models already exist")
        return

    print(f"Submitting {len(jobs_to_submit)} jobs via submitit...")

    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        slurm_partition="gpu_quad,gpu",
        slurm_qos="gpuquad_qos",
        gpus_per_node=1,
        cpus_per_task=4,
        mem="24G",
        timeout_min=60,
        slurm_job_name="shmoseq",
    )

    kappas_list = [job[0] for job in jobs_to_submit]
    seeds_list = [job[1] for job in jobs_to_submit]
    jobs = executor.map_array(fit_single_model_wrapper, kappas_list, seeds_list)

    print(f"Submitted {len(jobs)} jobs")
    for job, (kappa, seed_idx) in zip(jobs, jobs_to_submit):
        print(f"  Job {job.job_id}: kappa={kappa:.0f}, seed={seed_idx}")
    print("\nMonitor with: squeue -u $USER")
    print("After completion, run this script again to generate plots.")
```

Key points:
- The wrapper function must be **picklable** (module-level, no closures over unpicklable objects)
- Data is loaded **inside** each job (not serialized via submitit)
- Already-completed models are skipped (idempotent re-runs)
- Typical SLURM settings: 1 GPU, 4 CPUs, 24G RAM, 60 min timeout per model
- The script is designed to be run twice: once to submit jobs, once after completion for analysis

### Saved Model Format

Each fitted model produces two pickle files:

| File | Contents |
|------|----------|
| `kappa={k}_nstates={n}_seed={s}-state_sequences.p` | `Dict[str, np.ndarray]` mapping recording names to state sequences |
| `kappa={k}_nstates={n}_seed={s}-additional_info.p` | `{"params": dict, "hypparams": dict, "log_probs": list}` |

## Pipeline Workflow

### 1. Lagged Mutual Information (motivating diagnostic)

Compute MI at multiple lags to verify hierarchical structure exists and identify the target state-duration timescale. Real syllable data carries more information about its future at long lags than an equivalent first-order Markov chain; the lag where the **real** curve separates from the **Markov** curve before both decay to the shuffle floor is the timescale shMoSeq states should fall in — the anchor for picking kappa off the duration-vs-kappa panel. CPU-only; no GPU/SLURM needed.

Canonical computation + plot (from the state-moseq `example_data_tutorial`):

```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import state_moseq as sm

num_lags = 40
min_lag, max_lag = fps, fps * 300          # 1 s .. 300 s
lags = jnp.logspace(jnp.log10(min_lag), jnp.log10(max_lag), num_lags).astype(int)

seqs, mask, metadata = sm.batch(syllables_dict)
seqs, mask = jnp.array(seqs), jnp.array(mask)   # MUST be jax arrays (see gotchas)
real_mis, markov_mis, shuff_mis = sm.lagged_mutual_information(seqs, mask, lags)

for mis, label, color in zip([real_mis, markov_mis, shuff_mis],
                             ["real", "Markov", "shuffle"], ["red", "black", "gray"]):
    mean = mis.mean(0)                          # returns are (n_sequences, n_lags)
    err = mis.std(0) / np.sqrt(len(mis)) * 2
    plt.plot(lags / fps, mean, c=color, label=label)
    plt.fill_between(lags / fps, mean - err, mean + err, facecolor=color, alpha=0.3)
plt.xscale("log"); plt.yscale("log")           # log-log: a modest separation is invisible on a linear y-axis
plt.xlabel("lag (seconds)"); plt.ylabel("mutual information"); plt.legend()
```

**Gotchas (these have bitten us repeatedly):**
- `sequences`/`mask` **must be jax arrays** (`jnp.array(...)`). The function uses `mask.at[:, :lag].set(0)`; a NumPy array raises `AttributeError: 'numpy.ndarray' object has no attribute 'at'`.
- The three returns are **per-sequence**, shape `(n_sequences, n_lags)` — not a single curve. Reduce with `.mean(0)` (and `.std(0)/sqrt(n)` for an error band) before plotting, or you'll draw one line per recording.
- **Plot log-log.** On a linear y-axis a real-but-modest real-vs-Markov gap looks like the curves overlap. The separation often only reads clearly with `yscale("log")`.
- A **weak/absent** separation is a real result: it means the syllables are near-Markovian and shMoSeq may add little. Report it rather than forcing a kappa choice.

### 2. Kappa Scan

Fit models across a grid of `kappa` values (default `np.logspace(3, 7, 20)` — 10³ to 10⁷, wide enough that the log-prob curve actually rolls over on both ends rather than peaking at the boundary) with multiple random seeds (3-10) per kappa. Two approaches:
- **Tutorial/small datasets:** Nested for loop (see tutorial script reference below)
- **Production/HPC:** submitit SLURM parallelization (see "SLURM Parallelization with submitit" above) -- each (kappa, seed) pair runs as an independent GPU job

### 3. Plot Kappa Scan

Three-panel diagnostic: log prob vs kappa, median duration vs kappa, scatter of duration vs log prob colored by kappa.

### 4. Select Kappa

Choose kappa where:
- Log probability peaks (or is near peak)
- Median state duration matches the MI-identified timescale
- See docs: https://state-moseq.readthedocs.io/en/latest/parameter_scan_tutorial.html

### 5. Select Best Model

At chosen kappa, pick the seed with highest final log probability. Validate with confusion matrices and Rand scores across seeds.

### 6. Analyze Selected Model

Duration histogram, state frequency bar chart, transition matrix.

### 7. Compare Across Seeds

Confusion matrices (Hungarian-aligned) and adjusted Rand scores to assess reproducibility.

### 8. Grid Movies (optional)

Montage videos of behavior during each state for qualitative inspection.

### 9. SNUB Projects (optional)

Per-recording SNUB projects with video + state sequence heatmap for detailed browsing.

## Hyperparameter Selection

### Kappa (the critical parameter)

`trans_kappa` controls state duration (stickiness of the hidden Markov chain). Higher kappa = longer states. Typical range: 10^4 to 10^6. Selected via kappa scan (see above).

### n_states

Log probability increases monotonically with more states, so it is **not** a reliable selection metric. Instead:
1. Fit models at several `n_states` values (3-7) with multiple seeds.
2. Compute adjusted Rand scores between all pairs at each `n_states`.
3. Select `n_states` that maximizes **median Rand score** (most reproducible partition).
4. Check effective state count via frequency plot -- some states may have negligible frequency.

### Other hyperparameters

Usually left at defaults: `trans_beta=1`, `emission_base_sigma=1`, `emission_biases_sigma=1`, `emission_gd_iters=1000`, `emission_gd_lr=5e-3`, 100 Gibbs iterations.

## Shared Utilities

`shmoseq_utils.py` provides reusable functions for production analysis scripts:

| Function | Purpose |
|----------|---------|
| `load_syllables_dict(results_h5_path)` | Load syllable sequences from keypoint-MoSeq results.h5 |
| `load_centroids_dict(results_h5_path)` | Load centroids from results.h5 |
| `load_video_paths_dict(results_h5_path, video_dir, extract_base_name_func)` | Map recording keys to video files |
| `fit_single_model(kappa, seed_idx, syllables_dict, scan_dir, n_states, fps)` | Fit one model (for SLURM jobs) |
| `select_best_model(scan_dir, selected_kappa, seed=None)` | Pick best model at chosen kappa |
| `plot_kappa_scan_results(scan_dir, output_dir, fps, n_states)` | Three-panel kappa diagnostic |
| `compute_and_plot_lagged_mi(syllables_dict, output_dir, fps)` | Lagged MI analysis |
| `analyze_selected_model(states_dict, info, output_dir, fps, n_states)` | Duration + frequency plots |
| `compare_models_across_seeds(best_states_dict, scan_dir, selected_kappa, best_seed, n_states, output_dir)` | Confusion + Rand comparison |
| `analyze_state_transitions(states_dict, output_dir, n_states)` | State-to-state transition matrix |
| `visualize_state_sequences(states_dict, output_dir, fps, n_states)` | Per-recording state sequence plots |
| `generate_grid_movies(states_dict, video_paths_dict, centroids_dict, output_dir, n_states)` | Grid movies for each state |
| `create_snub_projects(states_dict, video_paths_dict, output_dir, fps)` | SNUB projects per recording |

## Important: Transition Matrices Are Always Bout-Level

When computing transition matrices between states (or between syllables), always use **bout-to-bout transitions** — i.e., one count per state-change event. **Self-transitions do not exist by construction**; the diagonal is always zero.

The frame-level alternative (counting every (state[t], state[t+1]) pair, including state[t] == state[t+1]) is wrong for this purpose: with multi-second bout durations and 30 fps recording, ~99% of frame-level transitions are self-loops, and the off-diagonal pattern that carries the behavioral structure gets drowned in noise.

**Practical recipe:** convert each per-frame state sequence to a per-bout state sequence first (collapse runs to a single label per run), then count consecutive (bout[i], bout[i+1]) pairs into the transition matrix. Equivalently: take the frame-level transition matrix from `count_transitions` and zero out the diagonal — you get the same off-diagonal counts. Row-normalize after zeroing the diagonal so each row is a valid distribution over the next state given that a state change occurred.

This rule applies whether you are comparing genotypes, building heatmaps, or feeding transition matrices into any downstream analysis.

## Important: Per-Frame vs Per-Bout Frequency

When computing state/syllable frequencies, **always compute both per-frame and per-bout (runlength) frequencies** — do not pick one. The two metrics answer different questions:

- **Per-frame** (`runlength=False`): proportion of total time spent in each state. Answers "how much time does the animal allocate to this state?"
- **Per-bout** (`runlength=True`): proportion of total state visits (bouts) that are each state. Answers "how often does the animal enter this state?"

These can diverge substantially when states have very different durations, so reporting both is the default in genotype comparison analyses.

## Tutorial Script Reference

The self-contained tutorial script at `scripts/shmoseq_tutorial.py` demonstrates the full pipeline without external utility dependencies. It is designed for lab members to copy and adapt. Below is the complete script for reference.

<details>
<summary>scripts/shmoseq_tutorial.py (full transcript)</summary>

```python
"""
State-moseq (shMoSeq) tutorial — a self-contained example pipeline.

shMoSeq discovers high-level behavioral states from frame-level syllable
sequences using a hierarchical hidden Markov model. It works with syllables
from any source — e.g., keypoint-MoSeq, depth MoSeq, or other segmentation methods.

Input
-----
This script needs a `syllables_dict`: a Dict[str, np.ndarray] mapping recording
names to 1D integer arrays of syllable labels (one per frame). The number of
keys in this dict is the number of recording sessions. Each 1D numpy array
is as long as the number of frames in that session.

Replace the placeholder in main() with your own data loading code. For example,
loading from a keypoint-MoSeq results.h5:

    import h5py
    syllables_dict = {}
    with h5py.File("results.h5", "r") as f:
        for key in f.keys():
            syllables_dict[key] = f[key]["syllable"][:]

Usage
-----
1. Fill in main() with your data loading code, and set the constants below.
2. Run the script: `python shmoseq_tutorial.py`
   - This fits models across a grid of kappa values and random seeds.
   - This tutorial just uses a nested loop - you will probably want to use an HPC cluster
   to parallelize this step for speed.
3. Review the kappa scan plot in output_dir.
4. Set `selected_kappa` to your chosen value (and optionally `seed`). Choose a model as described in https://state-moseq.readthedocs.io/en/latest/parameter_scan_tutorial.html#
5. Run the script again to select the best model, analyze it, and compare seeds. If you didn't set a seed in #4, the seed with the highest final log probability will be chosen.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tqdm

# Constants — edit these for your dataset
output_dir = Path("input/path/to/shmoseq_output_dir")

fps = 25           # Frame rate of your recordings
n_states = 5       # Number of high-level states to fit. Usually we find 3-5.
kappas = np.logspace(3, 7, 20)  # Kappa values to scan (controls state duration); span 10^3 to 10^7 so the log-prob curve rolls over on both ends
random_seeds = np.arange(5)     # Random seeds for each kappa

# Set these after reviewing the kappa scan plot:
selected_kappa = None  # e.g., 48329.3
seed = None            # If None, use highest log prob. If int, use that specific seed.


def fit_single_model(kappa, seed_idx, syllables_dict, scan_dir, n_states, fps):
    """Fit a single shMoSeq model with given kappa and random seed."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import state_moseq as sm
    from state_moseq.hhmm_efficient import resample_params, resample_states, log_params_prob

    scan_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = scan_dir / f"kappa={kappa:.1f}_nstates={n_states}_seed={seed_idx}"
    seq_path = Path(f"{file_prefix}-state_sequences.p")
    info_path = Path(f"{file_prefix}-additional_info.p")

    if seq_path.exists() and info_path.exists():
        print(f"kappa={kappa:.0f}, seed={seed_idx} already exists, skipping")
        return

    print(f"Fitting kappa={kappa:.0f}, seed={seed_idx}...")

    n_syllables = max(np.max(seq) for seq in syllables_dict.values()) + 1
    seg_length = max(len(seq) for seq in syllables_dict.values())
    seqs, mask, metadata = sm.batch(syllables_dict, seg_length=seg_length)

    data = {
        "syllables": jnp.array(seqs),
        "mask": jnp.array(mask),
    }

    hypparams = {
        "n_states": n_states,
        "emission_base_sigma": 1,
        "emission_biases_sigma": 1,
        "trans_beta": 1,
        "trans_kappa": kappa,
        "n_syllables": int(n_syllables),
        "emission_gd_iters": 1000,
        "emission_gd_lr": 5e-3,
    }

    key = jr.PRNGKey(seed_idx)
    states = jr.randint(key, data["syllables"].shape, 0, n_states)

    log_probs = []
    params = None
    for _ in tqdm.trange(100, desc="Gibbs sampling"):
        key, subseed = jr.split(key)
        params, losses = resample_params(subseed, data, states, hypparams, params)
        states, marginal_loglik = resample_states(key, data, params, parallel=False)
        log_probs.append(log_params_prob(params, hypparams).item() + marginal_loglik.item())

        if np.isnan(log_probs[-1]):
            raise RuntimeError("NaNs during fitting")

    states_dict = sm.unbatch(np.array(states), *metadata)
    joblib.dump(states_dict, seq_path)

    additional_info = {
        "params": jax.device_get(params),
        "hypparams": hypparams,
        "log_probs": log_probs,
    }
    joblib.dump(additional_info, info_path)
    print(f"Saved model to {seq_path}")


def run_kappa_scan(syllables_dict):
    """Fit models across all kappa values and random seeds."""
    scan_dir = output_dir / "kappa_scan"
    for kappa in kappas:
        for seed_idx in random_seeds:
            fit_single_model(kappa, int(seed_idx), syllables_dict, scan_dir, n_states, fps)


def plot_kappa_scan_results():
    """Plot kappa scan results: log prob, median duration, and scatter."""
    import state_moseq as sm

    scan_dir = output_dir / "kappa_scan"
    plot_path = output_dir / "kappa_scan_results.png"

    seq_files = sorted(scan_dir.glob("*-state_sequences.p"))
    if not seq_files:
        print("No kappa scan models found, skipping plot")
        return

    results = []
    for seq_file in seq_files:
        match = re.search(r"kappa=([0-9.e+]+)_nstates=(\d+)_seed=(\d+)", seq_file.name)
        if match:
            kappa_val = float(match.group(1))
            seed_idx = int(match.group(3))
            info_file = seq_file.parent / seq_file.name.replace(
                "-state_sequences.p", "-additional_info.p"
            )
            if info_file.exists():
                states_dict = joblib.load(seq_file)
                info = joblib.load(info_file)
                median_dur = np.median(sm.get_durations(states_dict)) / fps
                log_prob = info["log_probs"][-1]
                results.append((kappa_val, seed_idx, median_dur, log_prob))

    if not results:
        print("No valid kappa scan results found")
        return

    results = np.array(results)
    discovered_kappas = np.unique(results[:, 0])
    discovered_seeds = np.unique(results[:, 1])

    print(f"Found {len(results)} models: {len(discovered_kappas)} kappas x {len(discovered_seeds)} seeds")

    median_durations = np.zeros((len(discovered_kappas), len(discovered_seeds)))
    log_probs_final = np.zeros((len(discovered_kappas), len(discovered_seeds)))

    for row in results:
        kappa_val, seed_idx, median_dur, log_prob = row
        i = np.where(discovered_kappas == kappa_val)[0][0]
        j = int(seed_idx)
        median_durations[i, j] = median_dur
        log_probs_final[i, j] = log_prob

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].plot(discovered_kappas, np.median(log_probs_final, axis=1), c="k")
    axs[0].fill_between(
        discovered_kappas,
        *np.percentile(log_probs_final, [25, 75], axis=1),
        facecolor="k", alpha=0.3,
    )
    axs[0].set_xscale("log")
    axs[0].set_ylabel("log probability")
    axs[0].set_xlabel("kappa")

    axs[1].plot(discovered_kappas, np.median(median_durations, axis=1), c="k")
    axs[1].fill_between(
        discovered_kappas,
        *np.percentile(median_durations, [25, 75], axis=1),
        facecolor="k", alpha=0.3,
    )
    axs[1].set_xscale("log")
    axs[1].set_ylabel("median duration (s)")
    axs[1].set_xlabel("kappa")

    kappas_matrix = (
        np.broadcast_to(np.log(discovered_kappas).reshape(-1, 1), log_probs_final.shape)
        .ravel()
    )
    axs[2].scatter(median_durations.ravel(), log_probs_final.ravel(), c=kappas_matrix)
    axs[2].set_xlabel("median duration (s)")
    axs[2].set_ylabel("log probability")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved kappa scan plot to {plot_path}")


def select_best_model():
    """Select model at chosen kappa. Uses specific seed if set, else highest log prob."""
    scan_dir = output_dir / "kappa_scan"
    seq_files = sorted(scan_dir.glob("*-state_sequences.p"))

    if not seq_files:
        raise RuntimeError(f"No models found in {scan_dir}")

    kappa_files = []
    for seq_file in seq_files:
        match = re.search(r"kappa=([0-9.e+]+)_nstates=(\d+)_seed=(\d+)", seq_file.name)
        if match:
            kappa_val = float(match.group(1))
            if abs(kappa_val - selected_kappa) < 1.0:
                kappa_files.append(seq_file)

    if not kappa_files:
        raise RuntimeError(f"No models found with kappa={selected_kappa}")

    if seed is not None:
        for seq_file in kappa_files:
            match = re.search(r"kappa=([0-9.e+]+)_nstates=(\d+)_seed=(\d+)", seq_file.name)
            if match and int(match.group(3)) == seed:
                info_file = seq_file.parent / seq_file.name.replace(
                    "-state_sequences.p", "-additional_info.p"
                )
                if info_file.exists():
                    info = joblib.load(info_file)
                    best_kappa = float(match.group(1))
                    print(f"Selected model: kappa={best_kappa:.1f}, seed={seed}, "
                          f"log_prob={info['log_probs'][-1]:.2f}")
                    return best_kappa, seed, joblib.load(seq_file), info
        raise RuntimeError(f"No model found with kappa={selected_kappa} and seed={seed}")

    best_log_prob = -np.inf
    best_result = None

    for seq_file in kappa_files:
        match = re.search(r"kappa=([0-9.e+]+)_nstates=(\d+)_seed=(\d+)", seq_file.name)
        if match:
            info_file = seq_file.parent / seq_file.name.replace(
                "-state_sequences.p", "-additional_info.p"
            )
            if info_file.exists():
                info = joblib.load(info_file)
                log_prob = info["log_probs"][-1]
                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_result = (
                        float(match.group(1)),
                        int(match.group(3)),
                        joblib.load(seq_file),
                        info,
                    )

    if best_result is None:
        raise RuntimeError(f"No valid models found with kappa={selected_kappa}")

    print(f"Selected best model: kappa={best_result[0]:.1f}, seed={best_result[1]}, "
          f"log_prob={best_log_prob:.2f}")
    return best_result


def analyze_selected_model(states_dict, info):
    """Analyze selected model: duration and frequency distributions."""
    import state_moseq as sm
    from state_moseq.util import get_frequencies

    analysis_dir = output_dir / "selected_model_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    durations = sm.get_durations(states_dict)
    frequencies = get_frequencies(states_dict, num_states=n_states)
    median_duration = np.median(durations) / fps

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist(durations / fps, bins=40, range=(0, 180), density=True)
    ax.set_title(f"Median duration = {round(median_duration, 2)} seconds", size=10)
    ax.set_xlabel("state duration (seconds)")
    ax.set_ylabel("frequency")
    plt.tight_layout()
    fig.savefig(analysis_dir / "duration_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2))
    ax.bar(np.arange(len(frequencies)), frequencies)
    ax.set_xlabel("high-level states")
    ax.set_ylabel("prop. of frames")
    plt.tight_layout()
    fig.savefig(analysis_dir / "frequency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(analysis_dir / "model_summary.txt", "w") as f:
        f.write(f"Selected Model Analysis\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Median duration: {median_duration:.2f} seconds\n")
        f.write(f"\nState frequencies:\n")
        for i, freq in enumerate(frequencies):
            f.write(f"  State {i}: {freq:.4f}\n")
        f.write(f"\nFinal log probability: {info['log_probs'][-1]:.2f}\n")

    print(f"Saved analysis to {analysis_dir}")


def compare_models_across_seeds(best_states_dict, best_seed):
    """Compare the best model with other seeds at the same kappa."""
    import state_moseq as sm
    from state_moseq.util import get_adjusted_rand

    scan_dir = output_dir / "kappa_scan"
    comparison_dir = output_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    seq_files = sorted(scan_dir.glob("*-state_sequences.p"))

    other_models = {}
    for seq_file in seq_files:
        match = re.search(r"kappa=([0-9.e+]+)_nstates=(\d+)_seed=(\d+)", seq_file.name)
        if match:
            kappa_val = float(match.group(1))
            seed_idx = int(match.group(3))
            if abs(kappa_val - selected_kappa) < 1.0 and seed_idx != best_seed:
                other_models[seed_idx] = joblib.load(seq_file)

    if not other_models:
        print("No other seed models found for comparison")
        return

    confusion_matrices = []
    rand_scores = []
    seed_indices = sorted(other_models.keys())

    for seed_idx in seed_indices:
        states2 = other_models[seed_idx]
        confusion, permutation, accuracy = sm.compare_states(
            best_states_dict, states2, n_states
        )
        confusion_matrices.append((confusion[permutation], accuracy, seed_idx))
        rand_scores.append((seed_idx, get_adjusted_rand(best_states_dict, states2)))

    fig, axs = plt.subplots(1, len(confusion_matrices), figsize=(10, 1.5))
    if len(confusion_matrices) == 1:
        axs = [axs]
    for ax, (conf_mat, accuracy, seed_idx) in zip(axs, confusion_matrices):
        ax.imshow(conf_mat, vmin=0)
        ax.set_xlabel(f"states\n(seed={seed_idx})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"acc={accuracy:.2f}", size=8)
    axs[0].set_ylabel(f"states\n(seed={best_seed})")
    plt.tight_layout()
    fig.savefig(comparison_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([s for s, _ in rand_scores], [r for _, r in rand_scores])
    ax.set_xlabel("seed")
    ax.set_ylabel("adjusted rand score")
    ax.set_title("Model consistency across seeds")
    plt.tight_layout()
    fig.savefig(comparison_dir / "rand_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(comparison_dir / "comparison_summary.txt", "w") as f:
        f.write(f"Model Comparison Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Best model: kappa={selected_kappa:.1f}, seed={best_seed}\n\n")
        f.write(f"Comparisons with other seeds:\n")
        for seed_idx, (_, accuracy, _) in zip(seed_indices, confusion_matrices):
            f.write(f"  Seed {seed_idx}: accuracy={accuracy:.4f}\n")
        f.write(f"\nAdjusted Rand scores:\n")
        for seed_idx, rand_score in rand_scores:
            f.write(f"  Seed {seed_idx}: {rand_score:.4f}\n")

    print(f"Saved comparison results to {comparison_dir}")


def main():
    """Run the shMoSeq analysis pipeline."""
    # Load your syllable sequences here. Example for keypoint-MoSeq:
    #   syllables_dict = {}
    #   with h5py.File("path/to/results.h5", "r") as f:
    #       for key in f.keys():
    #           syllables_dict[key] = f[key]["syllable"][:]
    syllables_dict = ...  # Replace with your data loading code

    run_kappa_scan(syllables_dict)
    plot_kappa_scan_results()

    if selected_kappa is None:
        print("\nselected_kappa is None — review the kappa scan plot, set selected_kappa, "
              "and run again.")
        return

    best_kappa, best_seed, best_states_dict, best_info = select_best_model()
    analyze_selected_model(best_states_dict, best_info)
    compare_models_across_seeds(best_states_dict, best_seed)


if __name__ == "__main__":
    main()
```

</details>
