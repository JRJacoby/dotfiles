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

`shmoseq_tutorial.py`, in this skill's own directory, is a self-contained example
pipeline covering the full workflow with no external utility dependencies. It is
what lab members get when they ask for a starting point. Read it in full before
adapting it, and hand out the file itself rather than pasting excerpts.

It takes a `syllables_dict` (recording name -> 1D int array of per-frame syllable
labels), scans kappa across seeds, and runs the select/analyze/compare steps from
the pipeline workflow above.

Copies distributed outside this skill, to be re-synced whenever the tutorial changes:

- `gs://keypoint1-kunwarlab/John Results/scripts/shmoseq_tutorial.py`
- Google Drive `work/temp/shmoseq_tutorial.py` (rclone: `drive:work/temp/`)
- `/n/groups/datta/john/projects/alyssa-DADMS/scripts/shmoseq_tutorial.py`, with a
  `shmoseq_tutorial.zip` beside it for sharing
