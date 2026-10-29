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

Installation
------------
Go to https://github.com/dattalab/state-moseq and follow the installation
instructions. The [cuda] version is STRONGLY recommended — the CPU-only version
will be very slow.

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
kappas = np.logspace(3, 7, 20)  # Kappa values to scan (controls state
                                # duration); span 10^3 to 10^7 so the
                                # log-prob curve rolls over on both ends
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
