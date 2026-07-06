---
name: moseq3
description: Use when fitting depth-video MoSeq (AR-HMM) models with the moseq3 package — GPU PCA on extracted/size-normed depth frames, tuning kappa for syllable duration, and extracting syllable labels. For depth-MoSeq (not keypoint-MoSeq), moseq3 GPUPCA/whiten_all/run_arhmm_model, or jax-moseq depth AR-HMM pipelines on O2 GPUs.
---

# moseq3 — depth-video MoSeq (PCA + AR-HMM)

## Overview

moseq3 (`/n/groups/datta/john/projects/moseq3`) is the lab's JAX/GPU depth-MoSeq
package: GPU PCA on egocentric depth-height frames, then an AR-HMM (via
`jax-moseq`) on the PC scores to discover behavioral syllables. This is the
**depth** pipeline. For keypoint tracking data use the **keypoint-moseq** skill
instead.

Core flow: **depth frames → GPU PCA → whiten PC scores → AR-HMM (kappa tuned to
a target syllable duration) → per-frame syllable labels.**

## When to use

- Fitting an AR-HMM on extracted/size-normed depth frames with moseq3.
- Anything touching `GPUPCA`, `apply_pca`, `whiten_all`, `run_arhmm_model`,
  `_calculate_median_duration`.
- NOT for keypoint/pose data (→ keypoint-moseq) — though the AR-HMM backend
  (`jax_moseq.models.arhmm`) is shared.

## Environment (read this first — the install has a trap)

moseq3 depends on `jax-moseq[cuda]` (GPU JAX). **`uv pip install` of moseq3 alone
fails**: its `pyproject` pins `jax-moseq = { path = "pypi" }`, a directory that
does not exist — the real source resolves (via `uv.lock`) to the local checkout
`/home/joj144/projects/keypoint-moseq/jax-moseq`.

Two working recipes:

```bash
# A) use moseq3's own locked env
cd /n/groups/datta/john/projects/moseq3 && uv sync    # builds moseq3/.venv

# B) a fresh, purpose-built editable venv (both moseq3 + jax-moseq modifiable)
uv venv .venv-moseq3 --python 3.10
uv pip install --python .venv-moseq3 -e "/home/joj144/projects/keypoint-moseq/jax-moseq[cuda]"
uv pip install --python .venv-moseq3 -e /n/groups/datta/john/projects/moseq3 --no-deps
uv pip install --python .venv-moseq3 h5py "imageio[ffmpeg]" infomap matplotlib \
    numpy opencv-python scikit-image scikit-learn scikit-posthocs submitit tqdm
```

The bundled CUDA (jax-cuda12 ≈ CUDA 12.9) works on O2's current drivers
(verified on driver 570.144 / CUDA 12.8) — do NOT assume you need a
system-CUDA module or a `jax[cuda12-local]` build. GPU visibility is only
testable on a GPU node.

## Key API

`moseq3.pca`:
- `GPUPCA(n_components, batch_size=10_000, mini_batch_size=500)` — sufficient-
  statistics PCA. `partial_fit(X)` (call repeatedly, streaming), `finalize_fit()`,
  `transform(X)`, `.explained_variance_ratio_`, `.components_`, `plot_scree(pca)`,
  `plot_components(pca, h, w)`.
- `run_pca_preprocessing(...)` — OPTIONAL Gaussian + tail filter + max-height
  clip. Off by default in most pipelines; it writes a full preprocessed h5 and
  has **no hook in `apply_pca`**, so if you preprocess for the fit you must
  preprocess identically at apply, or you get a train/apply mismatch.
- `apply_pca(sessions_dict, pca, out_h5, h5_dataset_name='frames', batch_size=5000)`
  — per-session PC scores to one h5. Runs `transform` on **CPU** (numpy), GPU
  idle. `sessions_dict` = `{name: h5_path}`.
- `whiten_all(sessions_dict)` — `{name: (n, k) array}` → whitened dict.

`moseq3.model`:
- `run_arhmm_model(sessions_dict, kappa, latent_dim, num_iterations=100,
  robust_DOF=None, checkpoint_dir=None, checkpoint_frequency=None, gamma=1e3)`
  — init + Gibbs fit + unbatch. Returns the model dict with per-session labels
  under **`model['states']`** (a `{name: label_array}` dict). `sessions_dict` =
  `{name: (n, latent_dim) whitened scores}`.
- `_calculate_median_duration(states_dict)` — median syllable duration in
  **frames**; feed it `model['states']`.
- Hardcoded inside `init_arhmm_model` (not tunable through the API):
  `num_states=100`, `nlags=3`, `alpha=5.7`, and `gamma` default `1e3`.

`moseq3.io`: `compile_session_dict`, `load_pc_scores`.

## Kappa tuning (single stage — unlike keypoint-MoSeq)

Depth-MoSeq is **AR-HMM only** (no keypoint-SLDS full-model stage), so kappa is
tuned **once**. Binary-search kappa in log space to hit a target **median
syllable duration**, measured with `_calculate_median_duration(model['states'])`:

```python
log_lo, log_hi = 3, 18                      # 1e3 .. 1e18
for _ in range(max_probes):
    kappa = 10 ** ((log_lo + log_hi) / 2)
    model = run_arhmm_model(whitened, kappa, latent_dim, num_iterations=25)
    dur = _calculate_median_duration(model["states"])   # frames
    if target_min <= dur <= target_max: break
    if dur < target_min: log_lo = (log_lo + log_hi) / 2   # too short → MORE kappa
    else:                log_hi = (log_lo + log_hi) / 2    # too long  → LESS kappa
```

Pick the target from the animal's **behavioral timescale**, not a generic
default — e.g. the autocorrelation-decay tau or model-free changepoint interval
for that species/prep (rats ≈ 20–23 frames / ~700 ms at 30 fps; mice ≈ 10–12
frames). Convert ms→frames with the real fps.

## Canonical pipeline (gold standard)

Copy this and change the config block. Structure and gotcha-guards are load-
bearing — preserve them. Runs as a single GPU job.

```python
#!/usr/bin/env python
"""Canonical moseq3 depth AR-HMM pipeline: frames -> PCA -> whiten -> kappa -> labels."""
import pickle, json, os
from pathlib import Path
import numpy as np, h5py
from moseq3.pca import GPUPCA, plot_scree, plot_components, apply_pca, whiten_all
from moseq3.model import run_arhmm_model, _calculate_median_duration

# ---- config ----
SESSIONS = {...}          # {name: Path(frames_h5)}   (build from your manifest)
FRAMES_DS = "translated"  # dataset name inside each h5 (NOT moseq3's 'frames' default)
RES = (180, 180); D = RES[0]*RES[1]
OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)
PCA_FIT_FRAMES = 1_000_000; PCA_BATCH = 4000
N_COMP = 50; VAR_THRESH = 0.90
FPS = 30; TARGET_MIN, TARGET_MAX = 20.4, 23.4   # frames; set from behavioral timescale
KLOG_MIN, KLOG_MAX, MAX_PROBES, PROBE_ITERS, FINAL_ITERS = 3, 18, 12, 25, 100

# ---- GPU fail-fast: bad O2 nodes make JAX silently fall back to CPU ----
import jax
devs = jax.devices()
print("jax devices:", devs, flush=True)
assert any(d.platform == "gpu" for d in devs), f"No GPU visible to JAX ({devs}); refusing CPU run."

# ---- PCA fit: stream a subsample, one partial_fit per session ----
pca = GPUPCA(n_components=N_COMP, batch_size=PCA_BATCH, mini_batch_size=PCA_BATCH)  # equal => no vmap OOM
quota = PCA_FIT_FRAMES // len(SESSIONS)
for name, h5p in SESSIONS.items():
    with h5py.File(h5p) as f:
        ds = f[FRAMES_DS]; n = ds.shape[0]
        idx = np.arange(n) if quota >= n else np.unique(np.linspace(0, n-1, quota).astype(int))
        pca.partial_fit(ds[idx].reshape(len(idx), D).astype(np.float32))
pca.finalize_fit()
cum = np.cumsum(pca.explained_variance_ratio_)
assert cum[-1] >= VAR_THRESH, f"{cum[-1]:.3f} var in {N_COMP} comps < {VAR_THRESH} — something is wrong"
latent_dim = int(np.searchsorted(cum, VAR_THRESH) + 1)
print("latent_dim", latent_dim, "cumvar", float(cum[latent_dim-1]), flush=True)
pickle.dump(pca, open(OUT/"pca_model.pkl","wb"))
plot_scree(pca).savefig(OUT/"scree.png"); plot_components(pca, *RES).savefig(OUT/"components.png")
# INSPECT scree/components: float32 covariance at ~1M frames can distort them.

# ---- apply + whiten (un-centered; whiten_all does the covariance/Cholesky) ----
apply_pca({n: str(p) for n,p in SESSIONS.items()}, pca, str(OUT/"pc.h5"),
          h5_dataset_name=FRAMES_DS)
with h5py.File(OUT/"pc.h5") as f:
    scores = {k: f[k][:, :latent_dim].astype(np.float64) for k in f}
whitened = whiten_all(scores)      # applied to raw (un-centered) scores; cov -> I

# ---- kappa search (single stage) then final fit ----
lo, hi = KLOG_MIN, KLOG_MAX; kappa = 10**((lo+hi)/2)
for _ in range(MAX_PROBES):
    kappa = 10**((lo+hi)/2)
    dur = _calculate_median_duration(run_arhmm_model(whitened, kappa, latent_dim, num_iterations=PROBE_ITERS)["states"])
    print(f"kappa {kappa:.2e} -> {dur:.1f} frames", flush=True)
    if TARGET_MIN <= dur <= TARGET_MAX: break
    lo, hi = ((lo+hi)/2, hi) if dur < TARGET_MIN else (lo, (lo+hi)/2)
model = run_arhmm_model(whitened, kappa, latent_dim, num_iterations=FINAL_ITERS)  # atomic; no mid-fit resume
labels = model["states"]
pickle.dump(model, open(OUT/"arhmm_model.pkl","wb"))
with h5py.File(OUT/"syllables.h5","w") as f:
    for k,z in labels.items(): f.create_dataset(k, data=np.asarray(z).astype(np.uint16))
print("done; median", _calculate_median_duration(labels), "frames", flush=True)
```

## Gotchas

- **Install:** `uv pip install moseq3` fails on the dead `jax-moseq={path=pypi}`
  source. Use `uv sync` or install jax-moseq editable first + moseq3 `--no-deps`
  (see Environment).
- **GPUPCA OOM at high resolution:** `partial_fit` `vmap`s the per-mini-batch
  second moment, materializing `batch_size/mini_batch_size` separate `D×D`
  matrices at once. At `D=32,400` each `D×D` ≈ 4.2 GB, so the defaults
  (`batch_size=10000, mini_batch_size=500` → 20 matrices ≈ 84 GB) OOM. **Set
  `mini_batch_size == batch_size`** (one transient `D×D`). Fine for small mouse
  D (6,400); required for rat-scale D.
- **Can't flatten all frames:** materializing every frame flat is TB-scale.
  Fit PCA on a subsample streamed via repeated `partial_fit` (one call per
  session), not `run_pca` on a giant array.
- **`explained_variance_ratio_` is relative to the full-D trace** (all pixels),
  so cumsum≥threshold is a true fraction; if the threshold is unreachable in
  `n_components`, that's a data problem — hard-error, don't silently cap.
- **Whiten is applied to UN-centered scores.** `whiten_all` forms the covariance
  (`np.cov`, ddof=1) and Choleskys it, then applies `x @ inv(W.T)` to the raw
  scores (mean preserved; the AR-HMM absorbs it). Do NOT mean-center before
  applying. It loads everything into a dict in memory (fine — scores are small).
- **`apply_pca` default `h5_dataset_name='frames'`** — pass your real dataset
  name or every read `KeyError`s. Its `transform` is CPU/numpy (GPU idle).
- **`run_arhmm_model` return:** the labels are `model['states']` (unbatched
  `{name: array}`), not the return value itself; `_calculate_median_duration`
  wants that dict.
- **`robust_DOF` is broken:** `model.py` references `robust_arhmm` without
  importing it → `NameError`. Don't use robust. `num_states/nlags/alpha/gamma`
  are hardcoded (syllable count capped at 100).
- **O2 GPU nodes are flaky → JAX silently uses CPU.** A bad node throws
  `cuInit → CUDA_ERROR_NO_DEVICE` (or `CUDA_ERROR_ECC_UNCORRECTABLE`) and JAX
  falls back to CPU (intractable at scale) instead of crashing. **Always assert
  a CUDA device at startup and hard-crash** (see canonical script). On failure,
  resubmit (`--exclude` the bad node) — the broad pool lands elsewhere. A
  polluted `LD_LIBRARY_PATH` (ollama/vgl CUDA) is usually NOT the cause; the
  bundled CUDA works. Don't pin a GPU type (sits PENDING).
- **The AR-HMM is the GPU-memory driver, not PCA.** It batches all sessions into
  `(n_sessions, max_T, num_states)` and `jax-moseq` forces `jax_enable_x64` →
  float64 forward-backward ≈ 8 GB per message array, several at once (~30–40 GB
  for ~10M frames). Use a large-memory GPU; if it OOMs, reduce with
  `jax_moseq.utils.set_mixed_map_iters(N)` (serial chunks). PCA peak is only
  ~10–13 GB by comparison.
- **No mid-fit resume:** `fit_arhmm_model` only *writes* checkpoint pickles;
  there is no restore path and `run_arhmm_model` always re-inits. The final fit
  is atomic. Make stages resumable at the *stage* level (skip if the output h5
  exists) and persist a kappa-search state JSON (write temp + `os.replace` for
  atomicity).

## SLURM on O2 (see also the using-slurm skill)

- Broad pool, don't pin type: `-p gpu_quad,gpu --qos=gpuquad_qos --gres=gpu:1`.
  For a long **atomic** final fit, omit `gpu_requeue` (preemption restarts it);
  for short smoke jobs `gpu_requeue` is fine and picks up fast.
- `export XLA_PYTHON_CLIENT_PREALLOCATE=false`; run `python -u`; `unset
  SLURM_CPU_BIND SLURM_CPU_BIND_TYPE` if submitting from an interactive node.
- Exclude 16 GB V100 nodes (too small) and any node caught throwing
  NO_DEVICE/ECC.

## Success criteria

- `jax devices` shows a CUDA device (never silently CPU).
- Median syllable duration matches the target behavioral timescale.
- `latent_dim` is sane (depth-MoSeq is typically ~8–15 PCs at 90% variance);
  scree/components plots look like real pose components, not noise.
- Syllable-frequency spectrum is power-law-ish; crowd movies / averaged poses
  are interpretable behaviors.

## Smoke first

Before a full multi-hour run, run the pipeline on ~3 sessions (a `--limit`) to a
**throwaway results dir** (so skip-if-done doesn't poison the real run) and
confirm: GPU visible, all stages run, all outputs written, kappa converges,
`pipeline complete`.
