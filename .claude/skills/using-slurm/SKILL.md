---
name: using-slurm
description: Submit and manage SLURM jobs on the O2 cluster. ALWAYS use this skill when submitting ANY SLURM job (sbatch, srun, salloc, submitit) to ensure correct partitions, QOS, and resource settings.
---

# Using SLURM on O2

**IMPORTANT**: Do NOT use SLURM unless the user explicitly asks you to, or a
domain-specific skill (like moseq2) has SLURM built into its workflow. The user
is typically already on an interactive GPU node and wants you to run things
directly. This skill is reference material for when SLURM is actually needed.

The cluster is **O2** (Harvard Medical School Research Computing). Jobs are
submitted via SLURM, either directly with `sbatch`/`srun` or programmatically
via Python's `submitit` library.

## Partitions

| Partition | GPUs | Time Limit | Use for |
|-----------|------|------------|---------|
| `gpu_quad` | V100, RTX 8000, L40s, A40, A100 (33 nodes) | 5 days | Most GPU work |
| `gpu` | Mixed (11 nodes) | 5 days | Overflow if gpu_quad is full |
| `short` | CPU only (268 nodes) | 12 hours | Quick CPU jobs |
| `medium` | CPU only (262 nodes) | 5 days | Longer CPU jobs |
| `interactive` | CPU only | 12 hours | Interactive sessions |

## Default GPU Job Setup

**Always use this configuration for GPU jobs.** It submits to three partitions
for fastest scheduling and matches the user's `sa` interactive allocation setup.

```bash
#SBATCH -p gpu_quad,gpu,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -J my_job
```

Or via submitit:
```python
executor.update_parameters(
    slurm_partition="gpu_quad,gpu,gpu_requeue",
    slurm_qos="gpuquad_qos",
    gpus_per_node=1,
    cpus_per_task=4,
    slurm_mem="24G",
    timeout_min=120,
    slurm_job_name="my_job",
)
```

**Key points:**
- **Three partitions:** `gpu_quad,gpu,gpu_requeue` — SLURM picks whichever has
  availability first. `gpu_requeue` is preemptible but often starts immediately.
- **QOS:** Always `gpuquad_qos` — this works across all three GPU partitions.
- **Defaults:** 1 GPU, 4 CPUs, 24GB RAM. Adjust mem/time as needed per job.
- **Do not** submit to a single GPU partition unless there's a specific reason.

## When the user asks for a "GPU job" or to "request a GPU node"

That phrasing IS the explicit ask that licenses SLURM. It means: submit a batch
job with the Default GPU Job Setup above — the `sbatch` equivalent of their
interactive `sa --ampere` helper — rather than running on the current node.
(If the current node is broken, e.g. an uncorrectable ECC GPU error, batching is
also the fix — add the bad node to `--exclude` and submit.)

**Size `--time` from evidence, not the 2h default.** An oversized wall is the
main reason a job sits PENDING, so set `-t` from how long the work actually takes:

- **Already measured it this session** (a prior run, or timed epochs/steps of the
  same job) → extrapolate from that rate and add ~30–50% margin for warmup/compile
  and a slightly slower node. e.g. a training run that plateaued by ~40 min →
  `-t 1:30:00`, not `2:00:00`. Reuse the number you already have; don't re-profile.
- **No measurement yet** → profile a short pass on an interactive node first (see
  Fast-pickup section), or fall back to the Default setup's `-t 2:00:00`.

Tight `--time` and `--mem` backfill faster across `gpu_quad,gpu,gpu_requeue`. For
many independent jobs at once, see the Fast-pickup section below.

**After the job runs, check what it actually used and recalibrate.** Over-requesting
is not free: O2 runs an automated fairshare system that *lowers the user's
scheduling priority* when reservations chronically exceed usage — so a habit of
fat `-t`/`--mem` makes every future job sit longer in PENDING. Once a job finishes
(or after the first of a batch), compare request vs. actual and tighten the next
submission:

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,Timelimit,MaxRSS,ReqMem
```

- `Elapsed` vs `Timelimit` → how much of the wall you actually needed. If you used
  40 min of a 2h request, drop the next `-t` toward ~1h.
- `MaxRSS` vs `ReqMem` → peak host RAM vs reserved. If a 24G request peaked at 6G,
  request 8–10G next time.

Feed these numbers back into the per-workload `-t`/`--mem` so future runs both
schedule faster *and* keep priority high. (`MaxRSS` is per-step — read it off the
`.batch` row, not the parent.)

## Fast-pickup parallel jobs: tight resources on gpu_requeue (Ampere+)

When you have N independent GPU jobs (e.g. one per video/session/shard) and want
them all running **in parallel, as soon as possible**, submit each as its own job
to `gpu_requeue` with the tightest resource request that still fits. `gpu_requeue`
is a large preemptible pool, so a tight job backfills almost immediately — often
faster than waiting for a dedicated slot in `gpu_quad`. The wins compound: N tight
jobs running at once finish in roughly `1/N` of the sequential wall time.

**Profile first, then size tight.** Before submitting, run a short profiling pass
(a few hundred frames/steps) on an interactive GPU node and measure the three
numbers that set the request:

- **Host RAM** → `--mem`. Watch steady-state RSS (e.g. `/proc/self/status` VmRSS or
  `sacct ... MaxRSS` on a test job). Request that + headroom for model load, not the
  partition default. A 16G request schedules faster than a 24G one.
- **Peak GPU VRAM** → which nodes qualify. Most jobs fit in <16G; every Ampere+
  node has ≥40G, so VRAM is rarely the binding constraint — but confirm it so you
  don't need to restrict the pool.
- **Throughput** → `--time`. Time one full job from the measured rate (`total_items /
  items_per_sec`) and add margin for warmup/compile and a slightly slower node. A
  tight wall (e.g. `1:30:00` for a ~50 min job) backfills far faster than a 5-day
  default and survives requeue cheaply. **Oversized mem/time/cpu requests are the
  main reason a job sits PENDING** — size them from the profile, not from habit.

**Pin to Ampere+ with an allowlist `--nodelist`, not an exclude list, and never by
GPU type.** `gpu_requeue` overlays the *entire* GPU fleet, including a substantial
number of pre-Ampere nodes that run training ~5–10× slower (~70–110 s/epoch vs ~8 s
on Ampere for the same job). An exclude list can't keep up — there are too many slow
nodes to enumerate and the set drifts. Instead allowlist the known-good Ampere nodes
(the exact set the `sa --ampere` helper uses) while keeping the broad partition trio
so `gpu_requeue` still gives fast preemptible pickup:

```bash
#SBATCH -p gpu_quad,gpu,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --nodelist=compute-gc-17-[249,252-254,239-240],compute-g-17-[162-163,166-171,200-205]
```

Do **not** pin a GPU *type* (e.g. `--gres=gpu:l40s:1`) to force a fast card: a single
type is often fully reserved, so the job sits PENDING on "unavailable" indefinitely.
The allowlist spans several Ampere types, so one is usually free. (The non-Ampere `sa`
path, when GPU arch doesn't matter, instead just excludes one bad node:
`-x compute-g-17-164`.)

This allowlist is the current known-good set; node membership drifts, so re-derive it
from `sa`'s definition (`declare -f sa`) or `sinfo` if jobs stop landing:

```bash
sinfo -p gpu_requeue -N -o "%N %G %t" | sort -u   # GPU type + state per node
```

**Do NOT pin a `--nodelist` that spans partitions in `sbatch`.** `salloc` (and the
interactive `sa --ampere` helper) tolerates a `--nodelist` whose nodes live in
different partitions, but `sbatch` rejects it with `Requested nodes not in this
partition` because no single partition contains all the listed nodes. Either submit
to one partition and `-x` the nodes you don't want (preferred — keeps the pool big),
or use a nodelist drawn entirely from one partition. A tiny pinned nodelist also
defeats fast pickup: if those few nodes are busy you wait, instead of backfilling
anywhere in the pool.

**Preemption caveat.** `gpu_requeue` jobs can be preempted and requeued, restarting
the script from the top. Make the worker idempotent — overwrite its own partials and
skip already-finished work via a done-sentinel — so a requeue is just lost time, not
corruption. A tight `--time` keeps requeued jobs backfilling fast.

When submitting these from an interactive compute node, remember to `unset
SLURM_CPU_BIND SLURM_CPU_BIND_TYPE` first (see CPU binding error below).

## Using submitit (Python)

`submitit` is the preferred way to submit SLURM jobs from Python scripts.
It handles serialization, log collection, and job arrays.

```python
import submitit

executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(
    slurm_partition="gpu_quad,gpu",
    slurm_qos="gpuquad_qos",
    gpus_per_node=1,
    cpus_per_task=4,
    slurm_mem="24G",
    timeout_min=60,
    slurm_job_name="shmoseq",
)

# Submit array of jobs
jobs = executor.map_array(my_function, arg_list_1, arg_list_2)
```

**Important:** The wrapper function passed to `map_array` must be **picklable**
(module-level function, no closures over unpicklable objects). Load heavy data
inside the function, not before submission.

## Known Issues and Fixes

### CPU binding error when submitting from compute nodes

**Symptom:**
```
srun: error: CPU binding outside of job step allocation, allocated CPUs are: 0x...
srun: error: Task launch for StepId=... failed on node ...: Unable to satisfy cpu bind request
```

**Cause:** When you're on an interactive compute node (e.g., via `srun --pty bash`),
SLURM sets `SLURM_CPU_BIND` and `SLURM_CPU_BIND_TYPE` environment variables
with the parent job's CPU mask. Child jobs submitted via `sbatch` or `submitit`
inherit these variables, but their new allocation has different CPUs, causing the
binding to fail.

**Fix:** Unset the inherited variables before submitting jobs:

```python
import os

# Clear inherited CPU binding (needed when submitting from interactive nodes)
for var in list(os.environ):
    if var.startswith("SLURM_CPU_BIND"):
        del os.environ[var]
```

Or in bash:
```bash
unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE
```

Place this **before** any `submitit` executor creation or `sbatch` calls.

### submitit `mem` deprecation warning

submitit now expects `slurm_mem` instead of `mem`:
```python
# Old (deprecated, triggers warning)
executor.update_parameters(mem="24G")

# New
executor.update_parameters(slurm_mem="24G")
```

## SLURM-first / local-fallback dispatcher pattern

For batch workloads where you have N independent per-session jobs and queue wait
routinely exceeds per-job wall time (e.g., a 100s job waiting 30+ min), use this
hybrid pattern: fire all jobs at SLURM, then race them locally from the current
node.

**Two-script layout:**

- **`submit_<step>_batch_slurm.py`** — one-shot. Scans pending sessions,
  submits one sbatch per session, exits. Does NOT poll.
- **`run_<step>_batch_local.py`** — polling loop on the current node. For each
  pending session:
  - `RUNNING|CONFIGURING|COMPLETING` → skip (don't race live SLURM)
  - `PENDING` → `scancel` the queued job + run in-process (beat SLURM to it)
  - not in queue → run in-process (SLURM failed or never picked up)
  - done → skip

  After each local run, `break` and rescan — queue state is stale. Sleep
  (`--scan-sleep 30`) only when every pending session is on live SLURM.

**Required mechanics:**
- **Job naming**: `<step>_<slug>` — consistent prefix lets `squeue -n`/`scancel -n`
  target the exact job without tracking jobids across runs.
- **`is_done(sdir)` check**: primary output file exists and mtime is newer than
  all inputs. Used by both drivers to find pending sessions.
- **Per-session lock file** (`.<step>_local_in_progress`): prevents two local
  drivers from grabbing the same session. `touch` on claim, `unlink` in `finally`.
  Stale lock (>1h) → override.
- **Worker is idempotent**: writes to `_tmp_*` and renames atomically, so a
  killed scancel'd job or crashed local run leaves no partial output.
- **Strip `SLURM_CPU_BIND*` envvars** in the submit script (see "CPU binding
  error" section above).

**Two concurrent local drivers** can chew from both ends of a long session list:
the local driver accepts a `--reverse` flag so one iterates forward, one backward.

**When it pays off:** queue wait > 3× per-job wall, OR a single slow/dead node
is holding up a handful of jobs at the tail of a batch. When SLURM is snappy
(`short` partition, low load), just let SLURM do its thing and don't bother.

## Monitoring Jobs

```bash
# Check your queue
squeue -u $USER

# Check a specific job array
squeue -j 12345678

# Check completed/failed jobs (after they leave the queue)
sacct -j 12345678 --format="JobID%20,State,Elapsed,MaxRSS,ExitCode"

# Check why a job is pending
squeue -j 12345678 --format="%.12i %.8T %R"
# Common reasons: (Priority), (Resources), (QOSMaxJobsPerUser)

# Cancel jobs
scancel 12345678          # cancel entire array
scancel 12345678_5        # cancel single array element
```

## Interactive GPU Sessions

```bash
srun --pty -p interactive,gpu_quad --qos=gpuquad_qos --gres=gpu:1 \
     -c 4 --mem=24G -t 0-04:00 bash
```

## Node Exclusions

Some nodes may be problematic. Exclude them with:
```python
executor.update_parameters(slurm_exclude="compute-g-17-164")
```

Or in sbatch:
```bash
#SBATCH --exclude=compute-g-17-164
```
