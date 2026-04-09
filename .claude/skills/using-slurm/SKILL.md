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
