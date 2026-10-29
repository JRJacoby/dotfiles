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

## Checking that a compute node can reach the SLURM controller

Test with a real RPC, not `scontrol ping`. From O2 compute nodes `scontrol ping`
reports both controllers DOWN even while `squeue`, `sacct` and `sbatch` work
normally, so it is not evidence of an outage:

```bash
squeue -u $USER            # any job list, even just your own session, means RPCs work
sbatch --test-only --wrap='true' -t 0:01:00 -p short   # validates a request, creates no job
```

If those fail with `Unexpected message received`, the compute node's SLURM client is
older than the controller's protocol. That happens after a controller upgrade, when
login nodes are updated and compute nodes lag behind. Compare `sbatch --version`
here with the login node's:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 login.o2.rc.hms.harvard.edu 'sbatch --version'
```

Do not conclude the cluster is down. If TCP to port 6817 connects and
`munge -n | unmunge` returns Success while every RPC still fails, it is a version
skew, not an outage. Allocating a fresh interactive session does not help, because
the new node has the same stale client. Use the workaround below until the versions
match again.

### Workaround: submit over ssh from a login node

Passwordless ssh to `login.o2.rc.hms.harvard.edu` works from compute nodes. Every
SLURM interaction (`sbatch`, `squeue`, `sacct`, `scancel`) has to go through it.

**Write the remote command to a file and pipe it to `bash -s`** rather than
inlining it as an ssh argument. Long inline commands invite quoting mistakes, and
a wrong one silently submits from the wrong directory:

```bash
cat > /tmp/submit.sh <<'EOF'
set -e
cd /absolute/path/to/project     # sets SLURM_SUBMIT_DIR for the job
sbatch scripts/<unit>/run.sbatch <args>
EOF
ssh -o BatchMode=yes login.o2.rc.hms.harvard.edu 'bash -s' < /tmp/submit.sh
```

`cd` to the submission directory inside the script is required: ssh lands in
`$HOME`, so a relative script path fails with `Unable to open file`, and any job
reading `SLURM_SUBMIT_DIR` would otherwise resolve it to the home directory.

Note the login banner prints to stderr on every ssh call and clutters output;
filter it if you are parsing results.

## ⚠️ NEVER size a job from a number that describes the machine

**The host's resources are not yours. Your allocation's are.** Every number
below describes the *node* or the *cluster defaults*, and reading any of them
as "what this job has" is wrong:

| DO NOT read | it reports | what it is NOT |
|---|---|---|
| `free -g`, `/proc/meminfo` | the whole node's RAM, shared with every other job | your memory |
| `nproc` | every core on the node | your cores |
| `scontrol show partition <p>` → `DefMemPerNode` | a partition-level default that a cluster-wide `DefMemPerCPU` silently overrides | your default |
| `MaxMemPerNode` | the ceiling a request may ask for | what you were given |
| a comment, README or calibration note | what someone measured once, on some configuration | this job's peak |

**Read these instead.**

```bash
# What THIS interactive session has:
echo "$SLURM_MEM_PER_NODE MB, $SLURM_CPUS_ON_NODE cpus"      # e.g. 24576 MB, 4
# The cgroup that will actually kill you:
C=$(cut -d: -f3 /proc/self/cgroup | head -1)
cat /sys/fs/cgroup$C/memory.max 2>/dev/null

# What a job you SUBMITTED was actually granted -- always verify after sbatch:
scontrol show job <jobid> | tr ' ' '\n' | rg 'ReqTRES|MinMemory'

# What a finished job actually PEAKED at -- the only real sizing evidence:
sacct -j <jobid> --format=JobID%18,State,Elapsed,MaxRSS,ReqMem
```

### The failure modes this prevents

- **`DefMemPerNode=UNLIMITED` does not mean unlimited.** The cluster-wide
  `DefMemPerCPU` (1000 MB on O2) binds instead, so omitting `--mem` on an
  `-c 8` job grants **8 GB**, not the node. Memory is a consumable resource
  here (`SelectTypeParameters=CR_CORE_MEMORY`).
- **Running a big job "locally" on an interactive node.** A `srun`/`salloc`
  session is itself inside a cgroup, typically tens of GB. An 80 GB job dies
  with **exit 137 / Killed** there even when `free -g` shows hundreds of GB
  free, because that free memory belongs to other jobs. Anything near the
  allocation's limit goes to `sbatch`, not to the current shell.
- **Sizing `--mem` from a calibration note.** Notes describe the configuration
  they were taken on. A different code path in the same program can have a
  completely different appetite -- e.g. a compiled resampler that holds every
  session's message arrays at once versus a Python one that does them one at a
  time. If the note and `sacct` disagree, `sacct` wins.
- **A cap gets hit exactly.** When `MaxRSS` equals the request to the byte
  (e.g. `50,330,616K` against `--mem=48G`), the job was killed at the ceiling
  and its true peak is **unknown and higher**. Do not treat that number as a
  measurement; raise the ceiling until a job completes, then read `MaxRSS`.

**Rule of thumb:** before sizing anything, ask "is this number about the
machine, the config, or my job?" Only the third kind counts. And after every
`sbatch`, confirm the grant with `scontrol show job` rather than assuming the
directive took.

## Partitions

| Partition | GPUs | Time Limit | Use for |
|-----------|------|------------|---------|
| `gpu_quad` | V100, RTX 8000, L40s, A40, A100 (33 nodes) | 5 days | Most GPU work |
| `gpu` | Mixed (11 nodes) | 5 days | Overflow if gpu_quad is full |
| `short` | CPU only (268 nodes) | 12 hours | Quick CPU jobs |
| `medium` | CPU only (262 nodes) | 5 days | Longer CPU jobs |
| `interactive` | CPU only | 12 hours | Interactive sessions |

## Default GPU Job Setup

Use this as the starting configuration for short, disposable, or checkpoint-safe
GPU jobs. It submits to three partitions for fast scheduling and matches the
user's `sa` interactive allocation setup.

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
- **Partition choice:** use `gpu_quad,gpu` for long atomic runs, final/publication
  fits, or anything that cannot safely restart. Add `gpu_requeue` only when a
  preemption/requeue is disposable or the worker checkpoints and resumes safely,
  and then also add `--requeue` (see the preemption caveat below): without it a
  preempted job is cancelled, not requeued.
- **QOS:** Always `gpuquad_qos` — this works across all three GPU partitions.
- **Defaults:** 1 GPU, 4 CPUs, 24GB RAM. Adjust mem/time as needed per job.
- **Do not override `CUDA_VISIBLE_DEVICES`:** O2 sets it to the GPU assigned to
  the job.
- **Unset the inherited GPU vars before submitting from a GPU session** (see below) — otherwise
  the job can silently run on CPU.

### A job can land on a GPU node and silently train on CPU

Some nodes pass `nvidia-smi` (the batch script prints a GPU name normally) while
`torch.cuda.is_available()` returns False inside the job. Frameworks then **fall back to CPU
instead of erroring**: the job "runs", ~30× slower, and its outputs look valid. Observed on O2 with
one node reproducing it across three separate jobs while every job on other nodes was fine — so
treat it as a per-node fault, not a code or environment problem.

**Always log the resolved device in the job** (`device cuda` / `device cpu`, plus
`torch.cuda.get_device_name()`) and check it before trusting a run. `nvidia-smi` in the batch
script is *not* sufficient evidence — it reports the allocated GPU correctly even when the training
process cannot use it. A per-epoch wall time far above expectation is the other tell.

When a node reproduces it, exclude that node and resubmit:

```bash
#SBATCH --exclude=compute-gc-17-242     # add nodes as they are found bad
```

Also unset the inherited CPU-binding vars (and, harmlessly, the GPU ones) before submitting from an
interactive allocation:

```bash
unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
```

## Request a minimum GPU VRAM without pinning a card or node

O2 exposes GPU memory as a non-consumable `vram` GRES. When capacity is the
constraint, request the minimum card memory alongside the GPU count:

```bash
#SBATCH --gres=gpu:1,vram:48G
```

This means **at least 48 GB of advertised VRAM**; it does not select an exact GPU
model, architecture, or speed. O2's GPU partitions are heterogeneous, so inspect
the eligible pool and leave headroom above measured peak usage:

```bash
sinfo -p gpu,gpu_quad,gpu_requeue --Format=nodehost,gres:40
```

Prefer this selector over a drifting node exclusion list when the only requirement
is memory. Use `sbatch --test-only ... --gres=gpu:1,vram:48G` to validate a planned
request without creating a job. See the official [O2 GPU resource guide](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1629290761/Using%20O2%20GPU%20resources).

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

**Don't trust the scheduler's estimated start time.** `sbatch --test-only` and
`squeue --start` give a start-time *estimate*, and it is systematically pessimistic —
often by hours or days. The estimate assumes every currently-running job uses its
*full* requested walltime, but the same over-requesting described above means most
jobs finish long before their `-t`, freeing nodes early, so your job backfills much
sooner than predicted. An estimate of "tomorrow" routinely turns into minutes,
especially for a tight (short `-t`, modest `--mem`) job or one on `gpu_requeue`. So
treat the estimate as a worst case only — never conclude "that's a multi-day wait, use
a different/scarcer resource" from it. **When in doubt, just submit and watch
`squeue`** (a real submission reveals the truth; `--test-only` can't, because it
doesn't model backfill or preemption).

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

If the requirement is merely enough GPU memory, use the `vram` GRES above rather
than an allowlist. It does not guarantee Ampere-class performance, though.

**When architecture/speed itself matters, pin to Ampere+ with an allowlist
`--nodelist`, not an exclude list, and never by GPU type.** `gpu_requeue` overlays
the *entire* GPU fleet, including a substantial
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

**A multi-node `--nodelist` needs an explicit `--nodes`/`-N`.** With more than one
node listed, `sbatch` fails with `Node count specification invalid` unless you also
give the count (e.g. `-N 1`, as above). With `-N 1`, SLURM treats the list as a
*candidate pool*: it picks one node from the list that fits the request and does
**not** require every listed node to be in one partition. Verified here with `sbatch
--test-only` (and a real `salloc`): the full cross-partition allowlist above schedules
cleanly onto a `gpu_quad` node, silently ignoring the listed nodes that live in other
partitions. The only requirement is that at least one listed node belongs to a
requested partition. So the `sa --ampere` allowlist transfers to `sbatch` verbatim —
just keep `-N 1`. (A tiny pinned nodelist still defeats fast pickup, though: if those
few nodes are busy you wait instead of backfilling across the pool.)

`salloc` and `sbatch` behave identically on nodelists here; the one real asymmetry is
that `salloc` has **no `--test-only`**, so you can dry-run-validate an `sbatch` request
but not a `salloc` one. When dry-running with `--wrap`, pass an explicit `-t` —
`sbatch --test-only --wrap='true' -t 0:10:00 …` — since there is no script to supply
`#SBATCH --time`, and a missing time limit fails before any node/partition check.

**Preemption caveat.** `gpu_requeue` jobs can be preempted. The partition's
`PreemptMode=REQUEUE` requeues a preempted job only if the job is requeueable, and O2
sets `JobRequeue=0` cluster-wide, so by default a preempted job is **cancelled**: it
ends PREEMPTED and anything depending on it is killed. Ask for requeue explicitly, and
append to the log so the pre-preemption output survives the restart:

```bash
#SBATCH --requeue
#SBATCH --open-mode=append
```

Confirm it took: `scontrol show job <id> | tr ' ' '\n' | rg Requeue=` must show
`Requeue=1`. A requeued job restarts the script from the top (`SLURM_RESTART_COUNT`
counts restarts), so make the worker idempotent — overwrite its own partials and
skip already-finished work via a done-sentinel — so a requeue is just lost time, not
corruption. A tight `--time` keeps requeued jobs backfilling fast.

When submitting these from an interactive compute node, remember to `unset
SLURM_CPU_BIND SLURM_CPU_BIND_TYPE` first (see CPU binding error below).

## Chaining jobs with dependencies

To start a job only after another succeeds (e.g. a CPU render after a GPU fit):

```bash
fit=$(sbatch --parsable fit.sbatch)
sbatch --dependency=afterok:${fit%%;*} render.sbatch    # strip any ";cluster" suffix
```

The dependent job waits as `PENDING (Dependency)`. O2 sets
`DependencyParameters=kill_invalid_depend`, so if the parent fails, times out or is
cancelled (including a preempted job that was not requeueable), the dependent job is
cancelled automatically rather than left pending forever. After a failure, resubmit
both. `sbatch --test-only` cannot check a dependency on a job that does not exist
yet, so dry-run the dependent request without `--dependency`.

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

### `/tmp` is node-local — job output written there vanishes

Every compute node has its own `/tmp`. A job that writes results to `/tmp/foo.jsonl` writes
them to *that node's* disk, so the file does not exist on the submit host or on any other
node, and it is wiped when the node is reclaimed. The job itself succeeds — exit code 0, a
log full of normal progress lines — and only the output is missing, which makes this read
like a mysterious silent failure rather than a path mistake.

**Always point job output at a shared filesystem** (`/n/groups/...`, `$HOME`, the project
tree). This applies to anything the job is supposed to leave behind: result files, caches you
intend to reuse, checkpoints, and plots.

```bash
# WRONG — lands on the compute node's local disk, unreachable afterwards
sbatch run.sbatch --out /tmp/results.jsonl

# RIGHT — shared path, readable from anywhere
sbatch run.sbatch --out "$(pwd)/tables/2026-01-01_analysis/results.jsonl"
```

This bites hardest with agent/session scratch directories, which are usually under `/tmp` and
are perfectly good for *local* work on the submit host but are invisible to a SLURM job.
Before submitting, check that every output path the job will write is on shared storage.

`/tmp` is still the right place for genuinely throwaway per-job scratch that the job itself
reads back within the same allocation (unpacked archives, intermediate shards) — node-local
disk is often faster than the shared filesystem. The rule is about anything you want *after*
the job ends.

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
