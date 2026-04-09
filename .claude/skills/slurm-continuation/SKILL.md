---
name: slurm-continuation
description: Monitor long-running jobs and automatically hand off to a fresh SLURM job with a new Claude session when the current allocation is running out. Use when running jobs that may outlast the current SLURM allocation.
---

# SLURM Continuation Pattern

When running long jobs that may outlast the current SLURM allocation, use this pattern to automatically hand off to a continuation job. The continuation job launches a fresh Claude that reads a context file and picks up where the previous session left off — and can itself launch another continuation, creating a recursive chain.

## When to Use

- A job is running that will likely exceed the current SLURM time limit
- You need unattended monitoring with automatic failover
- The user asks you to "keep it running" or "make sure it finishes"

## The Pattern

### 1. Write a Context File

Write a markdown file to a stable path (not /tmp) with everything a fresh Claude needs:

```markdown
# [Job Name] Continuation Context

## What's Happening
[One paragraph describing the operation]

## Details
- Project path: /path/to/project
- Commands being run / API endpoints being used
- IDs / files being processed
- Current progress (update this as you monitor)

## What To Do When Done
1. Verify results (specific commands)
2. Clean up (specific commands)
3. Write results to [specific path]

## How To Check SLURM Time Remaining
[Include the exact command]

## How To Launch Another Continuation
[Include the complete sbatch script — see below]

## Important Notes
[Anything the next Claude needs to know]
```

**Key principles:**
- Use absolute paths everywhere
- Include exact commands, not descriptions
- Update the progress section as you monitor
- Include the full sbatch script inline so the next Claude can launch yet another continuation

### 2. Monitor with a Loop

Use `/loop` or `CronCreate` to check every N minutes:

```
/loop 20m Check job status, SLURM time remaining, and confirm progress
```

Each check should:
1. Check if the job process is still running
2. Check SLURM time remaining
3. Verify progress (file sizes, mtimes, log output)
4. If done: verify, clean up, write results
5. If running low on time: launch continuation

### 3. Check SLURM Time Remaining

Parse elapsed and limit, handling the day format (`D-HH:MM:SS`):

```bash
squeue -u $USER -o "%.18i %.30j %.10M %.10l" | grep JOB_NAME | python3 -c "
import sys, re
line = sys.stdin.read().strip().split()
e, l = line[-2], line[-1]
def to_min(t):
    m = re.match(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', t)
    if not m: return 0
    d, h, mn, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return d*1440 + h*60 + mn + s/60
r = to_min(l) - to_min(e)
print(f'Remaining: {int(r)} min ({r/60:.1f} hrs)')
if r < 40: print('WARNING: NEED CONTINUATION JOB')
"
```

### 4. Launch Continuation Job

When time is running low (< 40 minutes, or less than one monitoring interval + safety margin), submit the continuation. Match the resource requirements of whatever the job needs:

```bash
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH -J continuation-job-name
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -o /path/to/continuation_%j.log

cd /path/to/project

claude -p "Read /path/to/context-file.md for full context. You are continuing a [job description]. [Specific instructions for what to check, restart, and monitor]. If YOUR slurm job is running low on time (less than 40 min remaining), launch another continuation job using the sbatch script in the context file."
SBATCH_EOF
```

**Important sbatch parameters:**
- Match the resource requirements (GPU, memory, CPUs) of the original job
- Use `--nodelist` if specific hardware is needed (e.g., Ampere GPUs)
- Request enough time (24h is usually safe)
- Log output to a persistent path (not /tmp)

### 5. The Continuation Claude's Job

The `claude -p` prompt should instruct the continuation Claude to:

1. Read the context file
2. Check if the previous job completed (process still running? results exist?)
3. If not completed: restart any services, re-launch the operation on remaining items
4. Monitor with the same loop pattern
5. If ITS allocation is running low: launch ANOTHER continuation (recursive)
6. When done: verify, clean up, write results

### 6. Verify Progress Without Reading Locked Files

Files locked by a writer can't always be read. Use file stats instead:

```bash
# Check if a file is actively being written (mtime changing)
m1=$(stat --format='%Y' /path/to/file)
sleep 3
m2=$(stat --format='%Y' /path/to/file)
if [ "$m1" != "$m2" ]; then echo "ACTIVE"; else echo "idle"; fi

# Check file size for progress
stat --format='%s' /path/to/file | numfmt --to=iec
```

### 7. When to Launch Proactively

Don't wait until < 40 minutes to submit the continuation job. Submit it early so it's queued:
- The continuation job will check if work is already done before restarting anything
- Better to have an idle continuation job that exits quickly than to lose progress
- Submit when you estimate the current job won't finish in the remaining time

## Checklist Before Launching

- [ ] Context file written to persistent storage (not /tmp)
- [ ] Context file includes ALL details: paths, IDs, commands, sbatch script
- [ ] Context file progress section is up to date
- [ ] sbatch script matches current resource requirements
- [ ] Continuation prompt tells Claude to check if work is already done
- [ ] Continuation prompt tells Claude to launch another continuation if needed
- [ ] Log output path is persistent
