---
name: ln
description: Record results, decisions, or observations in the central lab notebook in the Obsidian vault. Use when the user says /ln followed by a message describing what happened.
---

# Lab Notebook Entry

The user is dictating a lab notebook entry. Your job: turn their message, together with the session's work, into a clean, technical notebook entry and append it to the **single central lab notebook** shared across all projects.

## Where the notebook lives

The central notebook is a single file in the user's Obsidian vault:

```
/home/joj144/Documents/Main Remote/Lab Notebook.md
```

This vault lives on the **O2 / HMS Harvard shared filesystem** (the user's work environment). All `/ln` entries from any project, in any working directory, go into this one file.

**If the vault or file is missing:** check the host and context before assuming an error. On a personal machine or any environment where the O2 vault isn't mounted, this skill likely doesn't apply — the central notebook is a work-environment artifact. Tell the user the central notebook isn't reachable here and ask how they want to proceed rather than creating a stray file.

**Ignore project-scoped lab notebooks.** Older projects may have their own `lab_notebook.md` in the project root (and there may be legacy per-project notebooks in the vault too). These are frozen history — do not read from or write to them. All new entries go to the central notebook only.

## What to record

`/ln` records the **work done on this project this session** (since the last notebook entry for it), reconstructed from the conversation — **not** just the text of the `/ln` message. The message is the trigger and may add emphasis or flag a specific thing to highlight, but even a bare `/ln` means: summarize everything of record we did this session. Reconstruct the account from the session itself — scripts run, files produced, decisions made, what was tried — and write that, weighting toward whatever the message emphasizes. Pulling this from the conversation is expected, not speculation.

Cover these, drawing on the whole session:
- **Results**: quantitative outcomes, visual assessments, pass/fail
- **Decisions**: "we're going with X", "decided to Y", "dropping Z"
- **Observations**: things noticed, hypotheses, caveats
- **Status updates**: "finished X", "started Y", "blocked on Z"
- **Process / time spent**: where the session's time actually went — the account you'd give your boss of what got done. What was built or run, what had to be tracked down or reconstructed, the dead ends, what got debugged and how, the rework. Record this in real detail — it is a primary purpose of the entry, not an afterthought — even when the final result is small. Let it run as long as the work warrants.

## Which project the entry belongs to

Every entry is filed under a project heading. Determine the project from conversation context (the work being discussed, the cwd, recent file paths). If it's genuinely unclear, **ask the user** which project this entry is for before writing. Use the project's normal name (e.g. `SCHEMA mice kpms`, `Lesion project`, `PigSeq`).

## Reasoning and gaps — think before writing

Before writing, work out the **reasoning**, not just the actions: why each thing was done, why a decision went the way it did, what a result means. A good entry captures *why*, not only *what*.

If the reasoning — or any required context below — isn't clear from the message or the session, **ask before writing**. Don't paper over a gap with a guess or a vague phrase: a one-line question to the user is always preferable to inventing a rationale. Gather the answers, then write.

## Required context per entry type

Each entry type requires specific supporting context. If it isn't clear from the message or the session, **ask before writing**:

### Results
- **What actions led to the result**: which script was run, with what arguments, on what data
- **What files were produced**: output paths (h5, npz, mp4, model checkpoints, etc.)
- **The result itself**: numbers, visual assessment, comparison to prior results

### Decisions
- **The reasoning**: why this choice over alternatives, what evidence or experience informed it
- **What it means going forward**: implications for the pipeline or next steps

### Observations
- **Possible interpretations**: what could explain the observation, competing hypotheses
- **How to distinguish**: what experiment or check would confirm/refute each interpretation

### Status updates
- **What was done**: script, command, SLURM job ID, wall time
- **Next steps**: what comes after this, what's blocking, what the user plans to do next

### Process / time spent
- **Where the time went**: the things worked on this session and why they took the effort they did — what was located or reconstructed, what was iterated, what broke, the dead ends and the debugging that ate time.
- **Detail follows effort**: a session full of hunting and debugging warrants a full account; a clean quick result warrants a short one. Do not drop the dead ends, the things you had to find, or the debugging just to keep the entry short — those are exactly what "where did the time go" means.

## How to write the entry

The notebook is organized **by date, then by project**:

```markdown
## 2026-07-06
### SCHEMA mice kpms
- entry...
### Lesion project
- entry...
```

1. **Verify today's date with a shell command** — never trust your own sense of the date:
   ```bash
   TZ="America/New_York" date +%Y-%m-%d
   ```

2. **Read the end of the notebook** (last ~40 lines) to find:
   - The `## YYYY-MM-DD` heading for today. If it exists, work under it; if not, create a new `## YYYY-MM-DD` section at the end of the file.
   - Under today's date, the `### ProjectName` subheading for this entry's project. If it exists, append under it; if not, add a new `### ProjectName` subheading under today's date.
   - Any `### Next steps` block belonging to that project — insert the new entry BEFORE it.

3. **Write the entry** as a paragraph or bulleted list under a `####` subheading if the content warrants one, or as a plain entry if it's brief. Match the existing notebook style:
   - Technical and complete — cover the full arc of the work, including dead ends and debugging; let length track how involved the work actually was
   - Include specific numbers, file paths, session IDs, script names, SLURM job IDs
   - Include what was run: exact script path and key arguments
   - Include what was produced: output file paths
   - Use backtick formatting for code, paths, and parameter values
   - Convert casual language to clear technical prose, but preserve the user's meaning exactly

4. **Use the Edit tool** to insert the entry at the right location. Don't rewrite surrounding content.

5. **Confirm** with a one-line summary of what was recorded and under which project.

## Example

User says: `/ln the out of sample 9wk video looks just as good as the training ones. no visible jitter at all. i think we can skip viterbi entirely`

You should ask: "Which project, and which session was this — the `--inference-only` run?" (if not already clear from conversation context). Then write, under today's date and the relevant project heading:

```markdown
Out-of-sample inference on `1901035_9wk_1` (different mouse, no timepoints in training data) via `18_dinov3_clean_multisession.py --inference-only` confirms the classifier generalizes: no visible jitter, head consistently oriented right. Output: `videos/.../1901035_9wk_1/cropped_height_3x_egocentric_dinov3_clean.mp4`. Viterbi/median-filter post-processing appears unnecessary — the raw per-frame classifier output is clean enough for production use.
```

## Important

- Do NOT run analysis scripts or pipelines. This skill only verifies the date, reads the notebook, and edits it.
- Do NOT add speculative content the user didn't say — but DO include interpretations and reasoning the user provides or that are clearly implied.
- Do NOT touch a `### Next steps` block unless the user explicitly updates it.
- If you lack required context (what script ran, what files were produced, what the reasoning was, or which project), **ask the user** before writing. Don't guess.
- Record the full substance — results, reasoning, and where the time went, including dead ends, things you had to track down, and debugging. Let length track the actual work; don't trim out the messy parts to keep it short. Avoid filler, not detail.
- You may use conversation context (recent tool outputs, file paths, job IDs) to fill in details the user omits, as long as they're factual.
