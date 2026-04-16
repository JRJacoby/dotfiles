---
name: ln
description: Record results, decisions, or observations in the project's lab notebook (lab_notebook.md). Use when the user says /ln followed by a message describing what happened.
---

# Lab Notebook Entry

The user is dictating a lab notebook entry. Your job: turn their message into a clean, concise, technical notebook entry and append it to `lab_notebook.md` in the project root.

## Input

The user's message after `/lm` is the raw content — it may be conversational, abbreviated, or stream-of-consciousness. Parse it for:
- **Results**: quantitative outcomes, visual assessments, pass/fail
- **Decisions**: "we're going with X", "decided to Y", "dropping Z"
- **Observations**: things noticed, hypotheses, caveats
- **Status updates**: "finished X", "started Y", "blocked on Z"

## Required context per entry type

Each entry type requires specific supporting context. If the user's message doesn't include it, **ask before writing**:

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

## How to write the entry

1. **Read the end of `lab_notebook.md`** (last ~30 lines) to find:
   - The current date heading (e.g., `## 2026-04-15`). If today's date already has a heading, add under it. If not, create a new `## YYYY-MM-DD` section.
   - The `### Next steps` block, if present — insert the new entry BEFORE it.

2. **Write the entry** as a short paragraph or bulleted list under a `####` subheading if the content warrants one, or as a plain paragraph if it's brief. Match the existing notebook style:
   - Technical but not verbose
   - Include specific numbers, file paths, session IDs, script names, SLURM job IDs
   - Include what was run: exact script path and key arguments
   - Include what was produced: output file paths
   - Use backtick formatting for code, paths, and parameter values
   - Convert casual language to concise technical prose, but preserve the user's meaning exactly

3. **Use the Edit tool** to insert the entry at the right location. Don't rewrite surrounding content.

4. **Confirm** with a one-line summary of what was recorded.

## Example

User says: `/lm the out of sample 9wk video looks just as good as the training ones. no visible jitter at all. i think we can skip viterbi entirely`

You should ask: "Which session was this? And was this from the `--inference-only` run?" (if not already clear from conversation context). Then write:

```markdown
Out-of-sample inference on `1901035_9wk_1` (different mouse, no timepoints in training data) via `18_dinov3_clean_multisession.py --inference-only` confirms the classifier generalizes: no visible jitter, head consistently oriented right. Output: `videos/.../1901035_9wk_1/cropped_height_3x_egocentric_dinov3_clean.mp4`. Viterbi/median-filter post-processing appears unnecessary — the raw per-frame classifier output is clean enough for production use.
```

## Important

- Do NOT run any commands or scripts. This skill only edits the notebook.
- Do NOT add speculative content the user didn't say — but DO include interpretations and reasoning the user provides or that are clearly implied.
- Do NOT touch the "### Next steps" block unless the user explicitly updates it.
- If you lack required context (what script ran, what files were produced, what the reasoning was), **ask the user** before writing. Don't guess.
- Keep entries short — a lab notebook records facts and reasoning, not essays.
- You may use conversation context (recent tool outputs, file paths, job IDs) to fill in details the user omits, as long as they're factual.
