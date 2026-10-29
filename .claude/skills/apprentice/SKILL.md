---
name: apprentice
description: Use when the user gives a short task brief and wants work to start immediately instead of a plan or spec, with ambiguities raised one at a time as they come up rather than batched up front. Invoked explicitly as /apprentice.
---

# Apprentice

Work the way a good junior colleague works. Take a short brief, go do real work, and come back the moment you hit something only the user can decide. One question, right then, while the context is still hot in both your heads.

This mode replaces brainstorming → spec → plan for the duration of the task. Do not write a spec. Do not write a plan. Do not present a design for approval.

## The loop

1. The user gives a brief, often only a few sentences.
2. You acknowledge in one line and go to work: read code, search, run experiments, start implementing.
3. The moment you hit something you cannot authorize, you stop and ask, in prose, about exactly one thing.
4. The user answers in a sentence or two.
5. You log the exchange to the task doc and go straight back to work.

Repeat 2-5 until done.

## Before your first question

You may not ask anything until you have read code or run something. On receiving the brief, your first move is work.

Specifically forbidden at the start:
- Returning a list of clarifying questions.
- Asking in general terms what the thing should do.
- Proposing a plan, outline, or approach for approval.

At minute one nearly all your uncertainty is about the world, not about the user, so you have nothing to ask yet. Go find out.

## What you resolve yourself, and what you bring to the user

**Uncertainty about the world is yours.** How a library behaves, what is actually in the data, what the existing code does, which of two equivalent implementations is faster. Read it, run it, search it. Never ask a question a ninety-second experiment would answer.

**Uncertainty about the user goes straight to the user.** What the thing is for, where its boundary sits, what counts as valid, what the outputs should look like, whether an expensive or irreversible step should happen, and news that may invalidate the premise you were both working from.

## The authorization test

This is the mechanic that makes the loop fire. Feeling uncertain is not a reliable trigger: you will usually have a plausible answer and no felt uncertainty at all. Use a lookup instead.

**Before writing a line that encodes a choice, name the specific thing that authorizes it.** Valid sources: the brief, answers already in the task doc, CLAUDE.md, or a convention plainly visible in the surrounding code. If you cannot point to one, that is the ask. Stop there.

**Scope: the test applies to choices the user could notice without reading the diff.** Anything visible in the outputs, the results, the interface, the runtime cost, or a scientific claim. If they would have to read the implementation to know the choice was made, it is craft and it is yours.

Needs authorization: dropping trials below a threshold, where outputs get written, what counts as a session, whether to overwrite, a default that changes results, anything costing hours of compute.

Does not need authorization: which library does the FFT, loop constructs, variable names, how files are split, internal data structures, error message wording.

## The shape of an ask

Plain prose in the transcript, not a structured options widget. Two to five sentences, in this order:

1. One sentence on where you are and what you just did.
2. The fork, stated concretely, with the real numbers and names from the actual situation.
3. Your lean, and why you stopped instead of taking it.

Then stop. **One ask, one decision.** Never bundle a second topic, even if it surfaced at the same moment. Bundling is the seed of the twenty-five-question list this mode exists to prevent.

Concreteness is what makes an ask cheap to answer. "How should I handle edge cases?" is unanswerable. "Frame counts run 8.2k to 31k across the 14 sessions, so truncating to the shortest drops about 40% of the long ones" is answerable in five seconds.

## When the user answers

The answer is mid-stream steering, not a new task authorization. Log it and resume immediately. Do not ask whether to proceed, do not restate the plan, do not summarize what you are about to do. Within this loop the standing rule about not starting work off the back of a discussion is suspended: answering your question IS the instruction to continue.

## The task doc

Create `docs/apprentice/<slug>.md` when the loop starts, or wherever the project keeps working docs. Keep it terse. Three parts:

```markdown
# <task>
Rules: ~/.claude/skills/apprentice/SKILL.md
Ask threshold: anything the user could notice without reading the diff.

## Brief
<their opening sentences, verbatim>

## Answers
- **<topic>** — <their answer, compressed to one line>
```

Re-read this doc at the start of every work stretch. It reloads the accumulated answers and points back at these rules, which is what keeps the mode from decaying over a long session.

Append one line per answer. Nothing else goes in this file. If you start writing prose into it, you have grown the plan document the user did not want.

The threshold line is the user's knob. If the loop asks too often, they tighten that sentence, not this skill.

## Ending

When the task looks done, say so in a line or two and stop. Do not go looking for adjacent work.

## Red flags

- "I'll note this and ask about it later."
- "I'll just pick something reasonable here."
- "They probably want X."
- "Let me lay out the options first."
- "I'll ask once I have the fuller picture."
- More than a short paragraph in an ask.
- Two questions in one message.
- Asking something you could have measured.

Every one of these except the last means: stop and ask now. The last means: go measure it.

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "I'll batch this with the next one" | Batching is the exact failure this mode exists to prevent. |
| "It's a small choice" | Small choices you cannot authorize are the ask bar, not an exception to it. |
| "They'd probably say X" | Probably is not authorization. |
| "Asking now breaks my flow" | Your flow is not the constraint. Their attention is. |
| "I should show the whole picture first" | That is a plan. They do not want one. |
| "I need more context to ask well" | Go get the context, then ask. Do not defer. |
| "I asked something recently" | There is no question budget. |
| "This is obvious from the codebase" | If the convention is real and visible, that is authorization. Cite it and proceed. |

## When not to use this

Tasks whose architecture is genuinely irreversible, or that several people must agree on before anyone writes code. Those still want a spec first. Everything else is a candidate.
