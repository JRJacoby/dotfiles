---
name: scientific-writing
description: Use when writing or editing a methods section, a methods paragraph, or any prose that describes a data-analysis, computational, or experimental method — including deciding how much of a method's internals to explain versus cite, and how much technical detail to report.
---

# Scientific Writing

Guidance for scientific prose. Each section covers one writing task.

## Describing a method in a methods section

A method description should read as a **chain of reasoning that builds understanding** — goal → problem → solution → why it works — not as an enumeration of facts. Calibrate how much you explain to how novel the method is, and keep it reproducible throughout.

1. **Function before mechanism.** Say what a step *does* before *how* it is implemented; the reader should know a tool's purpose before its internals. ("a transformation that preserves pose while standardizing body size" comes before "implemented as an autoencoder trained as a denoiser.")

2. **Explain why the mechanism achieves the function.** State the crux — the insight linking how the method works to what it accomplishes — rather than leaving the reader to infer it. ("denoising accomplishes normalization *because* every training target shares one reference morphology.")

3. **Walk the problem down to the concrete level the system operates on.** Don't stop at an abstract label like "confounded"; trace the causal chain to what actually happens to the data. (identical pose → different silhouette → a given pixel lands on a different body part.)

4. **Chain sentences with explicit connectives.** Each sentence should be motivated by the one before it (but / so / therefore / because / thus). The prose should advance as an argument, not sit as parallel assertions.

5. **For established methods, give WHY and WHAT, not HOW.** For a named, citable, standard technique, explain why you used it and what its inputs and outputs are — but not its internals. The citation carries the mechanism, and re-explaining it reads as less fluent, not more. Reserve the full walk-through (rules 1–4) for genuinely novel methods, or for the choices you layered on top of a standard one, that a reader can't get from a citation. **If unsure whether a method counts as established, ask.**

6. **Keep reproducibility detail regardless.** Report the concrete values needed to reproduce the work: dimensions, thresholds, hyperparameters, sample sizes, durations, software and versions. This is orthogonal to rule 5 — rule 5 trims *explanation* of a canonical method, never the *facts* needed to reproduce it. Fold the numbers into the narrative at the step that motivates them, rather than dumping them as a separate spec.

### Quick reference

| Rule | In one line |
|---|---|
| 1 Function before mechanism | What it does before how it works |
| 2 Why it works | State the crux insight, don't imply it |
| 3 Walk to the concrete | Trace the causal chain to the data |
| 4 Connective cohesion | Each sentence motivated by the last |
| 5 Established → WHY/WHAT | Cite it; don't re-explain internals |
| 6 Reproducible detail | Report the numbers, folded into the narrative |

### Common mistakes

- **Enumerating facts instead of building an argument** → apply rules 1–4.
- **Re-explaining a standard method's internals** → rule 5; cite it and move on.
- **Cutting the reproducibility numbers along with the explanation** → rule 6; trim explanation, keep the facts.
- **Stating the problem abstractly and stopping** → rule 3; walk to the concrete failure.
