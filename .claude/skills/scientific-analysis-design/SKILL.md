---
name: scientific-analysis-design
description: Design scientific analyses with full operationalization — every computed quantity defined in units and equations, every statistical choice justified, every assumption documented. Use instead of brainstorming when the task is a data analysis rather than a software feature.
---

# Scientific Analysis Design

Turn a natural-language analysis request into a fully operationalized analysis
spec where every computed quantity is defined in units and equations, every
statistical choice is justified, and every assumption is documented.

**This is NOT the brainstorming skill.** Brainstorming is for software features.
This skill is for scientific data analysis — where the primary risk is not bad
architecture but vague operationalization of scientific concepts.

## Hard Gate

Do NOT write implementation code, invoke writing-plans, or take any
implementation action until the spec is complete and the user has approved it.

## Core Principle: Nothing Left to English

Every quantity in the spec must be defined precisely enough that two independent
implementers would compute the same number. If a term can be interpreted two
ways, it is not yet specified.

**Bad:** "state frequency" — could mean proportion of frames or proportion of bouts.
**Good:** "state frequency (per-frame): for each session, the fraction of total
frames assigned to state k. F_k = n_frames_in_state_k / n_total_frames."

**Bad:** "normalize to baseline" — what is baseline? subtract? divide? z-score?
**Good:** "baseline-subtract: for each session, compute mean dF/F over the first
60 seconds (pre-stimulus period), then subtract this scalar from all timepoints.
Units: dF/F (dimensionless)."

**Bad:** "behavioral complexity"
**Good:** "usage entropy: Shannon entropy over the per-session state frequency
vector. H = -sum(p_k * log(p_k)) where p_k is the proportion of bout initiations
for state k out of all bout initiations in that session. Units: nats."

**Bad:** "remove outliers"
**Good:** "exclude frames where DLC confidence < 0.9 for any keypoint. Set
excluded keypoint coordinates to NaN; do not interpolate."

**Bad:** "a reasonable threshold"
**Good:** "2 cm radius from object center (object center defined as the median
x,y coordinate of the object keypoint across all frames where confidence > 0.99)."

## Process

### Step 1: Write the Bare-Bones Spec

Before doing anything else, write down exactly what the user said as a minimal
spec. Do not embellish, do not add steps they didn't ask for. Save to
`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`. This is the seed
document that will be iteratively refined.

### Step 2: Explore Project Context

Read relevant code, data files, existing analyses, and recent commits. Update
the spec with concrete details you can fill in from context (file paths, column
names, existing constants, data shapes, genotype labels, fps, etc.). Note any
ambiguities you discover.

### Step 3: Self-Answer

Before asking the user anything, try to resolve ambiguities yourself:
- Read existing code that does something similar
- Run quick diagnostic scripts (check column names, count sessions, inspect data shapes)
- Check how related quantities are computed elsewhere in the codebase
- Read docstrings and comments in relevant functions

Update the spec with anything you resolved. Only ask the user questions you
genuinely cannot answer from the codebase.

### Step 4: Iterative Question-and-Refine Loop

Ask questions **one at a time**. After each answer, immediately update the spec
to incorporate the answer. The spec should always reflect the current state of
understanding — not be written all at once at the end.

**Exit condition:** The loop ends when you have no remaining questions (per the
checklists below) OR the user says to move on.

### Step 5: Document Assumptions

Add an explicit "Assumptions" section to the spec listing every assumption the
analysis makes, even obvious ones. Examples:
- "Sessions are independent (no repeated measures on the same animal)"
- "State assignments are deterministic (single MAP estimate, not posterior samples)"
- "DLC confidence threshold of 0.9 is sufficient to exclude tracking failures"
- "The novel object does not move during the session"
- "Frame rate is constant at 25 fps throughout each recording"

### Step 6: User Reviews Spec

Ask the user to review the completed spec before proceeding.

### Step 7: Transition to Implementation

Invoke the `superpowers:writing-plans` skill to create an implementation plan.
Also invoke `pipeline-design` and `analysis-project-structure` skills. Existing
code that predates these conventions is grandfathered in, but all new scripts
must follow them.

## What Questions to Ask

The question loop draws from two checklists. The goal is to pre-empt every
comment that a plan-review or statistical-review would raise — so those reviews
find nothing.

### Checklist A: Operationalization and Logistics

These questions ensure every quantity is precisely defined and every practical
detail is pinned down. Drawn from the kinds of issues the `plan-review` skill
catches.

1. **Definitions** — Is every computed quantity defined in units and equations?
   Can two independent implementers produce the same number?
   - "You said 'state frequency' — do you mean the proportion of frames in each
     state (time allocation) or the proportion of bout initiations (event rate)?"
   - "You said 'speed' — is this displacement per frame, displacement per second,
     or something else? Pixel units or real-world units?"
   - "'Distance to center' — center of what? Arena center (known geometry),
     median animal position, or centroid of object?"
   - "You want to 'smooth the signal' — Gaussian? boxcar? Savitzky-Golay? What
     window size, in seconds or samples?"

2. **Inclusion/exclusion criteria** — What data goes in, what gets dropped, and why?
   - "Are all sessions included or are some excluded? What are the criteria?"
   - "How do you handle sessions where a state is never visited? NaN or zero?"
   - "What's the minimum number of bouts needed to compute a meaningful duration
     statistic for a state?"
   - "Are the first and last bouts per session censored (truncated by recording
     boundaries)?"

3. **Data provenance** — Where does each input come from?
   - "Which results.h5 file? Which parquet? Which column?"
   - "Are genotype labels in index.csv or derived from filenames?"
   - "What's the recording key format and how does it map to video filenames?"

4. **Output specification** — What exactly gets produced?
   - "What plots, tables, stats files?"
   - "What goes in the PDF report vs. what's just intermediate output?"
   - "Should there be a sidecar JSON with statistical details for each plot?"

5. **Edge cases** — What happens when things aren't clean?
   - "What if a session has fewer than N bouts of a state?"
   - "What if the DLC tracking is poor for an entire session?"
   - "What if a genotype group has only 2 animals?"

### Checklist B: Statistical Rigor

These questions ensure the analysis is statistically sound. Drawn from the kinds
of issues the `statistical-review` skill catches.

1. **Unit of independence** — What is the experimental unit?
   - "Is the unit of replication the animal, the session, or the trial? Are there
     multiple sessions per animal?"
   - "You're comparing across genotypes — how many independent animals per group?"
   - "Are left and right arena recordings from the same animal? If so, they're
     not independent."

2. **Multiple comparisons** — How is the false discovery rate controlled?
   - "You're computing 3 genotype pairs × 5 states = 15 tests. FDR correction
     within each metric, or across all metrics?"
   - "Are you treating this as exploratory (report effect sizes, flag interesting
     patterns) or confirmatory (control family-wise error rate)?"

3. **Test appropriateness** — Is the statistical test suitable for the data?
   - "Permutation test assumes exchangeability — is that valid here?"
   - "Bootstrap CIs assume the sample is representative of the population — with
     n=7 per group, is that reasonable?"
   - "Are you comparing means or medians? The choice matters if distributions
     are skewed."

4. **Effect size and power** — Can the study detect meaningful effects?
   - "With n=10 per group, what effect size can you reliably detect?"
   - "Are you reporting effect sizes (Cohen's d) alongside p-values?"
   - "Is the analysis underpowered for any comparisons?"

5. **Assumptions** — What does the analysis assume about the data?
   - "Does this assume normality? Homogeneity of variance?"
   - "Does the permutation test assume the null hypothesis is that groups are
     exchangeable, or something weaker?"
   - "Are you assuming stationarity (behavior doesn't change over the session)?"

### Checklist C: Scientific Intent

These questions ensure the analysis answers the scientific question the user
actually cares about, not just a technically correct but scientifically
meaningless version of it.

1. **What is the hypothesis?** — What biological question motivates this analysis?
   - "Are you testing whether genotype affects behavior, or characterizing how?"
   - "Is this exploratory (generating hypotheses) or testing a specific prediction?"
   - "What would a 'positive result' look like, and what would it mean biologically?"

2. **Does the metric capture the concept?** — Is there a gap between what you
   want to measure and what you're actually computing?
   - "You want to measure 'novel object investigation' — your metric is time in
     a behavioral state that includes object investigation but also other
     behaviors. Is that a good proxy?"
   - "You're measuring 'anxiety' via time in center. Is center-time actually a
     good anxiety metric for this arena layout?"
   - "You're comparing 'transition structure' via Frobenius norm of difference
     matrices. Does that capture the kind of transition difference you care about,
     or could two very different transition patterns have similar Frobenius norms?"

3. **Confounds** — What else could explain the result?
   - "Could locomotor differences drive the effect you're seeing in state
     frequency, independent of the behavior you care about?"
   - "Are there batch effects, cage effects, or sex differences that should be
     accounted for?"
   - "Does the recording order correlate with genotype?"

## Tone: Mentor, Not Sycophant

Do not simply accept every analysis choice the user makes. If something is
statistically questionable, methodologically unusual, or scientifically unclear,
say so directly. Push for justification.

**Acceptable user defenses:**
- "This is exploratory — I want to see where effects might be before investing
  in a rigorous confirmatory analysis."
- "We don't have enough power for X, so I'm using Y as a pragmatic alternative
  and will note the limitation."
- "The field standard is to do it this way, even though it's not ideal."

**Not acceptable:**
- "Just do it." (without any justification for a questionable choice)
- "It's fine." (in response to a legitimate statistical concern)

If the user insists on a questionable choice without justification, document it
in the spec's Assumptions section as a known limitation. Do not silently comply.

## Key Principles

- **One question per message** — don't overwhelm
- **YAGNI ruthlessly** — if the user didn't ask for it, don't add it
- **Follow existing codebase patterns** — existing code is grandfathered; new
  code follows pipeline-design and analysis-project-structure conventions
- **The spec is a living document** — update it after every question, not all
  at once at the end
- **Resolve before asking** — check the codebase, run a quick command, read the
  data. Only ask the user what you genuinely can't figure out yourself
