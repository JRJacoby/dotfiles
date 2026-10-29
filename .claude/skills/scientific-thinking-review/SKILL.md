---
name: scientific-thinking-review
description: Review the scientific reasoning in a plan, spec, or analysis — whether it poses a clear question and whether the method actually answers it. Use before implementation, alongside plan-review and statistical-review, when a design makes a scientific claim.
---

# Scientific Thinking Review

Review whether a plan, spec, or analysis is *good science* — independent of
whether it is well-engineered or statistically correct. The question here is not
"is the code right?" or "is the test valid?" but "does this analysis pose a clear
scientific question, and would its results actually answer that question and mean
what we would claim?"

A plan can pass plan-review and statistical-review and still be bad science: it
can measure the wrong thing, fail to distinguish the hypotheses it cares about,
leave an obvious alternative explanation open, or produce a number that no
outcome could ever disconfirm. This skill catches that.

## Relationship to sibling review skills

Stay in your lane. Do not re-litigate what the siblings own.

- **plan-review** — internal consistency, codebase accuracy, implementability.
- **statistical-review** — units of analysis, assumptions, multiple comparisons,
  power, resampling mechanics, effect size. *Do not re-derive statistical
  findings here.*
- **scientific-thinking-review** (this skill) — the logic connecting **question →
  method → measurement → interpretation**. Is the question stated? Does the
  method answer *that* question? Do the measured quantities capture the concept?
  Could the result mean something other than what we would claim?

When a finding is genuinely both (e.g., a confound that is also a statistical
unit-of-analysis problem), note it briefly and defer the mechanics to
statistical-review.

## Review process

Work through each dimension below. For every issue found:

1. **State the issue** — what specifically is unclear, misaligned, or unjustified.
2. **Name the consequence** — what wrong or unsupported conclusion it could lead
   to, or what question is left unanswered.
3. **Suggest a fix or flag a decision** — a concrete change, or an explicit
   question the author must answer.

Categorize every finding:

- **Blocking** — as designed, the analysis cannot answer its question, or its
  result would not support the intended claim. Must be resolved before building.
- **Important** — the analysis could mislead, or a plausible alternative
  explanation is left uncontrolled, or the claim would be weaker than implied.
- **Minor** — the reasoning is sound but a justification, scope limit, or
  robustness check should be made explicit.

If a dimension is clean, say "no issues found" and move on. Do not pad with praise.

## Review dimensions

### 1. Is the scientific question explicit and judgeable?

The spec must state, in words, the question it is trying to answer — not just the
procedure it will run. A pipeline that lists steps but never states the question
cannot be evaluated for whether the steps answer it.

Check that the spec makes explicit:

- **The question or hypothesis**, phrased as a claim that could be true or false
  ("KO mice spend more time in high-arousal states than WT"), not a task
  ("compare state frequencies across genotype").
- **The predicted result** if the hypothesis holds — direction, and where
  possible rough shape or magnitude.
- **The motivating "why"** — what larger question this bears on.

**Gotcha:** "Explore whether there are differences" is not a question; it is a
fishing license. Exploratory work is legitimate, but it must be *labeled*
exploratory, and its outputs framed as hypothesis-generating, not confirmatory.
If the spec's language ("we will demonstrate that...") outruns its design, flag it.

**Gotcha:** A question stated only in the chat/PI's head but not written in the
spec fails this check. The spec must be judgeable on its own.

### 2. Does the method answer *that* question? (method–question alignment)

Given the question from dimension 1, ask whether the planned analysis actually
bears on it. Walk the inference forward: if every step runs and produces the
hoped-for output, does that output license the claim?

- **Is there a gap between what is computed and what is asked?** ("Do KO mice
  transition faster between behaviors?" answered only by *mean state duration*
  misses transition structure; the metric may not track the construct.)
- **Does the comparison isolate the effect of interest?** Is the contrast the
  right one, against the right baseline/control condition, to attribute the
  difference to the intended cause rather than to something the groups also differ in?
- **Would a positive result be interpretable?** If the analysis "works," is the
  resulting number one whose meaning is clear, or an aggregate that could arise
  many ways?

**Gotcha:** Answering an easier, adjacent question than the one asked is the most
common failure. Name the substituted question explicitly.

### 3. Construct validity (does the measure capture the concept?)

Every named quantity ("behavioral flexibility," "arousal," "vigor," "engagement")
is a *construct* operationalized by a *measure*. The review must check the bridge
between them.

- **Is the operationalization stated?** ("We measure flexibility as state-sequence
  entropy.") If a construct appears with no defining computation, that is a gap.
- **Is the bridge defensible?** Does higher entropy really mean more flexibility,
  or could it mean more disorganization, more noise in state assignment, or
  shorter recordings? State the competing readings of the same number.
- **Does the measure have known failure modes** that decouple it from the
  construct (e.g., a ratio that blows up when the denominator is small; a metric
  confounded by recording length or by overall activity level)?

**Gotcha:** A measure that correlates with the construct in the easy case can
diverge from it exactly in the regime being studied. The label is not the thing.

### 4. Discriminating power (does the analysis distinguish the hypotheses?)

An informative analysis produces *different* expected results under the competing
hypotheses. If every hypothesis on the table predicts the same output, the
analysis cannot adjudicate between them, no matter how clean the result.

- **What are the competing hypotheses** (including the boring null and the
  "trivial confound" explanation)?
- **What does each predict** for the planned output?
- **If two hypotheses predict the same thing**, the analysis is uninformative for
  that contrast — flag it and ask what additional measurement would separate them.
- **Is this a severe test?** Does the hypothesis make a *risky* prediction — one
  that would probably fail if the hypothesis were false — or a safe one that
  almost any state of the world would satisfy?

**Gotcha:** "We found a significant difference" is not informative if the
difference was predicted equally by the hypothesis and by a mundane confound.

### 5. Alternative explanations (inferential, not statistical)

If the result comes out as hoped, what *else* could produce it? This is about the
logic of attribution, not the statistics of noise.

- **List the plausible alternative causes** of the predicted result.
- **For each, does the design rule it out** — by control condition, by covariate,
  by the structure of the comparison? If not, the claim is under-determined.
- **Batch/context/apparatus artifacts as scientific (not just statistical)
  confounds**: could the "effect" be a property of how the data were collected
  rather than of the biology?

**Gotcha:** The most dangerous alternative explanation is the one that predicts
the *same direction and magnitude* as the hypothesis. Those are the ones a
significant result cannot distinguish.

### 6. Controls and sanity checks

Good scientific designs include comparisons with a *known* expected answer, to
show the pipeline can detect what it claims to detect and does not manufacture
effects from nothing.

- **Positive control**: is there a manipulation or contrast where an effect is
  expected, so a null there would indict the pipeline (not the biology)?
- **Negative control**: is there a contrast where *no* effect is expected (e.g.,
  shuffled labels, a pre-stimulus baseline, a comparison known to be null), to
  bound the false-positive behavior of the whole procedure?
- **Recovery check**: can the analysis recover a synthetic or known effect
  injected into the data?

**Gotcha:** A pipeline that has never been shown to produce a null on null data
gives no assurance that its positive findings are real.

### 7. Falsifiability and pre-specified predictions

- **Is there an outcome that would disconfirm the hypothesis?** If any result can
  be narrated as support, the analysis is not testing anything.
- **Is the predicted direction/shape fixed in advance**, or will whatever emerges
  be interpreted post hoc as consistent with the story? Post-hoc directionality
  turns a test into a Rorschach.
- **Is the null meaningful and reachable** given the design, or is it a straw null
  that could never be true regardless of the biology?

**Gotcha:** Distinguish exploratory from confirmatory. Exploratory analyses need
not pre-specify — but then their outputs are hypotheses to test later, not
conclusions, and the writeup must say so.

### 8. Inferential chain from data to claim

Trace every link from raw measurement to the sentence the author wants to write.
Each hidden linking assumption is a place the argument can break.

- **Enumerate the inferential steps**: signal → derived quantity → group
  statistic → biological claim. Is each step licensed?
- **Flag silent leaps**, especially measurement-to-psychology ones ("more time in
  state X = more anxiety"; "faster transitions = more impulsive"). Is that mapping
  argued, cited, or just assumed?
- **Does the conclusion's strength match the evidence's strength?** Correlational
  design, causal sentence — flag the mismatch.

### 9. Scope of generalization

- **To what population, apparatus, and condition does the claim extend?** One
  cohort, one arena, one task, one age. Is the intended breadth of the conclusion
  matched by the breadth of the data?
- **Are there obvious limits** (single sex, single lab, single genetic background)
  the writeup should state rather than imply generality?

### 10. Robustness to analysis choices

A conclusion that depends on one specific pipeline setting is fragile science.

- **Which analyst-chosen parameters could flip the headline claim?** (number of
  states/clusters, kappa/smoothing, thresholds, inclusion cutoffs, binning).
- **Does the plan check that the conclusion survives reasonable alternatives**, or
  at least acknowledge the dependence? A "multiverse" spot-check beats a single
  point estimate presented as fact.
- **Is any inclusion/exclusion or threshold chosen after seeing its effect on the
  result?** (This is also circularity — note it and defer mechanics to
  statistical-review.)

### 11. Relevance to the motivating goal ("so what")

Even a clean, correctly-answered sub-question can be beside the point.

- **Does answering this question actually advance the stated aim** from dimension
  1's "why," or is it an adjacent-but-orthogonal result?
- **If the result is exactly as predicted, what does the author get to conclude** —
  and is that conclusion worth the work? If not, the scope may be wrong.

## Output format

```
## Scientific Thinking Review: [Plan/Spec Name]

### The question, as I understand it from the spec
[One or two sentences restating the scientific question the spec poses. If you
cannot state it, that is itself a Blocking finding under dimension 1.]

### Blocking
1. [Issue] → [consequence] → [fix or decision needed]

### Important
1. [Issue] → [consequence] → [fix or decision needed]

### Minor
1. [Issue] → [consequence] → [suggestion]

### Hypothesis-discrimination table
| Competing hypothesis | Predicted result under the planned analysis |
|----------------------|---------------------------------------------|
| (intended)           | ...                                         |
| (null)               | ...                                         |
| (confound/alt)       | ...                                         |
```

Fill the discrimination table whenever the spec makes a comparative claim — it is
the fastest way to expose an analysis that cannot distinguish its hypotheses.

After the findings, give an overall verdict: is the scientific reasoning sound
enough to build on, or does the question/method alignment need work first?
