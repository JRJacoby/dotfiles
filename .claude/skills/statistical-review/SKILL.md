---
name: statistical-review
description: Review the statistical methods in a plan or spec for rigor, completeness, and correctness. Use when a design includes hypothesis testing, bootstrapping, permutation tests, or group comparisons — before implementation begins.
---

# Statistical Review

Review statistical methods in a plan, spec, or script for rigor, completeness,
and common pitfalls. This skill is used **before implementation** to catch
problems that are expensive to fix later.

## When to Use

- A design spec includes hypothesis testing, CIs, or group comparisons
- A script performs bootstrap, permutation tests, or multiple comparisons
- Any analysis makes claims about statistical significance

## Review Process

Work through each section below in order. For each issue found:

1. **State the issue** clearly
2. **Explain the assumption** that is being violated or left implicit
3. **Describe the gotcha** — how it could lead to a wrong conclusion
4. **Suggest a fix** or flag it as a decision for the user

Categorize every finding as:
- **Critical**: will produce wrong results if not fixed
- **Important**: could mislead interpretation
- **Minor**: incomplete specification, unlikely to cause harm but should be documented

## Review Checklist

### 1. Unit of Analysis / Pseudoreplication

The most common statistical error in biology. Ask:

- **What is the experimental unit?** (animal? session? trial? frame?)
- **Does the statistical test treat each observation as independent?**
- **Are there nested/hierarchical relationships?** (frames within sessions,
  sessions within animals, animals within cages, cages within batches)
- **Is the resampling/permutation unit the same as the experimental unit?**

**Gotcha:** If two sessions come from the same animal, treating them as
independent inflates your effective n and produces false positives. Similarly,
analyzing per-frame statistics when the experimental unit is the session is
pseudoreplication — even with millions of frames, you may only have n=10
independent observations.

**Gotcha:** In paired designs (e.g., before/after treatment on the same animal),
using unpaired tests discards the pairing information and loses power.

### 2. Assumption Auditing

For every statistical test or model in the plan, explicitly identify:

- **What distributional assumptions does it make?** (normality, homoscedasticity,
  exchangeability, etc.)
- **Are those assumptions met, approximately met, or violated?**
- **What happens when assumptions are violated?** (conservative? anti-conservative?
  undefined?)

Common assumptions people forget:

| Method | Often-forgotten assumption |
|--------|---------------------------|
| Permutation test | Exchangeability under H0 — observations must be interchangeable under the null. Violated if groups have different variances even under H0. |
| Bootstrap CI | The sample is representative of the population. With n=5-10, the bootstrap distribution can be lumpy/multimodal. |
| BCa bootstrap | Requires the statistic to be smoothly transformable to normality. Can fail for discrete or heavily bounded statistics. |
| Mann-Whitney U | Assumes the two distributions have the same shape (tests for location shift only). If shapes differ, it tests something harder to interpret. |
| FDR correction | Assumes tests are independent or positively dependent (PRDS). Negatively correlated tests can violate the FDR guarantee. |
| Shannon entropy | Undefined when any probability is exactly zero. Requires a decision about how to handle zero-probability events. |

**Gotcha:** Non-parametric does not mean assumption-free. Every test has
assumptions. "Non-parametric" just means no distributional family is assumed.

### 3. Multiple Comparisons

- **Are ALL comparisons accounted for?** Count every test that will be performed.
  Include pairwise follow-ups, per-state tests, and tests across different
  analysis steps.
- **What is the family of tests for correction?** Justify why these tests are
  grouped together (or not) for FDR/FWER.
- **Is the correction method appropriate?**
  - FDR (Benjamini-Hochberg): controls false discovery rate, appropriate for
    exploratory analyses with many tests
  - FWER (Bonferroni, Holm): controls family-wise error rate, appropriate when
    any single false positive is costly
- **Are there hidden comparisons?** Choosing which states/conditions to analyze
  after seeing the data is an implicit comparison. Deciding to drop a state
  "because it's too rare" after fitting the model is data-dependent selection.

**Gotcha:** Per-step FDR correction does not control the project-wide false
discovery rate. If you run 5 independent analysis steps, each corrected at
alpha=0.05, your overall false discovery rate is higher than 5%. This is
acceptable for exploratory work but should be acknowledged.

**Gotcha:** With very few tests (e.g., 3 pairwise comparisons), FDR correction
provides little benefit over uncorrected p-values. The correction matters more
as the number of tests grows.

### 4. Effect Size and Practical Significance

- **Is effect size reported?** A p-value alone is insufficient. Report the
  magnitude of the difference (Cohen's d, percent change, odds ratio, etc.).
- **Could a statistically significant result be biologically trivial?** With
  enough sessions or frames, even tiny differences reach significance. Ask:
  "if this difference is real, does it matter?"
- **Could a non-significant result reflect low power rather than no effect?**
  With n=10/group, moderate effects may not reach significance. Absence of
  evidence is not evidence of absence.

**Gotcha:** A p-value of 0.001 does not mean the effect is large. It means
the effect is unlikely under the null. A tiny, meaningless difference can have
p < 0.001 with enough data.

**Gotcha:** Comparing p-values across tests ("this comparison was p=0.01 but
that one was p=0.04, so this effect is stronger") is invalid. P-values are not
effect size measures.

### 5. Power and Sample Size

- **Is n per group sufficient for the planned tests?** Rules of thumb:
  - Bootstrap/permutation: n >= 10 per group for reasonable CI coverage
  - Parametric tests: n >= 20-30 per group for CLT-based approximations
  - Detecting moderate effects (d=0.5): need ~60/group for 80% power with t-test
- **Are the planned number of bootstrap/permutation iterations sufficient?**
  - For CIs: 2,000-10,000 is typical
  - For p-values at alpha=0.05: at least 10,000 (so the smallest possible
    p-value is 0.0001, well below threshold)
  - For p-values at alpha=0.001: at least 100,000

**Gotcha:** Post-hoc power analysis (computing power after the experiment using
the observed effect size) is circular and uninformative. Power should be
estimated a priori or discussed in terms of the minimum detectable effect size.

### 6. Bootstrap and Permutation Specifics

For each bootstrap/permutation procedure:

- **What is being resampled?** (sessions? frames? residuals?)
- **Does the resampling unit match the experimental unit?** (see section 1)
- **What is the test statistic?** (difference of means? Frobenius norm? etc.)
  State it explicitly — "permutation test" is not a complete specification.
- **What is the null hypothesis?** (exchangeability of group labels?
  no difference in means? no difference in distributions?)
- **Is the test one-sided or two-sided?** Justify the choice.
- **What CI method is used?** (percentile, BCa, basic, studentized)
  - Percentile: simplest, but poor coverage for small n or skewed statistics
  - BCa: better coverage for small n, but requires the jackknife and can fail
    for discrete statistics
  - If BCa fails (e.g., all jackknife values identical), have a fallback

**Gotcha:** Permuting the wrong thing invalidates the test. If you permute
genotype labels across sessions but the sessions have different lengths, you
are testing a different null than intended (one that also permutes session-length
effects).

**Gotcha:** With n=10 per group, there are only C(20,10) = 184,756 possible
permutations for a two-group test. Exact enumeration may be feasible and
preferable to random sampling.

### 7. Circularity / Double-Dipping

- **Is the same data used for model selection and inference?** Selecting the
  "best model" by log probability and then testing that model's state assignments
  for genotype differences uses the same data twice. The selection step is not
  independent of the inference step.
- **Is any feature selection, threshold choice, or parameter tuning done on
  the same data that is later tested?**
- **Are "interesting" results selected for reporting based on the same data?**
  (e.g., "we found that state 3 differs between genotypes" — did you look at
  all states and only report the significant one?)

**Gotcha:** In shMoSeq specifically, the model is fit to all sessions pooled
across genotypes. The state assignments are then compared between genotypes.
This is generally fine (the model is not fit to maximize genotype differences),
but the state definitions themselves are influenced by the genotype composition
of the dataset. If one genotype dominates, states may be biased toward that
genotype's behavioral patterns.

### 8. Edge Cases and Degeneracies

Check what happens when:

- A state is never visited in a session (frequency = 0, duration undefined)
- A transition never occurs (zero row in transition matrix)
- All values in a group are identical (zero variance, many tests break)
- A session is much shorter or longer than others
- A bootstrap resample produces degenerate data (e.g., all same value)
- Shannon entropy is computed with zero-probability events (log(0) = -inf)

For each edge case, the plan should specify behavior: skip? impute? error?

**Gotcha:** Rare states are common in shMoSeq. A state used 0.1% of the time
may be visited 0 times in short sessions. If you compute per-session duration
for that state, you get NaN. If NaNs silently propagate into means and CIs,
results are wrong without any error being raised.

### 9. Interpretation Guardrails

Flag any place where the plan's framing could lead to over-interpretation:

- **"Significant difference" framing**: Does the plan distinguish between
  statistical significance and biological importance?
- **Exploratory vs. confirmatory**: Is this a hypothesis-generating analysis
  (exploratory) or testing a pre-registered hypothesis (confirmatory)? The
  language should match. Exploratory analyses should not use definitive
  language ("we demonstrate that...").
- **Absence of significance**: Does the plan acknowledge that non-significant
  results may reflect low power, not absence of effect?
- **Causal language**: Does the plan avoid causal claims from observational data?
  ("Grin2a mutation causes reduced state entropy" vs. "Grin2a HOM mice show
  reduced state entropy")

### 10. Completeness of Specification

Every statistical procedure must be fully specified before implementation.
Check that the plan includes:

- [ ] Test name and reference
- [ ] Test statistic (formula or clear description)
- [ ] Null hypothesis
- [ ] Alternative hypothesis (one-sided or two-sided)
- [ ] Significance level (alpha)
- [ ] Multiple comparison correction method and scope
- [ ] Sample size per group
- [ ] For bootstrap: resampling unit, n iterations, CI method
- [ ] For permutation: permutation unit, n permutations, test statistic
- [ ] How results are reported (plot format, sidecar file contents)
- [ ] Edge case handling (missing data, zero variance, etc.)

If any of these are "we'll figure it out during implementation," flag it.
Statistical methods should not be designed on the fly.

### 11. Domain-Specific Gotchas (Behavioral Neuroscience)

Common pitfalls in the specific domain of behavioral analysis:

- **Session-level confounds**: time of day, experimenter, cage, cohort, and
  batch effects can masquerade as genotype effects. Is there any way to check
  or control for these?
- **Behavioral autocorrelation**: consecutive frames/syllables are not independent.
  Any analysis treating frames as independent observations is pseudoreplication.
  Even session-level statistics can be autocorrelated if sessions are recorded
  on the same day.
- **Censored bouts**: state bouts at the start and end of a recording are
  truncated — their true duration is unknown. Including them in duration
  analysis biases estimates downward.
- **Simpson's paradox**: a trend that appears in each session can reverse when
  sessions are aggregated, and vice versa. This is especially relevant when
  sessions have different lengths or different baseline state frequencies.
- **Recording length effects**: if some sessions are longer, they contribute
  more state transitions and may dominate transition matrix estimates. Ensure
  that per-session normalization handles this.
- **State label alignment**: shMoSeq state labels are arbitrary across model
  fits. Comparing states between two separately-fit models requires alignment
  (e.g., Hungarian algorithm). Within a single model fit, labels are consistent.

## Output Format

Present findings as a structured report:

```
## Statistical Review: [Plan/Spec Name]

### Critical Issues
1. [Issue]: [Explanation] → [Fix]

### Important Issues
1. [Issue]: [Explanation] → [Fix or decision needed]

### Minor Issues
1. [Issue]: [Explanation] → [Suggestion]

### Assumptions Register
| Method | Assumption | Status | Risk if violated |
|--------|-----------|--------|------------------|
| ... | ... | Met/Violated/Unknown | ... |

### Completeness Checklist
[Filled-in checklist from section 10]
```
