---
name: dsa-review
description: Review a plan's data structures, algorithms, data access patterns, and user flows for completeness and coherence. Use when a design or implementation plan involves non-trivial data handling and needs DSA scrutiny before implementation begins.
---

# DSA Review

Review the data structures, algorithms, data access patterns, and user flows in
a plan or spec. The goal is to ensure these components are explicitly specified,
that their tradeoffs are acknowledged, and that the architecture connects
coherently from how data is stored and accessed through to how users interact
with the system.

This is a companion to plan-review (general plan quality) and statistical-review
(statistical rigor). Use all three together for thorough pre-implementation review.

## Input

The user will point you to a plan document. Read it thoroughly, then read any
code, data, or config files it references.

## Review Process

Work through each section below in order. For each issue found:

1. **State the issue** clearly
2. **Explain what is missing or under-specified**
3. **Describe the consequence** — what goes wrong if this isn't addressed
4. **Suggest a fix** or flag it as a decision for the user

Categorize every finding as:
- **Critical**: will cause implementation failure, data loss, or incorrect results
- **Important**: will cause performance problems, maintenance burden, or user confusion
- **Minor**: under-specified but unlikely to cause harm; should be documented

## Review Checklist

### 1. Data Access Patterns

**If the plan does not discuss data access patterns, that is a Critical issue.**

A plan that specifies data structures without specifying how they are read and
written is incomplete. For every data structure the plan introduces or uses, ask:

- **What are the read patterns?** Sequential scan? Random access by key? Range
  query? Iteration in sorted order? Full materialization into memory?
- **What are the write patterns?** Append-only? Random insert? Bulk load then
  read-only? Frequent updates? Write-once?
- **What is the expected data volume?** Number of records, size per record,
  total footprint in memory and on disk.
- **What is the access frequency?** Once at startup? Per frame? Per session?
  Per user interaction? In a tight loop?
- **Is there a mismatch between the chosen structure and the dominant access
  pattern?** (e.g., linear scan over a structure that should be indexed, or
  a hash map when ordered iteration is needed)

**Gotcha:** Plans often specify what data is stored but not how it is accessed.
"We store per-session statistics in a DataFrame" tells you the container but
not whether the dominant operation is row-wise iteration, column-wise aggregation,
random lookup by session ID, or merge/join with another table. The access pattern
determines whether the structure is appropriate.

**Gotcha:** "Load everything into memory" is an implicit access pattern decision.
For large datasets, this can silently become the bottleneck. The plan should
state expected sizes and justify that they fit in memory, or specify a
chunked/streaming approach.

### 2. User Flows

**If the plan does not discuss user flows, that is a Critical issue.**

Walk through how a user (or calling code) actually interacts with the system
end-to-end:

- **What triggers execution?** A CLI command? A function call? A scheduled job?
  An interactive UI action?
- **What inputs does the user provide?** Are they validated? What happens with
  bad input?
- **What does the user see at each step?** Progress? Intermediate output?
  Silence until completion?
- **What are the outputs?** Files? Plots? Return values? Side effects?
- **How does the user iterate?** If results are wrong or need tuning, what is
  the feedback loop? Does the user re-run everything or just a subset?

**Gotcha:** Plans that describe the computational pipeline without describing
how a human uses it produce code that is technically correct but painful to
use — wrong defaults, no progress feedback, outputs in unexpected locations,
no way to resume after failure.

### 3. Flow-to-Architecture Coherence

This is the core of the review: **do the data access patterns and user flows
connect to the data structures and algorithms, and vice versa?**

- **Forward check (user flow -> architecture):** For each step in the user flow,
  trace it through to the data structures and algorithms. Does the architecture
  support what the user needs to do? Are there user actions that require data
  access patterns the architecture doesn't support efficiently?
- **Backward check (architecture -> user flow):** For each data structure and
  algorithm, trace it back to a user need. Is there architecture that exists
  without a clear user-facing purpose? Is there complexity that doesn't serve
  a user flow?
- **Interface boundaries:** Where data crosses boundaries (disk to memory, one
  module to another, one pipeline stage to the next), are the formats and
  contracts explicit? Does the output of stage N match what stage N+1 expects?
- **State management:** Where does mutable state live? Who owns it? Can two
  parts of the system disagree about the current state?

**Gotcha:** A common failure mode is that the plan describes algorithms in
isolation ("we compute X, then Y, then Z") without showing how X's output
becomes Y's input. The data flow between steps is where most implementation
bugs live.

**Gotcha:** Plans that optimize for computational efficiency at the expense of
user flow (e.g., a single monolithic computation that can't be interrupted or
inspected) create systems that are fast but unusable in practice.

### 4. Data Structure Selection and Tradeoffs

For every data structure the plan introduces:

- **Why this structure?** What alternatives were considered? The plan should
  state the reason for the choice, not just the choice.
- **What are the tradeoffs?** Every data structure trades off between insertion,
  lookup, iteration, memory, and complexity. Which tradeoffs does this choice
  make, and are they appropriate for the access patterns identified in section 1?
- **What are the invariants?** What must always be true about this structure?
  (sorted? unique keys? non-empty? consistent with another structure?)
- **Who maintains the invariants?** Is there a single point of responsibility,
  or can multiple code paths modify the structure?

**Gotcha:** "Use a dict/DataFrame/list" without justification is a design
smell. These are fine defaults, but the plan should show awareness that a
choice was made. A dict with 10M entries has different characteristics than
one with 100.

**Gotcha:** Parallel data structures (two lists that must stay in sync by index,
a dict and a list that represent the same entities) are fragile. If the plan
introduces them, it should justify why a single structure won't work.

### 5. Algorithm Selection and Tradeoffs

For every non-trivial algorithm or computation the plan describes:

- **What is the expected time complexity?** Not necessarily big-O — a concrete
  estimate for expected data sizes is more useful. "This is O(n^2) but n is
  always < 100" is fine. "This is O(n^2)" without knowing n is not.
- **What is the expected space complexity?** Same — concrete estimates matter
  more than asymptotic class.
- **Are there edge cases where performance degrades?** (e.g., hash collisions,
  worst-case sort, degenerate input)
- **Is the algorithm deterministic?** If not, does the plan specify a seed or
  acknowledge non-reproducibility?
- **Are there simpler alternatives?** If the plan uses a complex algorithm,
  does it justify why a simpler approach won't work?

**Gotcha:** Plans sometimes specify a sophisticated algorithm when a naive
approach would work fine at the actual data scale. Unnecessary complexity is
a maintenance cost. Conversely, plans sometimes specify a naive approach
without checking whether it will be fast enough at expected scale.

### 6. Serialization and Persistence

If data is saved to or loaded from disk:

- **What format is used?** (HDF5, parquet, CSV, pickle, JSON, custom binary)
- **Why that format?** Does it match the access pattern? (e.g., parquet for
  columnar access, HDF5 for large arrays, CSV for human readability)
- **What is the schema?** Column names, data types, array shapes, group
  hierarchy. Is it explicitly specified or implicit?
- **Is there versioning?** If the schema changes, how are old files handled?
- **Is there a risk of partial writes?** If the process crashes mid-write,
  is the output corrupt? Should writes be atomic?
- **File sizes:** Are expected file sizes stated? Will output fit on the
  target filesystem?

**Gotcha:** Pickle is convenient but not portable, not human-readable, and
a security risk for untrusted data. If the plan uses pickle, it should
acknowledge this.

**Gotcha:** Plans that write intermediate results to disk without specifying
cleanup create storage bloat over time. Who deletes the intermediates?

### 7. Error Handling and Recovery

For each stage of the data pipeline:

- **What errors can occur?** (missing files, malformed data, out of memory,
  division by zero, empty input)
- **What happens when they occur?** (skip and log? abort? retry? fallback?)
- **Can the user resume from a failure?** If step 3 of 5 fails, does the user
  re-run from step 1 or from step 3? Does the plan support checkpointing
  or idempotent re-execution?
- **Are error messages actionable?** Will the user know what to fix?

**Gotcha:** Plans that say "handle errors appropriately" without specifying
behavior are under-specified. "Handle errors" is not a design — it's a wish.

### 8. Scalability Boundaries

Every design has a scale at which it breaks. The plan should be honest about
where those boundaries are:

- **At what data size does the in-memory approach fail?** State the expected
  working set and available memory.
- **At what data size does the algorithm become too slow?** State expected
  runtimes for typical and worst-case inputs.
- **What happens when the boundary is hit?** Graceful degradation? OOM crash?
  Silent incorrect results?
- **Does the plan need to scale beyond current data?** If yes, what is the
  growth trajectory?

If the plan says nothing about scale, flag it. Every plan should state the
expected operating range and what happens at the edges.

## Output Format

Present findings as a structured report:

```
## DSA Review: [Plan/Spec Name]

### Critical Issues
1. [Issue]: [Explanation] -> [Fix]

### Important Issues
1. [Issue]: [Explanation] -> [Fix or decision needed]

### Minor Issues
1. [Issue]: [Explanation] -> [Suggestion]

### Data Access Pattern Summary
| Structure | Read Pattern | Write Pattern | Volume | Justified? |
|-----------|-------------|---------------|--------|------------|
| ...       | ...         | ...           | ...    | ...        |

### User Flow Trace
[End-to-end walkthrough showing where each step connects to architecture]

### Tradeoff Register
| Decision | Alternatives considered | Tradeoff made | Appropriate? |
|----------|----------------------|---------------|--------------|
| ...      | ...                  | ...           | ...          |
```

After the report, give an overall assessment: are the DSA components of this
plan ready for implementation, or does it need revision?
