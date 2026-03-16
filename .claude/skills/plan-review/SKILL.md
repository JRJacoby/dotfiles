---
name: plan-review
description: Critique a design/implementation plan before handing off to an implementer. Use when a plan has been written and needs review before execution.
---

# Plan Review

Systematically critique a design or implementation plan. The goal is to catch problems before they reach implementation, where they're much more expensive to fix.

## Input

The user will point you to a plan document (usually in `docs/plans/`). Read it thoroughly, then read any code, data, or config files it references.

## Critique Checklist

Evaluate the plan against each of these criteria. For each one, either note specific issues or explicitly state "no issues found." Do not skip criteria.

1. **Internal contradictions** — Does the plan contradict itself? Does it say one thing in one section and something different elsewhere?

2. **Inconsistencies with the codebase** — Does the plan accurately describe the current state of the code and data? Are file paths, function names, variable names, column names, data structures correct? Does it reference things that don't exist or mischaracterize things that do?

3. **Glossed-over design decisions** — Are there decisions the plan treats as obvious or simple that actually have meaningful trade-offs or complexity? Would an implementer hit a "wait, how exactly should I do this?" moment?

4. **Missing details or decisions** — What does the plan fail to mention entirely? Are there steps, edge cases, error conditions, or dependencies that aren't addressed?

5. **Under-specified elements** — Where does the plan use vague language ("handle appropriately", "adjust as needed", "similar to X") instead of concrete instructions?

6. **Violations of project norms** — Does the plan propose doing something inconsistent with how the rest of the project works? Check existing patterns for naming, file organization, code style, data handling, etc. Functionality and data structures that are confined to a certain scope (file, module, directory, layer, class) should stay confined to that scope — e.g., if database operations only ever happen in the service layer, the plan should not introduce them elsewhere. If the plan intentionally breaks an established pattern, it must explicitly acknowledge and justify the deviation.

7. **Bad practices** — Does the plan introduce anything that would degrade code quality, maintainability, or correctness? Unnecessary complexity, duplication, brittle assumptions, etc.

## Process

1. Read the plan document
2. Read all files the plan references or modifies (skim large files, read relevant sections carefully)
3. For each criterion, investigate thoroughly — don't just skim the plan, cross-reference against the actual codebase
4. Present findings organized by criterion
5. For each issue, state: what the problem is, where in the plan it occurs, and what needs to change

## Output

Present the critique as a numbered list grouped by criterion. Be direct — if something is wrong, say so clearly. If the plan is solid on a criterion, say "no issues found" and move on. Don't pad the review with praise.

After the critique, give an overall assessment: is this plan ready for implementation, or does it need revision first?
