# Finding Airflow Trigger Rule Counterexamples with Z3

Aaron Chen / aaronyc

## Problem

Apache Airflow decides whether a task can run by evaluating trigger rules such as `all_success`, `none_failed`, and `none_failed_min_one_success` against the states of upstream task instances in a Dag. The input to this project is a trigger rule plus a bounded summary of upstream states: counts of `success`, `skipped`, `failed`, `upstream_failed`, `removed`, and unfinished task instances, plus two implementation flags for mapped task instances and terminal-state rewriting. The output is a minimal counterexample where the documentation-derived semantics and the implementation-derived semantics disagree. A solver is a good fit because the bug pattern is not one concrete execution trace; it is a small logical witness inside a combinatorial state space with two possible disagreement directions.

## Encoding

I encoded each upstream state count as a non-negative Z3 integer and bounded the total number of upstream task instances to at most five. I then wrote two predicates for each trigger rule. The first predicate models the public documentation: for example, `none_failed_min_one_success` means all upstream tasks are terminal, none are `failed` or `upstream_failed`, and at least one upstream task succeeded. The second predicate models the direct-upstream branch of Airflow's `TriggerRuleDep`: it includes the mapped-task branch where some rules subtract `removed` upstream task instances from failure counts, and the `flag_upstream_failed` branch where Airflow may rewrite the task instance to a terminal state. For each rule and scenario, I asked Z3 to satisfy either "implementation accepts and docs reject" or "docs accept and implementation rejects", then used optimization to minimize the upstream count and make the witness human-readable. I considered modeling individual upstream task instances directly, but count variables were enough for this part of the implementation and produced smaller explanations.

## Beats naive

A naive approach could manually read the docs and implementation, then hand-write a few unit tests. That found some likely issues, but it did not systematically answer whether there were other small disagreements, which direction each disagreement went, or whether the same pattern depended on mapped tasks or terminal-state rewriting. The solver made that precise: it enumerated both disagreement directions across 12 trigger rules and four scenarios, producing 15 raw witnesses that collapsed into 7 distinct patterns. The most useful "beats naive" moment was `none_failed_min_one_success`: the docs require at least one upstream success, but Z3 found zero-success witnesses where the dependency check still passed. That led to a concrete Airflow code PR with regression tests, while a separate family of witnesses around the `removed` state led to a merged documentation PR.

## What I learned

The main lesson is that formal modeling is useful even when the model is small and bounded. I did not need to prove the whole scheduler correct; translating one narrow contract into two predicates was enough to find real documentation drift and implementation edge cases in a large open-source project. I also learned that the hardest part is not writing Z3 syntax, but deciding what each side of the comparison means: documentation prose, implementation behavior, mapped task behavior, and terminal-state rewrites all describe slightly different contracts. The project used AI assistance from Codex/GPT-5 for code reading, Z3 model drafting, PR description drafting, and explanation review; I manually validated the claims against Airflow source code, CLI reproduction output, unit tests, and upstream pull request review.
