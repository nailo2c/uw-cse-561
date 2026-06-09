#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Find bounded TriggerRuleDep counterexamples with Z3.

This models the direct-upstream branch of
``airflow.ti_deps.deps.trigger_rule_dep.TriggerRuleDep._evaluate_direct_relatives``.

The model includes:

- ``is_mapped`` boolean: corresponds to whether ``ti.map_index > -1`` in the
  implementation (mapped task instances subtract ``removed`` from failure counts).
- ``flag_upstream_failed`` boolean: corresponds to ``dep_context.flag_upstream_failed``.
  When True, the implementation may rewrite ``ti.state`` to a terminal state.
- ``removed`` upstream count, no longer constrained to zero.

It still excludes:

- Setup and teardown constraints (``_evaluate_setup_constraint`` and
  ``_evaluate_teardown_scope``).
- ``ALL_DONE_SETUP_SUCCESS`` rule (requires setup-specific upstream state).
- ``wait_for_past_depends_before_skipping`` flag.
- The REMOVED rewrite branch for ``ALL_SUCCESS`` that depends on the specific
  ``ti.map_index`` value (``ti.map_index >= success``). The terminal-state
  rewrite for this rule is under-approximated; counterexamples involving
  ``ALL_SUCCESS`` with ``removed > 0`` and ``is_mapped=True`` should be
  manually cross-checked with the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from z3 import And, Bool, BoolRef, BoolVal, If, Int, Not, Optimize, Or, sat

MAX_UPSTREAM = 5

Direction = Literal[
    "implementation_accepts_spec_rejects",
    "spec_accepts_implementation_rejects",
]


class Rule(str, Enum):
    """Trigger rules modeled from Airflow's direct upstream dependency logic.

    ``all_done_setup_success`` is intentionally excluded; it requires upstream
    setup state that this model does not represent. Note that the trigger-rule
    docs in ``core-concepts/dags.rst`` also omit this rule, so the docs are
    incomplete relative to ``airflow.task.trigger_rule.TriggerRule``.
    """

    ALL_SUCCESS = "all_success"
    ALL_FAILED = "all_failed"
    ALL_DONE = "all_done"
    ALL_DONE_MIN_ONE_SUCCESS = "all_done_min_one_success"
    ALL_SKIPPED = "all_skipped"
    ONE_FAILED = "one_failed"
    ONE_SUCCESS = "one_success"
    ONE_DONE = "one_done"
    NONE_FAILED = "none_failed"
    NONE_FAILED_MIN_ONE_SUCCESS = "none_failed_min_one_success"
    NONE_SKIPPED = "none_skipped"
    ALWAYS = "always"


@dataclass(frozen=True)
class Counts:
    """Z3 integer variables for upstream task instance state counts."""

    success: object
    skipped: object
    failed: object
    upstream_failed: object
    removed: object
    unfinished: object

    @property
    def done(self) -> object:
        return self.success + self.skipped + self.failed + self.upstream_failed + self.removed

    @property
    def upstream(self) -> object:
        return self.done + self.unfinished

    @property
    def non_skipped_done(self) -> object:
        return self.success + self.failed + self.upstream_failed + self.removed

    @property
    def non_skipped_upstream(self) -> object:
        return self.upstream - self.skipped


@dataclass(frozen=True)
class ConcreteCounts:
    """Concrete upstream task instance state counts from a satisfying model."""

    success: int
    skipped: int
    failed: int
    upstream_failed: int
    removed: int
    unfinished: int

    @property
    def done(self) -> int:
        return self.success + self.skipped + self.failed + self.upstream_failed + self.removed

    @property
    def upstream(self) -> int:
        return self.done + self.unfinished

    @property
    def non_skipped_done(self) -> int:
        return self.success + self.failed + self.upstream_failed + self.removed

    @property
    def non_skipped_upstream(self) -> int:
        return self.upstream - self.skipped

    def as_state_dict(self) -> dict[str, int]:
        return {
            "success": self.success,
            "skipped": self.skipped,
            "failed": self.failed,
            "upstream_failed": self.upstream_failed,
            "removed": self.removed,
            "unfinished": self.unfinished,
        }


@dataclass(frozen=True)
class Counterexample:
    """A minimal model where TriggerRuleDep and the docs-derived spec disagree."""

    rule: Rule
    direction: Direction
    counts: ConcreteCounts
    is_mapped: bool
    flag_upstream_failed: bool
    dep_check_passes: bool
    rewrite_state: str | None
    would_run: bool
    docs_spec_passes: bool


def build_counts() -> Counts:
    """Build fresh Z3 variables for one solver query."""
    return Counts(
        success=Int("success"),
        skipped=Int("skipped"),
        failed=Int("failed"),
        upstream_failed=Int("upstream_failed"),
        removed=Int("removed"),
        unfinished=Int("unfinished"),
    )


def build_common_constraints(counts: Counts, max_upstream: int) -> list[BoolRef]:
    """Constrain the bounded direct-upstream model."""
    variables = (
        counts.success,
        counts.skipped,
        counts.failed,
        counts.upstream_failed,
        counts.removed,
        counts.unfinished,
    )
    return [
        *(variable >= 0 for variable in variables),
        counts.upstream >= 1,
        counts.upstream <= max_upstream,
    ]


def build_dependency_pass_condition(rule: Rule, counts: Counts, is_mapped: BoolRef) -> BoolRef:
    """Model the ``_failing_status`` checks in ``_evaluate_direct_relatives``.

    Returns the Z3 condition under which no failing status is yielded for the
    direct-upstream branch.
    """
    if rule == Rule.ALWAYS:
        return BoolVal(True)
    if rule == Rule.ONE_SUCCESS:
        return counts.success > 0
    if rule == Rule.ONE_FAILED:
        return Or(counts.failed > 0, counts.upstream_failed > 0)
    if rule == Rule.ONE_DONE:
        return counts.success + counts.failed > 0
    if rule == Rule.ALL_SUCCESS:
        return If(
            is_mapped,
            counts.upstream - counts.success - counts.removed <= 0,
            counts.upstream - counts.success <= 0,
        )
    if rule == Rule.ALL_FAILED:
        return If(
            is_mapped,
            counts.upstream - counts.failed - counts.upstream_failed - counts.removed <= 0,
            counts.upstream - counts.failed - counts.upstream_failed <= 0,
        )
    if rule == Rule.ALL_DONE:
        return counts.done >= counts.upstream
    if rule in (Rule.NONE_FAILED, Rule.NONE_FAILED_MIN_ONE_SUCCESS):
        return If(
            is_mapped,
            counts.upstream - counts.success - counts.skipped - counts.removed <= 0,
            counts.upstream - counts.success - counts.skipped <= 0,
        )
    if rule == Rule.NONE_SKIPPED:
        return And(counts.done >= counts.upstream, counts.skipped == 0)
    if rule == Rule.ALL_SKIPPED:
        return counts.upstream - counts.skipped <= 0
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        not_any_skipped = counts.skipped == 0
        all_non_skipped_done = If(
            is_mapped,
            counts.non_skipped_done - counts.removed >= counts.non_skipped_upstream - counts.removed,
            counts.non_skipped_done >= counts.non_skipped_upstream,
        )
        at_least_one_success = counts.success > 0
        return And(not_any_skipped, all_non_skipped_done, at_least_one_success)
    raise ValueError(f"Unsupported trigger rule: {rule}")


def build_terminal_state_rewrite(rule: Rule, counts: Counts, is_mapped: BoolRef) -> BoolRef:
    """Model whether ``ti.state`` would be rewritten to a terminal state.

    Only applies when ``dep_context.flag_upstream_failed`` is True.
    Terminal states: SKIPPED, UPSTREAM_FAILED, REMOVED.

    The REMOVED rewrite for ALL_SUCCESS depends on the specific ``ti.map_index``
    value relative to ``success``; this model does not include that branch.
    """
    upstream_done = counts.done >= counts.upstream
    if rule == Rule.ALL_SUCCESS:
        return Or(
            counts.upstream_failed > 0,
            counts.failed > 0,
            counts.skipped > 0,
        )
    if rule == Rule.ALL_FAILED:
        return Or(counts.success > 0, counts.skipped > 0)
    if rule == Rule.ONE_SUCCESS:
        return Or(
            And(upstream_done, counts.done == counts.skipped),
            And(upstream_done, counts.success <= 0),
        )
    if rule == Rule.ONE_FAILED:
        return And(upstream_done, counts.failed == 0, counts.upstream_failed == 0)
    if rule == Rule.ONE_DONE:
        return And(upstream_done, counts.failed == 0, counts.success == 0)
    if rule == Rule.NONE_FAILED:
        return Or(counts.upstream_failed > 0, counts.failed > 0)
    if rule == Rule.NONE_FAILED_MIN_ONE_SUCCESS:
        return Or(
            counts.upstream_failed > 0,
            counts.failed > 0,
            counts.skipped == counts.upstream,
        )
    if rule == Rule.NONE_SKIPPED:
        return counts.skipped > 0
    if rule == Rule.ALL_SKIPPED:
        return Or(counts.success > 0, counts.failed > 0, counts.upstream_failed > 0)
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        # The rewrite branch in trigger_rule_dep.py uses the unmapped form of
        # non_skipped_done / non_skipped_upstream (it does not subtract removed
        # for mapped tasks). Source: trigger_rule_dep.py lines 428-438.
        return Or(
            counts.skipped > 0,
            And(
                counts.non_skipped_done >= counts.non_skipped_upstream,
                counts.success == 0,
            ),
        )
    if rule in (Rule.ALWAYS, Rule.ALL_DONE):
        return BoolVal(False)
    raise ValueError(f"Unsupported trigger rule: {rule}")


def build_effective_can_run(
    rule: Rule,
    counts: Counts,
    is_mapped: BoolRef,
    flag_upstream_failed: BoolRef,
) -> BoolRef:
    """Combine dep check and rewrite into 'task effectively runs' predicate.

    A task effectively runs when the dep check passes AND the terminal-state
    rewrite does not fire (or ``flag_upstream_failed`` is False).
    """
    dep_passes = build_dependency_pass_condition(rule, counts, is_mapped)
    rewrite_terminates = build_terminal_state_rewrite(rule, counts, is_mapped)
    return And(dep_passes, Not(And(flag_upstream_failed, rewrite_terminates)))


def build_docs_spec_pass_condition(rule: Rule, counts: Counts) -> BoolRef:
    """Model the prose trigger-rule definitions in ``core-concepts/dags.rst``.

    The docs do not differentiate by ``ti.map_index`` or ``removed`` state, so
    this returns the same condition for any (is_mapped, removed) combination.
    """
    if rule == Rule.ALWAYS:
        return BoolVal(True)
    if rule == Rule.ALL_SUCCESS:
        return counts.success == counts.upstream
    if rule == Rule.ALL_FAILED:
        return counts.failed + counts.upstream_failed == counts.upstream
    if rule == Rule.ALL_DONE:
        return counts.unfinished == 0
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        return And(counts.unfinished == 0, counts.success > 0)
    if rule == Rule.ALL_SKIPPED:
        return counts.skipped == counts.upstream
    if rule == Rule.ONE_FAILED:
        return Or(counts.failed > 0, counts.upstream_failed > 0)
    if rule == Rule.ONE_SUCCESS:
        return counts.success > 0
    if rule == Rule.ONE_DONE:
        return counts.success + counts.failed > 0
    if rule == Rule.NONE_FAILED:
        return And(counts.unfinished == 0, counts.failed == 0, counts.upstream_failed == 0)
    if rule == Rule.NONE_FAILED_MIN_ONE_SUCCESS:
        return And(
            counts.unfinished == 0,
            counts.failed == 0,
            counts.upstream_failed == 0,
            counts.success > 0,
        )
    if rule == Rule.NONE_SKIPPED:
        return And(counts.unfinished == 0, counts.skipped == 0)
    raise ValueError(f"Unsupported trigger rule: {rule}")


def evaluate_dependency_pass(rule: Rule, counts: ConcreteCounts, is_mapped: bool) -> bool:
    """Concrete counterpart to ``build_dependency_pass_condition``."""
    if rule == Rule.ALWAYS:
        return True
    if rule == Rule.ONE_SUCCESS:
        return counts.success > 0
    if rule == Rule.ONE_FAILED:
        return counts.failed > 0 or counts.upstream_failed > 0
    if rule == Rule.ONE_DONE:
        return counts.success + counts.failed > 0
    if rule == Rule.ALL_SUCCESS:
        num_failures = counts.upstream - counts.success
        if is_mapped:
            num_failures -= counts.removed
        return num_failures <= 0
    if rule == Rule.ALL_FAILED:
        num_success = counts.upstream - counts.failed - counts.upstream_failed
        if is_mapped:
            num_success -= counts.removed
        return num_success <= 0
    if rule == Rule.ALL_DONE:
        return counts.done >= counts.upstream
    if rule in (Rule.NONE_FAILED, Rule.NONE_FAILED_MIN_ONE_SUCCESS):
        num_failures = counts.upstream - counts.success - counts.skipped
        if is_mapped:
            num_failures -= counts.removed
        return num_failures <= 0
    if rule == Rule.NONE_SKIPPED:
        return counts.done >= counts.upstream and counts.skipped == 0
    if rule == Rule.ALL_SKIPPED:
        return counts.upstream - counts.skipped <= 0
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        non_skipped_done = counts.non_skipped_done
        non_skipped_upstream = counts.non_skipped_upstream
        if is_mapped:
            non_skipped_done -= counts.removed
            non_skipped_upstream -= counts.removed
        if counts.skipped > 0:
            return False
        if non_skipped_done < non_skipped_upstream:
            return False
        if counts.success == 0:
            return False
        return True
    raise ValueError(f"Unsupported trigger rule: {rule}")


def evaluate_rewrite_state(rule: Rule, counts: ConcreteCounts, is_mapped: bool) -> str | None:
    """Concrete counterpart to ``build_terminal_state_rewrite``.

    Returns "skipped", "upstream_failed", "removed", or None. The REMOVED
    rewrite branch for ALL_SUCCESS is not modeled (see module docstring).
    """
    upstream_done = counts.done >= counts.upstream
    if rule == Rule.ALL_SUCCESS:
        if counts.upstream_failed or counts.failed:
            return "upstream_failed"
        if counts.skipped:
            return "skipped"
        return None
    if rule == Rule.ALL_FAILED:
        if counts.success or counts.skipped:
            return "skipped"
        return None
    if rule == Rule.ONE_SUCCESS:
        if upstream_done and counts.done == counts.skipped:
            return "skipped"
        if upstream_done and counts.success <= 0:
            return "upstream_failed"
        return None
    if rule == Rule.ONE_FAILED:
        if upstream_done and not (counts.failed or counts.upstream_failed):
            return "skipped"
        return None
    if rule == Rule.ONE_DONE:
        if upstream_done and not (counts.failed or counts.success):
            return "skipped"
        return None
    if rule == Rule.NONE_FAILED:
        if counts.upstream_failed or counts.failed:
            return "upstream_failed"
        return None
    if rule == Rule.NONE_FAILED_MIN_ONE_SUCCESS:
        if counts.upstream_failed or counts.failed:
            return "upstream_failed"
        if counts.skipped == counts.upstream:
            return "skipped"
        return None
    if rule == Rule.NONE_SKIPPED:
        if counts.skipped:
            return "skipped"
        return None
    if rule == Rule.ALL_SKIPPED:
        if counts.success or counts.failed or counts.upstream_failed:
            return "skipped"
        return None
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        if counts.skipped > 0:
            return "skipped"
        if counts.non_skipped_done >= counts.non_skipped_upstream and counts.success == 0:
            return "upstream_failed"
        return None
    if rule in (Rule.ALWAYS, Rule.ALL_DONE):
        return None
    raise ValueError(f"Unsupported trigger rule: {rule}")


def evaluate_effective_can_run(
    rule: Rule,
    counts: ConcreteCounts,
    is_mapped: bool,
    flag_upstream_failed: bool,
) -> bool:
    """Concrete counterpart to ``build_effective_can_run``."""
    dep_passes = evaluate_dependency_pass(rule, counts, is_mapped)
    rewrite = evaluate_rewrite_state(rule, counts, is_mapped)
    if flag_upstream_failed and rewrite is not None:
        return False
    return dep_passes


def evaluate_docs_spec(rule: Rule, counts: ConcreteCounts) -> bool:
    """Concrete counterpart to ``build_docs_spec_pass_condition``."""
    if rule == Rule.ALWAYS:
        return True
    if rule == Rule.ALL_SUCCESS:
        return counts.success == counts.upstream
    if rule == Rule.ALL_FAILED:
        return counts.failed + counts.upstream_failed == counts.upstream
    if rule == Rule.ALL_DONE:
        return counts.unfinished == 0
    if rule == Rule.ALL_DONE_MIN_ONE_SUCCESS:
        return counts.unfinished == 0 and counts.success > 0
    if rule == Rule.ALL_SKIPPED:
        return counts.skipped == counts.upstream
    if rule == Rule.ONE_FAILED:
        return counts.failed > 0 or counts.upstream_failed > 0
    if rule == Rule.ONE_SUCCESS:
        return counts.success > 0
    if rule == Rule.ONE_DONE:
        return counts.success + counts.failed > 0
    if rule == Rule.NONE_FAILED:
        return counts.unfinished == 0 and counts.failed == 0 and counts.upstream_failed == 0
    if rule == Rule.NONE_FAILED_MIN_ONE_SUCCESS:
        return (
            counts.unfinished == 0
            and counts.failed == 0
            and counts.upstream_failed == 0
            and counts.success > 0
        )
    if rule == Rule.NONE_SKIPPED:
        return counts.unfinished == 0 and counts.skipped == 0
    raise ValueError(f"Unsupported trigger rule: {rule}")


def find_counterexample(
    rule: Rule,
    direction: Direction,
    *,
    fix_is_mapped: bool,
    fix_flag_upstream_failed: bool,
    max_upstream: int = MAX_UPSTREAM,
) -> Counterexample | None:
    """Find a minimal bounded counterexample for one (rule, direction, scenario)."""
    counts = build_counts()
    is_mapped = Bool("is_mapped")
    flag_upstream_failed = Bool("flag_upstream_failed")

    can_run = build_effective_can_run(rule, counts, is_mapped, flag_upstream_failed)
    docs_passes = build_docs_spec_pass_condition(rule, counts)

    solver = Optimize()
    solver.add(*build_common_constraints(counts, max_upstream))
    solver.add(is_mapped if fix_is_mapped else Not(is_mapped))
    solver.add(flag_upstream_failed if fix_flag_upstream_failed else Not(flag_upstream_failed))

    if direction == "implementation_accepts_spec_rejects":
        solver.add(can_run, Not(docs_passes))
    else:
        solver.add(docs_passes, Not(can_run))

    solver.minimize(counts.upstream)
    solver.minimize(counts.success)
    solver.minimize(counts.skipped)
    solver.minimize(counts.failed)
    solver.minimize(counts.upstream_failed)
    solver.minimize(counts.removed)
    solver.minimize(counts.unfinished)

    if solver.check() != sat:
        return None

    model = solver.model()
    concrete = ConcreteCounts(
        success=model.eval(counts.success).as_long(),
        skipped=model.eval(counts.skipped).as_long(),
        failed=model.eval(counts.failed).as_long(),
        upstream_failed=model.eval(counts.upstream_failed).as_long(),
        removed=model.eval(counts.removed).as_long(),
        unfinished=model.eval(counts.unfinished).as_long(),
    )
    return Counterexample(
        rule=rule,
        direction=direction,
        counts=concrete,
        is_mapped=fix_is_mapped,
        flag_upstream_failed=fix_flag_upstream_failed,
        dep_check_passes=evaluate_dependency_pass(rule, concrete, fix_is_mapped),
        rewrite_state=evaluate_rewrite_state(rule, concrete, fix_is_mapped),
        would_run=evaluate_effective_can_run(rule, concrete, fix_is_mapped, fix_flag_upstream_failed),
        docs_spec_passes=evaluate_docs_spec(rule, concrete),
    )


def iter_counterexamples() -> list[Counterexample]:
    """Find minimal counterexamples for every (rule, direction, is_mapped, flag)."""
    counterexamples: list[Counterexample] = []
    directions: tuple[Direction, ...] = (
        "implementation_accepts_spec_rejects",
        "spec_accepts_implementation_rejects",
    )
    for rule in Rule:
        for direction in directions:
            for is_mapped in (False, True):
                for flag_upstream_failed in (False, True):
                    counterexample = find_counterexample(
                        rule,
                        direction,
                        fix_is_mapped=is_mapped,
                        fix_flag_upstream_failed=flag_upstream_failed,
                    )
                    if counterexample:
                        counterexamples.append(counterexample)
    return counterexamples


def format_counterexample(counterexample: Counterexample) -> str:
    """Format one counterexample as a Markdown table row."""
    counts = counterexample.counts
    direction = {
        "implementation_accepts_spec_rejects": "impl accepts, spec rejects",
        "spec_accepts_implementation_rejects": "spec accepts, impl rejects",
    }[counterexample.direction]
    rewrite = counterexample.rewrite_state or "-"
    state_str = (
        f"success={counts.success}, skipped={counts.skipped}, failed={counts.failed}, "
        f"upstream_failed={counts.upstream_failed}, removed={counts.removed}, "
        f"unfinished={counts.unfinished}"
    )
    return (
        f"| `{counterexample.rule.value}` | {direction} | "
        f"is_mapped={counterexample.is_mapped}, "
        f"flag_upstream_failed={counterexample.flag_upstream_failed} | "
        f"{state_str} | {counterexample.dep_check_passes} | {rewrite} | "
        f"{counterexample.would_run} | {counterexample.docs_spec_passes} |"
    )


def main() -> None:
    """Run the bounded counterexample search and print Markdown output."""
    counterexamples = iter_counterexamples()
    print(f"# TriggerRuleDep Z3 counterexamples, max_upstream={MAX_UPSTREAM}")
    print()
    if not counterexamples:
        print("No counterexamples found.")
        return

    print(
        "| Rule | Direction | Scenario | Upstream counts | Dep check passes | "
        "Rewrite (if flag=True) | Would run | Docs spec passes |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for counterexample in counterexamples:
        print(format_counterexample(counterexample))


if __name__ == "__main__":
    main()
