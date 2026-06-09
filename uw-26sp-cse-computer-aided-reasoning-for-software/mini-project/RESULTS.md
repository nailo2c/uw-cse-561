# Results

Running `python trigger_rule_counterexamples.py` prints 15 raw counterexample
rows. These collapse into 7 distinct disagreement patterns.

| # | Rule | Direction | Minimal witness | Meaning |
| --- | --- | --- | --- | --- |
| 1 | `all_success` | implementation accepts, docs reject | `removed=1`, mapped downstream | The implementation can ignore a removed mapped upstream index, while the docs literally require every upstream to succeed. |
| 2 | `all_failed` | implementation accepts, docs reject | `removed=1`, mapped downstream | The implementation can ignore a removed mapped upstream index, while the docs literally require every upstream to fail. |
| 3 | `all_done_min_one_success` | docs accept, implementation rejects | `success=1, skipped=1` | The docs allow skipped upstreams, but the implementation rejects any skipped upstream. |
| 4 | `none_failed` | docs accept, implementation rejects | `removed=1`, non-mapped downstream | The docs say no upstream failed, but the implementation treats `removed` as a blocking non-success/non-skipped state for non-mapped downstreams. |
| 5 | `none_failed_min_one_success` | implementation accepts, docs reject | `skipped=1`, non-mapped, `flag_upstream_failed=False` | The dependency check did not enforce the "at least one success" part unless the rewrite path ran. |
| 6 | `none_failed_min_one_success` | implementation accepts, docs reject | `removed=1`, mapped downstream | All relevant mapped upstream task instances can be removed, leaving zero successes while the dependency check still passes. |
| 7 | `none_failed_min_one_success` | docs accept, implementation rejects | `success=1, removed=1`, non-mapped downstream | The docs accept because there is a success and no failure, but the implementation treats non-mapped `removed` as blocking. |

Two of these findings were turned into concrete Apache Airflow changes:

- PR 67452: documentation change, merged. It clarifies trigger-rule behavior for
  the `removed` upstream state.
- PR 67873: code change, open as of June 1, 2026. It fixes
  `none_failed_min_one_success` so zero-success upstream states fail the
  dependency check and mapped all-removed upstreams are rewritten to
  `UPSTREAM_FAILED` when the scheduler rewrite path runs.

The model intentionally does not cover setup/teardown constraints,
`all_done_setup_success`, `wait_for_past_depends_before_skipping`, or the exact
`all_success` removed-index rewrite that depends on a concrete `map_index`.
