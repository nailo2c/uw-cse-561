# Z3 Counterexamples for Airflow Trigger Rules

This mini-project uses Z3 to find semantic counterexamples between the prose
documentation for Apache Airflow trigger rules and the implementation in
`TriggerRuleDep`.

## Files

- `trigger_rule_counterexamples.py`: the bounded Z3 model and counterexample search.
- `main.py`: submission entry point that runs the counterexample search.
- `requirements.txt`: Python dependency list.
- `RESULTS.md`: expected output and interpretation of the discovered patterns.
- `WRITEUP.md`: the 1-2 page final write-up source.
- `WRITEUP.pdf`: the final write-up PDF.
- `DEMO_LINK.txt`: the demo recording link.

## Demo recording

The demo recording is available as an unlisted YouTube video:

https://youtu.be/pGnlRGU1etg?si=csd2tiu7Cb_7mXbz

## How to run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Expected runtime: under 5 seconds on a laptop.

Expected result: the script prints a Markdown table with 15 raw counterexample
rows. These collapse into 7 distinct disagreement patterns, summarized in
`RESULTS.md`.

## What the solver checks

The model compares two predicates:

- a docs-derived trigger-rule predicate, based on the Airflow trigger-rule prose;
- an implementation-derived predicate, based on the direct-upstream branch of
  `TriggerRuleDep._evaluate_direct_relatives`.

For each trigger rule, Z3 searches both disagreement directions:

- implementation accepts, docs reject;
- docs accept, implementation rejects.

The search is bounded to at most five upstream task instances, but the witnesses
found are minimal and small enough to inspect manually.

## Upstream impact

The solver results led to two Apache Airflow pull requests:

- https://github.com/apache/airflow/pull/67452, merged documentation PR clarifying
  trigger-rule behavior for the `removed` upstream state.
- https://github.com/apache/airflow/pull/67873, code PR fixing
  `none_failed_min_one_success` checks for zero-success upstream states.
