"""Benchmark driver for Option 5b — runs the three encodings on the
queens-with-regions instances bundled in `queens_bench/` and emits
per-row timings as CSV.

Usage:

    python bench_opt5b.py                    # writes results.csv
    python bench_opt5b.py my_results.csv     # custom output path

Once your three `solve_queens_*` functions work, run this and import the
resulting CSV into a spreadsheet (Excel, Google Sheets, etc.) to produce
the runtime chart that Option 5b asks you to submit.

The bundled examples are a mix of SAT and UNSAT partitions over N=4..9.
Each row of the CSV is `encoding, name, N, expected, seconds, status`.
`status` is `sat`, `unsat_or_none`, or `error:<ExceptionType>`. The
driver times only `solve_queens_*`; loading the example is excluded.
"""
import csv
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from opt5b_queens import solve_queens_sat, solve_queens_smt, solve_queens_lia


_ENCODINGS = [
    ("sat", solve_queens_sat),
    ("smt", solve_queens_smt),
    ("lia", solve_queens_lia),
]

_BENCH_DIR = _HERE / "queens_bench"


def _load_examples():
    examples = []
    for path in sorted(_BENCH_DIR.glob("seed_*.json")):
        data = json.loads(path.read_text())
        n = data["instance"]["n"]
        regions = [[tuple(cell) for cell in region]
                   for region in data["instance"]["regions"]]
        examples.append((path.stem, n, regions, data.get("expected")))
    return examples


def _classify(out):
    if out is None:
        return "unsat_or_none"
    return "sat"


def main(out_path="results.csv"):
    examples = _load_examples()
    if not examples:
        raise RuntimeError(f"no examples found under {_BENCH_DIR}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoding", "name", "N", "expected", "seconds", "status"])
        for name, n, regions, expected in examples:
            for enc_name, fn in _ENCODINGS:
                t0 = time.perf_counter()
                try:
                    out = fn(n, regions)
                    dt = time.perf_counter() - t0
                    status = _classify(out)
                except Exception as e:
                    dt = time.perf_counter() - t0
                    status = f"error:{type(e).__name__}"
                w.writerow([enc_name, name, n,
                            expected if expected is not None else "UNSAT",
                            f"{dt:.4f}", status])
                print(f"{enc_name:>3}  {name}  N={n:>2}  "
                      f"{(expected or 'UNSAT'):<5}  {dt:7.3f}s  {status}",
                      flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    main(out)
