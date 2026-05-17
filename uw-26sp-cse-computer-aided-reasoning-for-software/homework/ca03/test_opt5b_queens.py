"""Tests for opt5b_queens.py — run with `python test_opt5b_queens.py`."""
from itertools import combinations

from opt5b_queens import solve_queens_sat, solve_queens_smt, solve_queens_lia


_ENCODINGS = [
    ("sat", solve_queens_sat),
    ("smt", solve_queens_smt),
    ("lia", solve_queens_lia),
]


def _attacks(p, q):
    (r1, c1), (r2, c2) = p, q
    return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)


def _is_valid(n, regions, placement):
    if placement is None:
        return False
    if len(placement) != n:
        return False
    region_sets = [set(region) for region in regions]
    for i, pos in enumerate(placement):
        if pos not in region_sets[i]:
            return False
    for i, j in combinations(range(n), 2):
        if _attacks(placement[i], placement[j]):
            return False
    return True


def test_1():
    n = 4
    regions = [[(r, c) for c in range(n)] for r in range(n)]
    for name, fn in _ENCODINGS:
        p = fn(n, regions)
        assert _is_valid(n, regions, p), name


def test_2():
    n = 3
    regions = [[(r, c) for c in range(n)] for r in range(n)]
    for name, fn in _ENCODINGS:
        assert fn(n, regions) is None, name


def test_3():
    n = 4
    regions = [[(r, c) for r in range(n)] for c in range(n)]
    for name, fn in _ENCODINGS:
        p = fn(n, regions)
        assert _is_valid(n, regions, p), name


if __name__ == "__main__":
    tests = [test_1, test_2, test_3]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(tests)} passed")
