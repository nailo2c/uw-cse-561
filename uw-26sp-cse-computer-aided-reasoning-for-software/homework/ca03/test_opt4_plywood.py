"""Tests for opt4_plywood.py — run with `python test_opt4_plywood.py`."""
from opt4_plywood import cut_plywood


def _valid(W, H, pieces, sol):
    if sol is None or len(sol) != len(pieces):
        return False
    for (w, h), (x, y) in zip(pieces, sol):
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            wi, hi = pieces[i]
            wj, hj = pieces[j]
            xi, yi = sol[i]
            xj, yj = sol[j]
            if not (xi + wi <= xj or xj + wj <= xi
                    or yi + hi <= yj or yj + hj <= yi):
                return False
    return True


def test_1():
    W, H = 10, 10
    pieces = [(4, 5), (3, 6), (5, 4)]
    sol = cut_plywood(W, H, pieces)
    assert _valid(W, H, pieces, sol)


def test_2():
    assert cut_plywood(10, 10, [(5, 6)] * 4) is None


def test_3():
    assert cut_plywood(10, 2, [(2, 10)]) is None


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
