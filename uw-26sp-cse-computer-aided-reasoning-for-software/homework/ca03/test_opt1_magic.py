"""Tests for opt1_magic.py — run with `python test_opt1_magic.py`."""
from opt1_magic import solve_magic_square


def _is_magic(square):
    if square is None:
        return False
    N = len(square)
    if any(len(row) != N for row in square):
        return False
    T = N * (N * N + 1) // 2
    flat = [v for row in square for v in row]
    if sorted(flat) != list(range(1, N * N + 1)):
        return False
    for r in range(N):
        if sum(square[r]) != T:
            return False
    for c in range(N):
        if sum(square[r][c] for r in range(N)) != T:
            return False
    if sum(square[i][i] for i in range(N)) != T:
        return False
    if sum(square[i][N - 1 - i] for i in range(N)) != T:
        return False
    return True


def test_1():
    sq = solve_magic_square(3)
    assert _is_magic(sq)


def test_2():
    sq = solve_magic_square(5)
    assert _is_magic(sq)


def test_3():
    assert solve_magic_square(2) is None


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
