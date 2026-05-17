"""Tests for opt2_sliding.py — run with `python test_opt2_sliding.py`."""
from opt2_sliding import solve_sliding_puzzle


def _validate_trace(N, start, goal, K, trace):
    if trace is None:
        return False
    if len(trace) != K + 1:
        return False
    if trace[0] != start.index(0):
        return False
    if trace[-1] != goal.index(0):
        return False
    board = list(start)
    for i in range(K):
        a, b = trace[i], trace[i + 1]
        if not (0 <= a < N * N and 0 <= b < N * N):
            return False
        ra, ca = divmod(a, N)
        rb, cb = divmod(b, N)
        same_row_adj = (ra == rb and abs(ca - cb) == 1)
        same_col_adj = (ca == cb and abs(ra - rb) == 1)
        if not (same_row_adj or same_col_adj):
            return False
        if board[a] != 0:
            return False
        board[a], board[b] = board[b], board[a]
    return board == list(goal)


def test_1():
    N = 3
    start = [0, 2, 3, 1, 4, 5, 7, 8, 6]
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    trace = solve_sliding_puzzle(N, start, goal, 4)
    assert _validate_trace(N, start, goal, 4, trace)


def test_2():
    N = 3
    start = [0, 2, 3, 1, 4, 5, 7, 8, 6]
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    assert solve_sliding_puzzle(N, start, goal, 3) is None


def test_3():
    N = 3
    start = [2, 1, 3, 4, 5, 6, 7, 8, 0]
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    assert solve_sliding_puzzle(N, start, goal, 20) is None


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
