"""Tests for opt5a_lia_flatten.py — run with `python test_opt5a_lia_flatten.py`."""
from lia_cnf import Atom
from opt5a_lia_flatten import solve_lia_cnf


def _atom_holds(atom, sol):
    lhs = sum(c * sol[v] for v, c in atom.coeffs.items())
    return lhs <= atom.rhs


def _validate(variables, cnf, sol):
    if sol is None:
        return False
    if set(sol.keys()) != set(variables):
        return False
    for v in sol.values():
        if not isinstance(v, int):
            return False
    for clause in cnf:
        if not any(_atom_holds(a, sol) for a in clause):
            return False
    return True


def test_1():
    variables = ["x", "y"]
    cnf = [
        [Atom({"x": -1}, -5), Atom({"y": 1}, 3)],
        [Atom({"x": 1, "y": 1}, 10)],
        [Atom({"x": -1, "y": -1}, -10)],
    ]
    sol = solve_lia_cnf(variables, cnf)
    assert _validate(variables, cnf, sol)


def test_2():
    variables = ["x"]
    cnf = [[Atom({"x": 1}, 2)], [Atom({"x": -1}, -5)]]
    assert solve_lia_cnf(variables, cnf) is None


def test_3():
    variables = ["x", "y"]
    cnf = [
        [Atom({"x": 1}, 0), Atom({"y": 1}, 0)],
        [Atom({"x": -1}, -1)],
        [Atom({"y": -1}, -1)],
    ]
    assert solve_lia_cnf(variables, cnf) is None


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
