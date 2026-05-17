"""Tests for opt3_family.py — run with `python test_opt3_family.py`."""
from family_tree import AgeFact, ParentFact, GENERATION_GAP
from opt3_family import solve_family_tree


def _eval_age_fact(fact, ages):
    lhs = ages[fact.person]
    rhs = ages[fact.value] if isinstance(fact.value, str) else fact.value
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<":  lhs <  rhs,
        "<=": lhs <= rhs,
        ">":  lhs >  rhs,
        ">=": lhs >= rhs,
    }[fact.op]


def _validate(people, age_facts, parent_facts, ages):
    if ages is None:
        return False
    if set(ages.keys()) != set(people):
        return False
    for v in ages.values():
        if not isinstance(v, int) or v < 0:
            return False
    for f in age_facts:
        if not _eval_age_fact(f, ages):
            return False
    for f in parent_facts:
        if f.kind == "parent_of":
            child, par = f.a, f.b
            if ages[par] < ages[child] + GENERATION_GAP:
                return False
        elif f.kind == "same_parent":
            a, b = f.a, f.b
            ok = any(
                ages[p] >= ages[a] + GENERATION_GAP and
                ages[p] >= ages[b] + GENERATION_GAP
                for p in people
            )
            if not ok:
                return False
        else:
            return False
    return True


def test_1():
    people = ["Alice", "Bob", "Carol"]
    age_facts = [
        AgeFact("Alice", "==", 8),
        AgeFact("Carol", "==", 10),
    ]
    parent_facts = [
        ParentFact("parent_of", "Alice", "Bob"),
        ParentFact("parent_of", "Carol", "Bob"),
    ]
    ages = solve_family_tree(people, age_facts, parent_facts)
    assert _validate(people, age_facts, parent_facts, ages)


def test_2():
    people = ["Alice", "Bob", "Carol", "Dave"]
    age_facts = [
        AgeFact("Alice", "<",  "Bob"),
        AgeFact("Bob",   "!=", 30),
        AgeFact("Dave",  ">",  0),
    ]
    parent_facts = [ParentFact("same_parent", "Alice", "Bob")]
    ages = solve_family_tree(people, age_facts, parent_facts)
    assert _validate(people, age_facts, parent_facts, ages)


def test_3():
    people = ["Alice", "Bob"]
    age_facts = [
        AgeFact("Alice", ">=", 100),
        AgeFact("Bob",   "<=", 50),
    ]
    parent_facts = [ParentFact("parent_of", "Alice", "Bob")]
    assert solve_family_tree(people, age_facts, parent_facts) is None


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
