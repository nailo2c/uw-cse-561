"""BitBlastSolver — a Solver tuned for bit-vector queries.

The default Z3 Solver runs a preprocessing pipeline that doesn't always do
the right thing for QF_BV problems built mostly out of equalities and small
integer arithmetic. This helper instead returns a Solver that eagerly
bit-blasts your bit vectors into pure SAT before solving — much faster on
the magic-square problem, and on bit-vector problems in general.

You use it exactly like Z3's Solver:

    s = BitBlastSolver()
    s.add(...)
    s.check()
    s.model()
"""
from z3 import Then


def BitBlastSolver():
    return Then("simplify", "bit-blast", "simplify", "tseitin-cnf", "sat").solver()
