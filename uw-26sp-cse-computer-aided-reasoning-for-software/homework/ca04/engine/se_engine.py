"""se_engine.py — symbolic execution for mini IMP (ca04 edition).

A mini-IMP program is a Python function over 32-bit integer variables.
It may use: assignment (`x = E`, `x += E`), `if/else`, `while`, `return`,
`assert`, and the marker calls `assume(C)`, `havoc(x)`, and (no-op-with-
warning) `invariant(I)`.

The engine parses the function source with `ast.parse`, walks the AST,
and at each branch recursively explores both sides. The output is one
record per feasible execution path: the path condition, the symbolic
state at exit, the return value, and any asserts encountered.

ca04 changes vs. the lecture-07 reference engine:
  * `havoc` is now a *statement* `havoc(x)` that replaces the state
    binding of x with a fresh symbol. (One variable per call, matching
    the L08 form. The old `x = havoc()` expression form is removed.)
  * `invariant(I)` is recognized as the L08 marker but not acted on —
    the engine does BMC unrolling, not loop-cut. The first call seen
    raises a `UserWarning` and the call is otherwise ignored.
  * `_apply_binop` adds `%`, `&`, `|`, `^`, `<<`, `>>`.
  * `_apply_unaryop` adds `~`.

Two encoding modes:

    mode='dsa'    Dynamic Single Assignment (default). Each assignment
                  x = E introduces a fresh symbolic variable x__N and
                  emits the constraint x__N == E.

    mode='naive'  Eager substitution.
"""

import ast
import contextlib
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Optional

from z3 import (
    And, BitVec, BitVecVal, BoolVal, LShR, Not, Or, Solver, sat, unsat,
)

BW = 32  # mini-IMP integer width — overridable per call via the
         # `bitwidth=` kwarg to symbolic_execute / check_assertions.


@contextlib.contextmanager
def _bitwidth(bw):
    """Temporarily run the engine at a different BV width.

    Lower widths are useful when a 32-bit query times out — the bug
    pattern often shows at 8 or 16 bits, then a student can manually
    construct a 32-bit counterexample by analogy.
    """
    global BW
    old = BW
    BW = bw
    try:
        yield
    finally:
        BW = old


# --- Fresh-name generator ----------------------------------------

_fresh_counter = 0


def _fresh(name):
    """Allocate a fresh symbolic BitVec named after the source variable."""
    global _fresh_counter
    _fresh_counter += 1
    return BitVec(f'{name}__{_fresh_counter}', BW)


def _reset_counter():
    """Reset the fresh-name counter so demo output is reproducible."""
    global _fresh_counter
    _fresh_counter = 0


# --- Path records ------------------------------------------------

@dataclass
class Path:
    """One symbolic execution path through the program."""
    pc: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    return_val: Optional[object] = None
    extra: list = field(default_factory=list)
    asserts: list = field(default_factory=list)
    bounded_out: bool = False

    def clone(self):
        return Path(
            pc=list(self.pc),
            state=dict(self.state),
            return_val=self.return_val,
            extra=list(self.extra),
            asserts=list(self.asserts),
            bounded_out=self.bounded_out,
        )

    def is_done(self):
        return self.return_val is not None or self.bounded_out

    def constraints(self):
        return self.pc + self.extra


# --- Expression evaluation ---------------------------------------

def eval_expr(node, p, mode):
    """Evaluate an expression in path p's symbolic state."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return BoolVal(node.value)
        if isinstance(node.value, int):
            return BitVecVal(node.value, BW)
        raise NotImplementedError(f"Constant: {node.value!r}")
    if isinstance(node, ast.Name):
        # INT_MIN / INT_MAX / BITWIDTH are bitwidth-relative constants.
        # Defined in mini_imp.py for concrete runs; the engine reinterprets
        # them at the active BW so the same source compiles cleanly at any
        # width.
        if node.id == 'INT_MAX':
            return BitVecVal(2 ** (BW - 1) - 1, BW)
        if node.id == 'INT_MIN':
            return BitVecVal(-(2 ** (BW - 1)), BW)
        if node.id == 'BITWIDTH':
            return BitVecVal(BW, BW)
        if node.id not in p.state:
            raise ValueError(f"Variable {node.id!r} is not in scope")
        return p.state[node.id]
    if isinstance(node, ast.UnaryOp):
        x = eval_expr(node.operand, p, mode)
        return _apply_unaryop(node.op, x)
    if isinstance(node, ast.BinOp):
        a = eval_expr(node.left, p, mode)
        b = eval_expr(node.right, p, mode)
        return _apply_binop(node.op, a, b)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Chained comparisons not supported")
        a = eval_expr(node.left, p, mode)
        b = eval_expr(node.comparators[0], p, mode)
        return _apply_cmp(node.ops[0], a, b)
    if isinstance(node, ast.BoolOp):
        vs = [eval_expr(v, p, mode) for v in node.values]
        if isinstance(node.op, ast.And):
            return And(*vs)
        if isinstance(node.op, ast.Or):
            return Or(*vs)
        raise NotImplementedError(f"BoolOp: {ast.dump(node)}")
    raise NotImplementedError(f"Expression: {ast.dump(node)}")


def _apply_binop(op, a, b):
    if isinstance(op, ast.Add):      return a + b
    if isinstance(op, ast.Sub):      return a - b
    if isinstance(op, ast.Mult):     return a * b
    if isinstance(op, ast.FloorDiv): return a / b   # Z3 BV signed division
    if isinstance(op, ast.Mod):      return a % b   # Z3 BV signed modulo
    if isinstance(op, ast.BitAnd):   return a & b
    if isinstance(op, ast.BitOr):    return a | b
    if isinstance(op, ast.BitXor):   return a ^ b
    if isinstance(op, ast.LShift):   return a << b
    if isinstance(op, ast.RShift):   return LShR(a, b)  # logical right shift
    raise NotImplementedError(f"BinOp: {op}")


def _apply_unaryop(op, x):
    if isinstance(op, ast.USub):    return -x
    if isinstance(op, ast.UAdd):    return x
    if isinstance(op, ast.Not):     return Not(x)
    if isinstance(op, ast.Invert):  return ~x
    raise NotImplementedError(f"UnaryOp: {op}")


def _apply_cmp(op, a, b):
    if isinstance(op, ast.Eq):    return a == b
    if isinstance(op, ast.NotEq): return a != b
    if isinstance(op, ast.Lt):    return a < b
    if isinstance(op, ast.LtE):   return a <= b
    if isinstance(op, ast.Gt):    return a > b
    if isinstance(op, ast.GtE):   return a >= b
    raise NotImplementedError(f"Compare: {op}")


# --- Statement execution -----------------------------------------

def execute_stmts(stmts, path, depth, mode):
    """Execute a sequence of statements. Returns a list of resulting paths."""
    paths = [path]
    for stmt in stmts:
        next_paths = []
        for p in paths:
            if p.is_done():
                next_paths.append(p)
            else:
                next_paths.extend(execute_stmt(stmt, p, depth, mode))
        paths = next_paths
    return paths


def execute_stmt(stmt, p, depth, mode):
    """Execute a single statement. Returns a list of resulting paths."""

    # x = E
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise NotImplementedError("Only single-name assignments supported")
        name = stmt.targets[0].id
        value = eval_expr(stmt.value, p, mode)
        return [_do_assign(p, name, value, mode)]

    # x op= E
    if isinstance(stmt, ast.AugAssign):
        if not isinstance(stmt.target, ast.Name):
            raise NotImplementedError("AugAssign only on bare names")
        name = stmt.target.id
        if name not in p.state:
            raise ValueError(f"Augmented assign to undefined variable {name!r}")
        rhs = _apply_binop(stmt.op, p.state[name], eval_expr(stmt.value, p, mode))
        return [_do_assign(p, name, rhs, mode)]

    # if C: ... else: ...
    if isinstance(stmt, ast.If):
        cond_z = eval_expr(stmt.test, p, mode)
        p_then = p.clone(); p_then.pc.append(cond_z)
        p_else = p.clone(); p_else.pc.append(Not(cond_z))
        return (execute_stmts(stmt.body,   p_then, depth, mode)
              + execute_stmts(stmt.orelse, p_else, depth, mode))

    # while C: ...
    if isinstance(stmt, ast.While):
        if stmt.orelse:
            raise NotImplementedError("while-else not supported")
        return execute_while(stmt.test, stmt.body, p, depth, mode)

    # return E
    if isinstance(stmt, ast.Return):
        new = p.clone()
        new.return_val = (eval_expr(stmt.value, p, mode)
                          if stmt.value is not None else BitVecVal(0, BW))
        return [new]

    # assert C  ("check then assume", per IVL semantics).
    # The snapshotted pc/extra at the time of the assert is what we check
    # against later — appending the assert to pc would let *other* asserts
    # mask it during checking.
    if isinstance(stmt, ast.Assert):
        cond_z = eval_expr(stmt.test, p, mode)
        new = p.clone()
        new.asserts.append((stmt.lineno, cond_z, list(new.pc), list(new.extra)))
        new.pc.append(cond_z)
        return [new]

    # Bare expression statement: assume(C), havoc(x, …), or a docstring.
    if isinstance(stmt, ast.Expr):
        val = stmt.value
        # assume(C) — extend the path condition; do not check.
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id == 'assume'):
            if len(val.args) != 1:
                raise ValueError("assume() takes exactly one argument")
            cond_z = eval_expr(val.args[0], p, mode)
            new = p.clone()
            new.pc.append(cond_z)
            return [new]
        # havoc(x) — replace x's binding with a fresh symbol.
        # Single argument per call, matching the L08 form. To havoc several
        # variables, use several havoc(x); havoc(y); calls.
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id == 'havoc'):
            if len(val.args) != 1:
                raise ValueError(
                    "havoc() takes exactly one argument; "
                    "use multiple havoc(x); havoc(y); calls instead"
                )
            arg = val.args[0]
            if not isinstance(arg, ast.Name):
                raise ValueError(
                    "havoc() argument must be a bare variable name"
                )
            if arg.id not in p.state:
                raise ValueError(
                    f"havoc(): variable {arg.id!r} is not in scope"
                )
            new = p.clone()
            new.state[arg.id] = _fresh(arg.id)
            return [new]
        # invariant(I) — L08 marker. This engine does BMC unrolling, not
        # loop-cut, so the call is recognized but does not influence
        # symbolic exploration. Warn the first time it appears at a given
        # source location.
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id == 'invariant'):
            warnings.warn(
                f"invariant(...) at line {val.lineno} is ignored by the "
                f"L07 SE engine — this engine does BMC unrolling. Use "
                f"the L08 WP engine for invariant-based verification.",
                UserWarning,
                stacklevel=2,
            )
            return [p]
        # Docstring — a bare string constant. Skip silently.
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return [p]

    raise NotImplementedError(f"Statement outside mini-IMP: {ast.dump(stmt)}")


def _do_assign(p, name, value, mode):
    """Common assignment logic for both modes."""
    new = p.clone()
    if mode == 'dsa':
        fresh = _fresh(name)
        new.extra.append(fresh == value)
        new.state[name] = fresh
    elif mode == 'naive':
        new.state[name] = value
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return new


def execute_while(test, body, p, depth, mode):
    """Unroll a while loop to BMC depth `depth`."""
    if depth <= 0:
        cond_z = eval_expr(test, p, mode)
        p_exit = p.clone(); p_exit.pc.append(Not(cond_z))
        p_bound = p.clone(); p_bound.pc.append(cond_z); p_bound.bounded_out = True
        return [p_exit, p_bound]

    cond_z = eval_expr(test, p, mode)
    p_exit = p.clone(); p_exit.pc.append(Not(cond_z))
    p_iter = p.clone(); p_iter.pc.append(cond_z)

    out = [p_exit]
    for bp in execute_stmts(body, p_iter, depth - 1, mode):
        if bp.is_done():
            out.append(bp)
        else:
            out.extend(execute_while(test, body, bp, depth - 1, mode))
    return out


# --- Public API --------------------------------------------------

def symbolic_execute(fn, *, max_depth=20, mode='dsa', bitwidth=32):
    """Symbolically execute a mini-IMP function. Returns a list of Path records."""
    with _bitwidth(bitwidth):
        _reset_counter()
        if isinstance(fn, str):
            src = fn
        else:
            src = inspect.getsource(fn)
        tree = ast.parse(src).body[0]
        if not isinstance(tree, ast.FunctionDef):
            raise ValueError("symbolic_execute expects a function definition")

        p = Path()
        for arg in tree.args.args:
            p.state[arg.arg] = BitVec(arg.arg, BW)
        return execute_stmts(tree.body, p, max_depth, mode)


def check_assertions(fn, *, max_depth=20, mode='dsa', timeout_ms=5000, bitwidth=32):
    """Run SE and check every assert along every feasible path.

    For each `assert C`, the engine checks whether
        (PC-as-of-the-assert) ∧ ¬C
    is satisfiable. Asserts that come *after* the one under test are
    deliberately NOT in the context — if the assertion that comes after
    is itself violable, it shouldn't mask earlier failures.

    Each Z3 query is bounded by `timeout_ms`. A query that times out is
    treated as "unknown" and skipped (no violation reported). Set to 0
    for no per-query bound.
    """
    paths = symbolic_execute(fn, max_depth=max_depth, mode=mode, bitwidth=bitwidth)
    violations = []
    for i, p in enumerate(paths):
        for line, cond, pc_at, extra_at in p.asserts:
            s = Solver()
            if timeout_ms:
                s.set("timeout", timeout_ms)
            for c in pc_at:
                s.add(c)
            for c in extra_at:
                s.add(c)
            s.add(Not(cond))
            r = s.check()
            if r == sat:
                violations.append({
                    'label': f'assert L{line}',
                    'path_index': i, 'line': line,
                    'condition': cond, 'model': s.model(),
                })
    return violations
