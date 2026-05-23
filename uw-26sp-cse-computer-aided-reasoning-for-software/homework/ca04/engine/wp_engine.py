"""WP-based VCG for mini IMP with loop invariants.

A small verification condition generator. Given an annotated mini-IMP
function, it produces one VC per assertion (and per loop's entry,
preservation, and sufficiency obligations) and dispatches each to Z3.

Input language: Python subset (`mini_imp.py` markers `assume`, `havoc`,
`invariant`). Output: list of (label, result) records, one per
obligation.

The engine works in two passes:

    1. Loop-cut. Every `while C: invariant(I); body` is rewritten as:
           assert I;            # entry obligation
           havoc x; ...         # forget loop targets
           assume I;            # reason about an arbitrary iteration
           if C:
               body;
               assert I;        # preservation obligation
               assume False;    # dead path
           else:
               skip;            # loop exited; continue past

    2. Walk. The cut program is loop-free. Walking it forward with a
       symbolic state and an "assume context" produces one VC per
       assert: assume_context => assert_condition.

The walk is forward symbolic execution on the cut form. Since the
cut form is loop-free, per-assert forward SE is equivalent to
per-obligation backward WP, so the engine's output matches what the
WP rules from Theory would derive.
"""

import ast
import inspect

from z3 import (
    Int, And, Or, Not, Implies, BoolVal, IntVal,
    Solver, sat, unsat,
)


# --- Fresh-name counter --------------------------------------------

_fresh_counter = 0

def _fresh(name):
    """Allocate a fresh Z3 Int named after the source variable."""
    global _fresh_counter
    _fresh_counter += 1
    return Int(f'{name}__{_fresh_counter}')


def _reset_counter():
    """Reset the fresh-name counter (called at the start of each vcgen)."""
    global _fresh_counter
    _fresh_counter = 0


# --- Expression evaluation ----------------------------------------

def eval_expr(node, state):
    """Evaluate Python AST expression `node` against symbolic `state`."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return BoolVal(node.value)
        return IntVal(node.value)

    if isinstance(node, ast.Name):
        if node.id in state:
            return state[node.id]
        return Int(node.id)

    if isinstance(node, ast.UnaryOp):
        x = eval_expr(node.operand, state)
        if isinstance(node.op, ast.USub):
            return -x
        if isinstance(node.op, ast.Not):
            return Not(x)

    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, state)
        right = eval_expr(node.right, state)
        if isinstance(node.op, ast.Add):       return left + right
        if isinstance(node.op, ast.Sub):       return left - right
        if isinstance(node.op, ast.Mult):      return left * right
        if isinstance(node.op, ast.FloorDiv):  return left / right  # Z3 Int div

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise NotImplementedError('chained comparisons')
        left = eval_expr(node.left, state)
        right = eval_expr(node.comparators[0], state)
        op = node.ops[0]
        if isinstance(op, ast.Eq):     return left == right
        if isinstance(op, ast.NotEq):  return left != right
        if isinstance(op, ast.Lt):     return left < right
        if isinstance(op, ast.LtE):    return left <= right
        if isinstance(op, ast.Gt):     return left > right
        if isinstance(op, ast.GtE):    return left >= right

    if isinstance(node, ast.BoolOp):
        vs = [eval_expr(v, state) for v in node.values]
        if isinstance(node.op, ast.And):  return And(*vs)
        if isinstance(node.op, ast.Or):   return Or(*vs)

    raise NotImplementedError(f'eval_expr: {ast.dump(node)}')


# --- Identify marker calls ----------------------------------------

def _is_marker(stmt, name):
    """True if `stmt` is a top-level call to the mini-IMP marker `name`."""
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == name)


def _marker_arg(stmt):
    """Return the AST node for the marker call's single argument."""
    return stmt.value.args[0]


# --- Collect variables modified by a statement list ---------------

def _targets(stmts):
    """Return the set of variable names assigned anywhere in stmts."""
    out = set()
    for s in stmts:
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name):
                out.add(s.target.id)
        elif isinstance(s, ast.If):
            out |= _targets(s.body) | _targets(s.orelse)
        elif isinstance(s, ast.While):
            out |= _targets(s.body)
    return out


# --- Loop-cut transformation --------------------------------------

def loop_cut(stmts):
    """Replace every `while C: invariant(I); body` with the cut form.

    Returns a new list of statements. Recursive: loops inside if-branches
    are also cut. Synthetic asserts carry an `_obligation_kind` attribute
    ('entry' or 'preserved') so the walker can label them.
    """
    out = []
    for s in stmts:
        if isinstance(s, ast.While):
            if not (s.body and _is_marker(s.body[0], 'invariant')):
                raise ValueError(
                    f'while at line {s.lineno} needs invariant(...) as first stmt')
            I_node = _marker_arg(s.body[0])
            body = loop_cut(s.body[1:])
            targets = sorted(_targets(body))

            # assert I (entry obligation)
            out.append(_make_assert(I_node, s.lineno, kind='entry'))

            # havoc x (one per loop target)
            for tname in targets:
                out.append(_make_marker('havoc',
                                        ast.Name(id=tname, ctx=ast.Load()),
                                        s.lineno))

            # assume I (we now reason about an arbitrary iteration)
            out.append(_make_marker('assume', I_node, s.lineno))

            # if C: body; assert I (preserved); assume False; else: pass
            then_body = list(body)
            then_body.append(_make_assert(I_node, s.lineno, kind='preserved'))
            then_body.append(_make_marker('assume',
                                          ast.Constant(value=False),
                                          s.lineno))
            # orelse=[] is the "else skip" branch in the loop-cut form.
            out.append(ast.If(test=s.test, body=then_body, orelse=[],
                              lineno=s.lineno, col_offset=0,
                              end_lineno=s.lineno, end_col_offset=0))

        elif isinstance(s, ast.If):
            new_if = ast.If(test=s.test,
                            body=loop_cut(s.body),
                            orelse=loop_cut(s.orelse),
                            lineno=s.lineno, col_offset=0,
                            end_lineno=s.lineno, end_col_offset=0)
            out.append(new_if)

        else:
            out.append(s)

    return out


def _make_assert(test_node, lineno, kind=None):
    """Build a synthetic ast.Assert at `lineno`.

    If `kind` is given ('entry' or 'preserved'), tag the assert with
    an `_obligation_kind` attribute so the walker can label its VC.
    """
    a = ast.Assert(test=test_node, msg=None,
                   lineno=lineno, col_offset=0,
                   end_lineno=lineno, end_col_offset=0)
    if kind is not None:
        a._obligation_kind = kind
    return a


def _make_marker(name, arg, lineno):
    """Build a synthetic call to the marker function `name(arg)`."""
    return ast.Expr(value=ast.Call(
                        func=ast.Name(id=name, ctx=ast.Load()),
                        args=[arg], keywords=[]),
                    lineno=lineno, col_offset=0,
                    end_lineno=lineno, end_col_offset=0)


# --- Walk the cut IVL: emit per-assert VCs ------------------------

def walk(stmts, state, assumptions, vcs, label_of):
    """Forward walk over loop-free IVL. Emit VCs at each assert.

    `state` is the current symbolic state (var name -> Z3 expr).
    `assumptions` is the list of assume conditions accumulated.
    `vcs` is the output list of (label, formula) records.
    `label_of` makes a human-readable label from an assert AST node.

    When the walk hits an `if`, it forks into two continuations, each
    carrying the rest of the program after the if. (Same shape as L07's
    SE engine forking at branches.) When the walk hits `assume(False)`,
    it stops the current path silently.
    """
    for idx, stmt in enumerate(stmts):

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise NotImplementedError('single-name assignment only')
            name = stmt.targets[0].id
            value = eval_expr(stmt.value, state)
            state = {**state, name: value}

        elif isinstance(stmt, ast.AugAssign):
            name = stmt.target.id
            current = state.get(name, Int(name))
            rhs = eval_expr(stmt.value, state)
            if   isinstance(stmt.op, ast.Add):  new_val = current + rhs
            elif isinstance(stmt.op, ast.Sub):  new_val = current - rhs
            elif isinstance(stmt.op, ast.Mult): new_val = current * rhs
            else: raise NotImplementedError(ast.dump(stmt.op))
            state = {**state, name: new_val}

        elif isinstance(stmt, ast.Assert):
            cond = eval_expr(stmt.test, state)
            vc = Implies(And(*assumptions), cond) if assumptions else cond
            vcs.append((label_of(stmt), vc))
            assumptions = assumptions + [cond]

        elif _is_marker(stmt, 'assume'):
            arg = _marker_arg(stmt)
            if _is_false_literal(arg):
                return  # assume(False) marks a dead path; stop walking
            cond = eval_expr(arg, state)
            assumptions = assumptions + [cond]

        elif _is_marker(stmt, 'havoc'):
            target = _marker_arg(stmt)
            assert isinstance(target, ast.Name)
            state = {**state, target.id: _fresh(target.id)}

        elif isinstance(stmt, ast.If):
            guard = eval_expr(stmt.test, state)
            rest = stmts[idx + 1:]
            walk(list(stmt.body)   + rest, state, assumptions + [guard],      vcs, label_of)
            walk(list(stmt.orelse) + rest, state, assumptions + [Not(guard)], vcs, label_of)
            return

        elif (isinstance(stmt, ast.Expr)
              and isinstance(stmt.value, ast.Constant)
              and isinstance(stmt.value.value, str)):
            # Docstring (bare string expression). Skip silently.
            pass

        elif isinstance(stmt, ast.Return):
            # The function returns. Subsequent statements are
            # unreachable; the WP engine has no notion of "return
            # value" — verification is by assertions, not by the
            # returned expression — so we stop walking this path.
            return

        else:
            raise NotImplementedError(f'walk: {ast.dump(stmt)}')


def _is_false_literal(node):
    """True if `node` is the literal `False` (not the integer 0)."""
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, bool)
            and node.value is False)


# --- Top-level VCG ------------------------------------------------

def _tag_sufficiency(stmts):
    """Tag user asserts after a loop-cut entry assert as 'sufficiency'.

    Heuristic: at the top level of a function body, once we have seen
    the entry assert produced by `loop_cut`, any subsequent user assert
    (one without `_obligation_kind` already set) is the postcondition
    of that loop. Does not recurse into if-branches.
    """
    saw_entry = False
    for stmt in stmts:
        if isinstance(stmt, ast.Assert):
            kind = getattr(stmt, '_obligation_kind', None)
            if kind == 'entry':
                saw_entry = True
            elif kind is None and saw_entry:
                stmt._obligation_kind = 'sufficiency'


def vcgen(fn):
    """Generate per-obligation VCs for an annotated mini-IMP function.

    Returns a list of (label, vc_formula) records. Labels distinguish:
        - 'entry'        (loop's entry-invariant obligation)
        - 'preserved'    (loop's preservation obligation)
        - 'sufficiency'  (postcondition assert after a loop)
        - 'assert L'     (an explicit assert in the source at line L)
    """
    _reset_counter()
    src = fn if isinstance(fn, str) else inspect.getsource(fn)
    tree = ast.parse(src).body[0]
    if not isinstance(tree, ast.FunctionDef):
        raise ValueError('expected a function def')

    cut_body = loop_cut(tree.body)
    _tag_sufficiency(cut_body)

    state = {arg.arg: Int(arg.arg) for arg in tree.args.args}

    def label_of(stmt):
        kind = getattr(stmt, '_obligation_kind', None)
        if kind is not None:
            return kind
        return f'assert L{stmt.lineno}'

    vcs = []
    walk(cut_body, state, [], vcs, label_of)
    return vcs


# --- Discharge VCs to Z3 ------------------------------------------

def check(vcs):
    """Check each VC by asking whether its negation is satisfiable.

    Returns a list of (label, result, counterexample) records.
    """
    results = []
    for label, vc in vcs:
        s = Solver()
        s.add(Not(vc))
        r = s.check()
        if r == unsat:
            results.append((label, 'VALID', None))
        elif r == sat:
            results.append((label, 'NOT VALID', s.model()))
        else:
            results.append((label, 'UNKNOWN', None))
    return results


def _format_model(model):
    """Render a Z3 model as `name = value, ...` with fresh-var suffixes stripped.

    The havoc rule emits fresh variables `x__1, s__2, ...`; for display we
    collapse these back to their source-level names so the counterexample
    reads in terms of variables the student wrote.

    When the model has both a bare symbol (e.g. `x`) and a havoc'd symbol
    (e.g. `x__1`) for the same source name, prefer the havoc'd value,
    since that is the loop-body state where the assertion fired.
    """
    by_source = {}
    for decl in model.decls():
        full_name = decl.name()
        is_fresh = '__' in full_name
        source_name = full_name.split('__')[0] if is_fresh else full_name
        try:
            value = model[decl].as_long()
        except Exception:
            value = model[decl]
        # Prefer the havoc'd value over the bare symbol.
        if source_name not in by_source or is_fresh:
            by_source[source_name] = value
    parts = sorted(by_source.items())
    return ', '.join(f'{n} = {v}' for n, v in parts)


def verify(fn):
    """Convenience: run vcgen + check, print results."""
    vcs = vcgen(fn)
    results = check(vcs)
    width = max((len(r[0]) for r in results), default=12)
    for label, result, model in results:
        if result == 'VALID':
            print(f'  {label:<{width}} : VALID')
        elif result == 'NOT VALID':
            print(f'  {label:<{width}} : NOT VALID')
            print(f'    counterexample: {_format_model(model)}')
        else:
            print(f'  {label:<{width}} : UNKNOWN')
    return results
