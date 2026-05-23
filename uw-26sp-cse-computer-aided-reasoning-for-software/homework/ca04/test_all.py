"""Run your ca04 annotations through the SE and WP engines.

Usage:  python test_all.py

Each problem runs in a child process with a hard 5-second budget. If
the child doesn't finish in time, it's killed and the problem is
marked TIMEOUT.

Part A problems are PASS if the L07 SE engine finds at least one
assertion violation. Part B problems are PASS if the L08 WP engine
reports every VC as VALID.
"""
import importlib
import multiprocessing as mp
import time


BUDGET_S = 5.0
GRACE_S = 0.5  # small wall-clock slack on top of BUDGET_S before terminating

PART_A = ['partA_clamp_sub', 'partA_safe_mul', 'partA_count_groups']
PART_B = ['partB_make_change', 'partB_isqrt_newton']


def _part_a_worker(q, mod_name):
    from engine.se_engine import check_assertions
    m = importlib.import_module(mod_name)
    fn_name = mod_name.removeprefix('partA_')
    fn = getattr(m, fn_name)
    t0 = time.time()
    violations = check_assertions(
        fn, max_depth=m.MAX_DEPTH, bitwidth=m.BITWIDTH, timeout_ms=5000,
    )
    q.put({
        'time': time.time() - t0,
        'violations': len(violations),
        'depth': m.MAX_DEPTH,
        'bw': m.BITWIDTH,
    })


def _part_b_worker(q, mod_name):
    from engine.wp_engine import vcgen, check
    m = importlib.import_module(mod_name)
    fn_name = mod_name.removeprefix('partB_')
    fn = getattr(m, fn_name)
    t0 = time.time()
    results = check(vcgen(fn))
    bad = [(label, r) for (label, r, _model) in results if r != 'VALID']
    q.put({
        'time': time.time() - t0,
        'vcs': len(results),
        'bad': bad,
    })


def _run_in_child(worker, mod_name):
    """Spawn `worker(q, mod_name)`, kill it after BUDGET_S + GRACE_S."""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(q, mod_name))
    p.start()
    p.join(BUDGET_S + GRACE_S)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return 'ERROR'


def run_part_a(mod_name):
    fn_name = mod_name.removeprefix('partA_')
    r = _run_in_child(_part_a_worker, mod_name)
    if r is None:
        print(f'  [TIMEOUT] {fn_name}: exceeded {BUDGET_S}s')
        return False
    if r == 'ERROR':
        print(f'  [ERROR] {fn_name}: child crashed before reporting')
        return False
    ok = r['violations'] > 0 and r['time'] < BUDGET_S
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {fn_name}: '
          f'depth={r["depth"]} bw={r["bw"]} '
          f'time={r["time"]:.2f}s violations={r["violations"]}')
    return ok


def run_part_b(mod_name):
    fn_name = mod_name.removeprefix('partB_')
    r = _run_in_child(_part_b_worker, mod_name)
    if r is None:
        print(f'  [TIMEOUT] {fn_name}: exceeded {BUDGET_S}s')
        return False
    if r == 'ERROR':
        print(f'  [ERROR] {fn_name}: child crashed before reporting')
        return False
    ok = not r['bad'] and r['time'] < BUDGET_S
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {fn_name}: time={r["time"]:.2f}s '
          f'vcs={r["vcs"]} not_valid={len(r["bad"])}')
    for label, result in r['bad']:
        print(f'      {label}: {result}')
    return ok


def main():
    print('Part A (SE engine):')
    a = [run_part_a(m) for m in PART_A]
    print()
    print('Part B (WP engine):')
    b = [run_part_b(m) for m in PART_B]
    passed = sum(a) + sum(b)
    total = len(PART_A) + len(PART_B)
    print()
    print(f'{passed}/{total} problems passed')


if __name__ == '__main__':
    main()
