#!/usr/bin/env python3
"""Compare DLS / QP / mink on 5 scenarios.  python scripts/ik_benchmark.py [--n 100]"""
import os, sys, time, argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ik_lab import IKProblem, solve_dls, solve_qp, solve_mink

METHODS = {"DLS": solve_dls, "QP": solve_qp, "mink": solve_mink}


def scenarios(prob, rng, n):
    lo, hi = prob.qmin, prob.qmax
    c, h = (lo + hi) / 2, (hi - lo) / 2
    rq = lambda s=0.9: rng.uniform(c - h * s, c + h * s)                       # random config
    near = lambda qg, d: np.clip(qg + rng.uniform(-d, d, prob.n), lo, hi)       # start near a goal config
    S = {}
    gs = [rq() for _ in range(n)]
    S["1 near start (+-0.5 rad), wide limits"] = [(near(q, 0.5), *prob.fk(q)) for q in gs]
    gs = [rq() for _ in range(n)]
    for q in gs: q[2] = rng.uniform(-0.03, 0.03)                               # elbow straight -> singular
    S["2 goal near singularity"] = [(near(q, 0.5), *prob.fk(q)) for q in gs]
    S["3 tight limits, goal near limit"] = "TIGHT"
    S["4 unreachable target (1.5 m)"] = []
    for _ in range(n):
        p, R = prob.fk(rq()); S["4 unreachable target (1.5 m)"].append((rq(), p / np.linalg.norm(p) * 1.5, R))
    S["5 far random start (global IK)"] = [(rq(), *prob.fk(rq())) for _ in range(n)]
    return S


def run(prob, name, fn, cases):
    ok = it = viol = 0; t = 0; ep = []; er = []
    for c in cases:
        q0, p_t, R_t = c
        t0 = time.perf_counter(); r = fn(prob, q0, p_t, R_t); t += time.perf_counter() - t0
        ok += r.success; it += r.iters; ep.append(r.pos_err); er.append(r.rot_err)
        viol += not prob.within_limits(r.q, 1e-4)
    n = len(cases)
    return dict(success=100 * ok / n, iters=it / n, ms=1e3 * t / n, pos_mm=1e3 * np.median(ep), rot_deg=np.degrees(np.median(er)), viol=100 * viol / n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
    prob = IKProblem(); rng = np.random.default_rng(1)
    S = scenarios(prob, rng, a.n)
    print(f"{'scenario':40s}{'method':6s}{'succ%':>7s}{'iters':>7s}{'ms/solve':>10s}{'med pos err mm':>16s}{'med rot deg':>13s}{'limit viol%':>13s}")
    for sname, cases in S.items():
        if cases == "TIGHT":
            prob.set_limit_scale(0.35); r2 = np.random.default_rng(2)
            c, h = (prob.qmin + prob.qmax) / 2, (prob.qmax - prob.qmin) / 2
            cases = []
            for _ in range(a.n):
                qg = c + h * r2.choice([-1, 1], prob.n) * r2.uniform(0.85, 1.0, prob.n)   # goal pressed against limits
                cases.append((c + h * r2.uniform(-1, 1, prob.n), *prob.fk(qg)))
        for mname, fn in METHODS.items():
            m = run(prob, mname, fn, cases)
            print(f"{sname:40s}{mname:6s}{m['success']:7.0f}{m['iters']:7.0f}{m['ms']:10.1f}{m['pos_mm']:16.2f}{m['rot_deg']:13.2f}{m['viol']:13.0f}")
        prob.set_limit_scale(1.0)
        print()

if __name__ == "__main__":
    main()
