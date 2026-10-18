"""Task / dexterous workspace computation by IK sampling + table-design search.

 task workspace      : TCP positions (x,y,z) reachable with the gripper pointing DOWN for at least one yaw
 dexterous workspace : positions reachable with the gripper pointing DOWN for EVERY sampled yaw in [-pi, pi)
 cube-dexterous      : same but only yaws in [-pi/4, pi/4] (a square cube has 4-fold symmetry, so this is what the task needs)

IK per position: random / neighbour-seeded DLS for the first yaw, then continuation over yaw (warm start from the previous
solution, trying both directions of wrist rotation).  Joint limits enforced by clipping, no collision model (kinematic only).
"""
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from .config import Config
from .scene import build_scene
from .robots import get_robot
from .geometry import topdown_R, wrap, largest_true_rectangle
from ik_lab import IKProblem, solve_dls

_G = {}


def _init(cfg_json):
    import json
    cfg = Config(); cfg.update(**json.loads(cfg_json))
    sc = build_scene(cfg, None)
    _G["prob"] = IKProblem(model=sc.model, ee_site="tcp", n_arm=cfg.n_arm)
    _G["cfg"] = cfg
    _G["rng"] = np.random.default_rng(1234)


def _solve(prob, q0, p, R, iters=120):
    r = solve_dls(prob, q0, p, R, max_iter=iters, lam=0.02, step=1.0)
    return r.q if r.success else None


def _first_solution(prob, rng, p, R, seeds, n_random=6):
    for q0 in seeds:
        q = _solve(prob, q0, p, R, 80)
        if q is not None: return q
    for _ in range(n_random):
        q = _solve(prob, rng.uniform(prob.qmin, prob.qmax), p, R, 100)
        if q is not None: return q
    return None


def _sweep_yaws(prob, p, q_ref, yaws, phi0):
    """continuation over yaw from the phi0 solution; returns bool array over yaws"""
    ok = np.zeros(len(yaws), bool)
    order = np.argsort(np.abs(wrap(yaws - phi0)))
    sols = {}
    for k in order:
        R = topdown_R(yaws[k]); best = None
        # warm starts: reference solution and the nearest already-solved yaw, with wrist (joint 6) rotated by +-dphi
        cands = [q_ref] + [sols[j] for j in sols][-2:]
        for qc in cands:
            for sgn in (1, -1):
                q0 = qc.copy(); q0[-1] = wrap(q0[-1] + sgn * wrap(yaws[k] - phi0))
                best = _solve(prob, q0, p, R, 60)
                if best is not None: break
            if best is not None: break
        if best is not None: ok[k] = True; sols[k] = best
    return ok


def _level_job(args):
    z, xs, ys, yaws = args
    prob, rng = _G["prob"], _G["rng"]
    nx, ny = len(xs), len(ys)
    task = np.zeros((nx, ny), bool); dex = np.zeros((nx, ny), bool); cubedex = np.zeros((nx, ny), bool)
    frac = np.zeros((nx, ny))
    q_seed = np.array(get_robot(_G["cfg"].robot).q_nom, float)
    seeds = [q_seed, np.zeros(len(q_seed))]
    near = np.abs(wrap(yaws)) <= np.pi / 4 + 1e-9
    last = None
    for ix, x in enumerate(xs):
        rng_y = range(ny) if ix % 2 == 0 else range(ny - 1, -1, -1)      # snake order -> good neighbour seeds
        for iy in rng_y:
            p = np.array([x, ys[iy], z])
            s = ([last] if last is not None else []) + seeds
            q = _first_solution(prob, rng, p, topdown_R(0.0), s)
            if q is None: continue
            last = q
            ok = _sweep_yaws(prob, p, q, yaws, 0.0)
            ok |= False
            task[ix, iy] = True                    # phi=0 solved => at least one yaw feasible
            frac[ix, iy] = ok.mean()
            dex[ix, iy] = ok.all()
            cubedex[ix, iy] = ok[near].all()
    return z, task, dex, cubedex, frac


def compute_workspace(cfg: Config, xs, ys, zs, n_yaw=12, workers=None, log=print):
    import json
    yaws = np.linspace(-np.pi, np.pi, n_yaw, endpoint=False)
    workers = workers or max(1, min(cpu_count(), len(zs)))
    t0 = time.time()
    jobs = [(z, xs, ys, yaws) for z in zs]
    with Pool(workers, initializer=_init, initargs=(json.dumps(cfg.__dict__, default=list),)) as pool:
        res = []
        for out in pool.imap_unordered(_level_job, jobs):
            res.append(out); log(f"  z={out[0]:.3f} done  ({len(res)}/{len(jobs)})  task {out[1].sum():4d}  dex {out[2].sum():4d}  [{time.time()-t0:.0f}s]")
    res.sort(key=lambda r: r[0])
    return dict(zs=np.array([r[0] for r in res]), task=np.stack([r[1] for r in res]), dex=np.stack([r[2] for r in res]),
                cubedex=np.stack([r[3] for r in res]), frac=np.stack([r[4] for r in res]), xs=xs, ys=ys, yaws=yaws)


def design_table(cfg: Config, ws, H_candidates, x_min_table=0.22, margin_cells=1, use="dex"):
    """For every candidate table height H, AND the masks at the grasp / lift TCP heights, find the largest all-True rectangle
    and return the best design (largest area) + a per-H summary list."""
    xs, ys, zs = ws["xs"], ws["ys"], ws["zs"]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    zi = lambda z: int(np.argmin(np.abs(zs - z)))
    rows, best = [], None
    for H in H_candidates:
        need = [H + cfg.grasp_h, H + cfg.pregrasp_h, H + cfg.lift_h]
        idx = [zi(z) for z in need if np.min(np.abs(zs - z)) < 0.026]
        if len(idx) < 2: continue
        mask = np.logical_and.reduce([ws[use][i] for i in idx])
        mask[xs < x_min_table, :] = False                      # keep the table clear of the robot base
        area, i0, i1, j0, j1 = largest_true_rectangle(mask)
        if area == 0: rows.append(dict(H=H, area=0.0)); continue
        i0 += margin_cells; i1 -= margin_cells; j0 += margin_cells; j1 -= margin_cells
        if i1 < i0 or j1 < j0: rows.append(dict(H=H, area=0.0)); continue
        L, W = (i1 - i0 + 1) * dx, (j1 - j0 + 1) * dy
        d = dict(H=float(H), area=float(L * W), L=float(L), W=float(W), xmin=float(xs[i0] - dx / 2), xmax=float(xs[i1] + dx / 2),
                 ymin=float(ys[j0] - dy / 2), ymax=float(ys[j1] + dy / 2), z_levels=[float(zs[i]) for i in idx])
        rows.append(d)
        if best is None or d["area"] > best["area"]: best = d
    return best, rows


def finish_design(cfg: Config, best: dict) -> dict:
    """tray + spawn region inside the chosen rectangle"""
    L, W, H = best["L"], best["W"], best["H"]
    cx, cy = (best["xmin"] + best["xmax"]) / 2, (best["ymin"] + best["ymax"]) / 2
    outer = cfg.tray_inner[1] + 2 * cfg.tray_wall_t
    tray_y = best["ymax"] - outer / 2 - 0.02
    tray_c = [cx, tray_y]
    m = 0.04
    spawn = dict(xmin=best["xmin"] + m, xmax=best["xmax"] - m, ymin=best["ymin"] + m, ymax=tray_y - outer / 2 - cfg.spawn_tray_margin)
    return dict(H=H, L=L, W=W, cx=cx, cy=cy, xmin=best["xmin"], xmax=best["xmax"], ymin=best["ymin"], ymax=best["ymax"],
                tray_center=tray_c, spawn=spawn, area=best["area"])
