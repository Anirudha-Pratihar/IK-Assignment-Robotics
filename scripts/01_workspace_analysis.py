#!/usr/bin/env python3
"""STEP 1 - task + dexterous workspace, table design (L, W, H), tray + cube-spawn region, home configuration.

  python scripts/01_workspace_analysis.py                 # default resolution (~minutes)
  python scripts/01_workspace_analysis.py --res 0.1 --quick
Outputs:
  configs/table_design.json            <- consumed by every later script
  results/workspace.npz                <- raw masks
  report/figures/workspace_*.png, report/tables/table_design.tex
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pick_place.config import Config
from pick_place.robots import get_robot
from pick_place.workspace import compute_workspace, design_table, finish_design, _init, _G, _first_solution
from pick_place.scene import build_scene
from pick_place.geometry import topdown_R
from ik_lab import IKProblem


def _solve_home(prob, q0, p):
    from ik_lab import solve_dls
    r = solve_dls(prob, np.clip(q0, prob.qmin, prob.qmax), p, topdown_R(0.0), max_iter=200, lam=0.02, step=1.0)
    return r.q if r.success else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="heal", choices=["heal", "franka"])
    ap.add_argument("--use", default="auto", choices=["auto", "dex", "cubedex"], help="mask used for the table: auto = all-yaw dexterous, falling back to the cube-relevant yaw window")
    ap.add_argument("--min-area", type=float, default=0.05)
    ap.add_argument("--res", type=float, default=0.05, help="grid step in x/y [m]")
    ap.add_argument("--yaws", type=int, default=8, help="yaw samples over [-pi,pi)")
    ap.add_argument("--H", type=str, default=None, help="candidate table heights (default: per robot)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides key=value")
    a = ap.parse_args()
    cfg = Config.load(overrides=a.set, robot=a.robot); R = get_robot(a.robot); G = R.grid
    pre = "" if a.robot == "heal" else f"{a.robot}_"                      # HEAL file names stay unchanged
    Hs = [float(h) for h in a.H.split(",")] if a.H else list(G["H"])
    xs = np.arange(G["x"][0], G["x"][1] + 1e-9, a.res); ys = np.arange(G["y"][0], G["y"][1] + 1e-9, a.res)
    levels = sorted({round(H + o, 3) for H in Hs for o in (cfg.grasp_h, cfg.pregrasp_h, cfg.lift_h)})
    # collapse levels closer than 2.6 cm (they are covered by the nearest-level lookup in design_table)
    zs = []
    for z in levels:
        if not zs or z - zs[-1] > 0.026: zs.append(z)
    print(f"grid {len(xs)}x{len(ys)}  levels {zs}  yaws {a.yaws}")
    t0 = time.time()
    ws = compute_workspace(cfg, xs, ys, zs, n_yaw=a.yaws, workers=a.workers)
    print(f"workspace done in {time.time()-t0:.0f}s")
    os.makedirs(os.path.join(cfg.out_dir), exist_ok=True)
    np.savez_compressed(os.path.join(cfg.out_dir, "workspace.npz"), **ws)

    best, rows, basis = None, [], None
    for use in (["dex", "cubedex"] if a.use == "auto" else [a.use]):
        best, rows = design_table(cfg, ws, Hs, x_min_table=G["x_min_table"], use=use)
        if best is not None and best["area"] >= a.min_area: basis = use; break
    if best is not None and basis is None: basis = use
    if best is None:
        print("!! no dexterous rectangle found - try a finer --res, other --H, or use='cubedex'"); sys.exit(1)
    design = finish_design(cfg, best); design["basis"] = basis; design["robot"] = a.robot; design["base_xy"] = list(R.base_xy)
    print(f"table basis = {basis} (dex = all sampled yaws, cubedex = yaws within +-45 deg)")
    # home configuration: pre-grasp pose above table centre
    _init(json.dumps(cfg.__dict__, default=list))
    prob, rng = _G["prob"], _G["rng"]
    p_home = np.array([design["cx"], 0.0, design["H"] + cfg.pregrasp_h])
    q_nom = np.array(R.q_nom, float)
    cands = []
    for k in range(60):
        q0 = q_nom + (0 if k == 0 else rng.normal(0, 0.6, cfg.n_arm))
        r = _solve_home(prob, q0, p_home)
        if r is not None: cands.append(r)
    q_home = min(cands, key=lambda q: np.linalg.norm(q - q_nom)) if cands else None
    design["q_home"] = None if q_home is None else [float(v) for v in q_home]
    design["pregrasp_home_tcp"] = [float(v) for v in p_home]
    design["H_sweep"] = rows
    os.makedirs(os.path.dirname(cfg.design_json), exist_ok=True); json.dump(design, open(cfg.design_json, "w"), indent=2)
    print(json.dumps({k: v for k, v in design.items() if k != "H_sweep"}, indent=1))

    # ---------------- plots ----------------
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); fig_dir = os.path.join(ROOT, "report", "figures"); os.makedirs(fig_dir, exist_ok=True)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # (1) 3-D point clouds
    fig = plt.figure(figsize=(11, 4.6))
    for k, (title, key) in enumerate([("Task workspace (reachable TCP, gripper down)", "task"), ("Dexterous workspace (all yaws)", "dex")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for iz, z in enumerate(ws["zs"]):
            m = ws[key][iz]
            sc = ax.scatter(X[m], Y[m], np.full(m.sum(), z), c=np.full(m.sum(), z), vmin=zs[0], vmax=zs[-1], s=10, cmap="viridis")
        ax.scatter([R.base_xy[0]], [R.base_xy[1]], [0], c="r", marker="^", s=60, label="robot base"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(title, fontsize=9); ax.view_init(24, -60)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, pre + "workspace_3d.png"), dpi=170); plt.close(fig)
    # (2) slices at grasp level of the chosen table
    zi = int(np.argmin(np.abs(ws["zs"] - (design["H"] + cfg.grasp_h))))
    zl = int(np.argmin(np.abs(ws["zs"] - (design["H"] + cfg.lift_h))))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, iz, name in [(axs[0], zi, "grasp height"), (axs[1], zl, "lift height")]:
        img = np.zeros(ws["task"][iz].shape); img[ws["task"][iz]] = 1; img[ws["dex"][iz]] = 2
        ax.pcolormesh(ys - a.res / 2, xs - a.res / 2, img, cmap=plt.matplotlib.colors.ListedColormap(["#eeeeee", "#9ecae1", "#2171b5"]), vmin=0, vmax=2, shading="nearest")
        ax.add_patch(Rectangle((design["ymin"], design["xmin"]), design["W"], design["L"], fill=False, ec="k", lw=2, label="table"))
        tx, ty = design["tray_center"]; o = cfg.tray_inner[0] / 2 + cfg.tray_wall_t
        ax.add_patch(Rectangle((ty - o, tx - o), 2 * o, 2 * o, fc="#333333", alpha=0.8, label="tray"))
        sp = design["spawn"]; ax.add_patch(Rectangle((sp["ymin"], sp["xmin"]), sp["ymax"] - sp["ymin"], sp["xmax"] - sp["xmin"], fill=False, ec="r", ls="--", label="cube spawn"))
        ax.plot(R.base_xy[1], R.base_xy[0], "r^", ms=10); ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]"); ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_title(f"TCP z = {ws['zs'][iz]:.2f} m ({name}); light=unreachable, mid=task, dark=dexterous", fontsize=8); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, pre + "workspace_table_design.png"), dpi=170); plt.close(fig)
    # (3) height sweep
    fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar([f"{r['H']:.2f}" for r in rows], [r.get("area", 0) for r in rows], color="#2171b5")
    ax.set_xlabel("table height H [m]"); ax.set_ylabel("largest dexterous rectangle [m$^2$]"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, pre + "table_height_sweep.png"), dpi=170); plt.close(fig)
    # (4) latex table
    tex = ("\\begin{tabular}{llll}\\hline Parameter & Symbol & Value & Rationale\\\\\\hline\n"
           f"Table length & $L$ & {design['L']:.2f} m & fits the dexterous $x$-range at grasp and lift heights\\\\\n"
           f"Table width & $W$ & {design['W']:.2f} m & fits the dexterous $y$-range\\\\\n"
           f"Table height & $H$ & {design['H']:.2f} m & maximises dexterous area; vertical approach within joint limits\\\\\\hline\\end{{tabular}}\n")
    open(os.path.join(ROOT, "report", "tables", pre + "table_design.tex"), "w").write(tex)
    print("saved", cfg.design_json)


if __name__ == "__main__":
    main()
