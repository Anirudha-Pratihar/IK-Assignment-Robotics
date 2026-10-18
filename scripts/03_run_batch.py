#!/usr/bin/env python3
"""STEP 3 - batch demonstrations: N episodes x methods with IDENTICAL seeds, per-episode logs (JSONL), traces, failure screenshots.

  python scripts/03_run_batch.py                               # 25 episodes x (mink, dls, qp)
  python scripts/03_run_batch.py --n 100 --methods qp,mink --tag big
  python scripts/03_run_batch.py --stress all --tag stress --screenshots     # provoke failures for the failure analysis
  python scripts/03_run_batch.py --set grasp_yaw_retries=4 --tag mitigated   # mitigation experiment (alternate grasp yaw)
  python scripts/03_run_batch.py --set use_collision_avoidance=false --tag no_ca   # ablation
Outputs in results/<tag>/ : episodes_<method>.jsonl, traces/*.npz, screenshots/*.png, config.json, design.json
Resume: re-running with the same --tag skips episodes already logged (use --fresh to restart)."""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pick_place.config import Config, load_design
from pick_place.stress import apply_stress
from pick_place.sim_env import SimEnv
from pick_place.scene import build_scene
from pick_place.planner import Planner
from pick_place.episode import run_episode
from pick_place.metrics import save_jsonl, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="heal", choices=["heal", "franka"])
    ap.add_argument("--split", action="store_true", help="COMBINED mode: each seed is run by ONE method only (round-robin over --methods), e.g. 25 seeds -> 9/8/8 episodes. Re-run without --split (same --tag) later to complete every method on every seed.")
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="mink,dls,qp"); ap.add_argument("--tag", default="main")
    ap.add_argument("--stress", default="none", choices=["none", "reach", "tray", "limits", "all"])
    ap.add_argument("--screenshots", action="store_true", help="save a PNG at the moment of every failure (needs MUJOCO_GL=egl|glfw|osmesa)")
    ap.add_argument("--shot-width", type=int, default=960); ap.add_argument("--shot-height", type=int, default=540)
    ap.add_argument("--traces", action="store_true"); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--config", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = Config.load(a.config, a.set, robot=a.robot); design = load_design(cfg)
    cfg, design = apply_stress(cfg, design, a.stress)
    out = os.path.join(cfg.out_dir, a.tag); os.makedirs(out, exist_ok=True)
    json.dump(json.loads(cfg.to_json()), open(os.path.join(out, "config.json"), "w"), indent=2); json.dump(design, open(os.path.join(out, "design.json"), "w"), indent=2)
    env = SimEnv(cfg, design); psc = build_scene(cfg, design, for_planning=True)
    print(f"robot={cfg.robot}  gripper={env.sc.gripper.kind}  split={a.split}  stress={a.stress}  episodes={a.n}  seeds={a.seed}..{a.seed + a.n - 1}  out={out}")
    rec = None
    if a.screenshots:
        try:
            from pick_place.viz import Recorder; rec = Recorder(env, a.shot_width, a.shot_height)
        except Exception as e: print("!! screenshots disabled (no GL?):", e)

    methods = a.methods.split(",")
    for mi, method in enumerate(methods):
        path = os.path.join(out, f"episodes_{method}.jsonl")
        logs = [] if a.fresh or not os.path.exists(path) else load_jsonl(path)
        done = {l["episode"] for l in logs}
        pl = Planner(cfg, psc, method); t0 = time.time()
        for ep in range(a.n):
            if ep in done: continue
            if a.split and ep % len(methods) != mi: continue          # round-robin seed assignment
            l = run_episode(env, pl, cfg, design, ep, a.seed + ep, method, recorder=rec,
                            shot_dir=os.path.join(out, "screenshots") if rec else None,
                            trace_dir=os.path.join(out, "traces") if a.traces else None)
            logs.append(l.to_dict()); save_jsonl(logs, path)
            print(f"[{method:4s}] ep{ep:03d} seed={a.seed+ep:4d} cube=({l.cube_x:.3f},{l.cube_y:.3f},{np.degrees(l.cube_yaw):6.1f}deg) "
                  + ("OK  " if l.success else f"FAIL {l.failure_reason}@{l.failure_phase}") + f"  it={l.ik_iters:4d} plan={l.plan_time_s*1e3:6.1f}ms clr={l.min_clear_robot_env*1e3:5.1f}mm")
        S = [l for l in logs]; k = sum(l["success"] for l in S)
        if a.split: print(f"   [{method}] seeds covered: {sorted(l['seed'] for l in S)}")
        print(f"== {method}: {k}/{len(S)} success ({100*k/max(1,len(S)):.0f}%)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
