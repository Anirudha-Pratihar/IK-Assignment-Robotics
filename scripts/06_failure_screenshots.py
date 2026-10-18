#!/usr/bin/env python3
"""STEP 6 - re-run logged failures and save a screenshot at the moment of failure (for the failure-analysis section).
  MUJOCO_GL=egl python scripts/06_failure_screenshots.py --tag stress --method qp --max 4
Reads results/<tag>/failures.csv + config.json + design.json (written by 03/04) so the episode is reproduced exactly.
Output: report/figures/fail_<tag>_<method>_ep<k>_<reason>_<phase>.png  (one per distinct reason/phase unless --all)"""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pick_place.config import Config
from pick_place.sim_env import SimEnv
from pick_place.scene import build_scene
from pick_place.planner import Planner
from pick_place.episode import run_episode
from pick_place.viz import Recorder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="heal", choices=["heal", "franka"]); ap.add_argument("--tag", default="stress"); ap.add_argument("--method", default="qp"); ap.add_argument("--max", type=int, default=4)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=360)
    a = ap.parse_args()
    d = os.path.join(ROOT, "results", a.tag) if a.robot == "heal" else os.path.join(ROOT, "results", a.robot, a.tag)
    pre = "" if a.robot == "heal" else a.robot + "_"
    cfg = Config(); cfg.update(**json.load(open(os.path.join(d, "config.json")))); design = json.load(open(os.path.join(d, "design.json")))
    cfg.tray_inner = tuple(cfg.tray_inner)
    bad = pd.read_csv(os.path.join(d, "failures.csv")); bad = bad[bad.method == a.method]
    if not a.all: bad = bad.drop_duplicates(["failure_reason", "failure_phase"])
    bad = bad.head(a.max)
    env = SimEnv(cfg, design); psc = build_scene(cfg, design, for_planning=True); pl = Planner(cfg, psc, a.method)
    rec = Recorder(env, a.width, a.height); out = os.path.join(ROOT, "report", "figures")
    for _, r in bad.iterrows():
        l = run_episode(env, pl, cfg, design, int(r.episode), int(r.seed), a.method, recorder=rec, shot_dir=out)
        src = os.path.join(out, f"{a.method}_ep{int(r.episode):03d}_{l.failure_reason}_{l.failure_phase}.png")
        dst = os.path.join(out, f"fail_{pre}{a.tag}_{a.method}_ep{int(r.episode)}_{l.failure_reason}_{l.failure_phase}.png")
        if os.path.exists(src): os.replace(src, dst); print("saved", dst, "|", l.failure_detail)


if __name__ == "__main__":
    main()
