#!/usr/bin/env python3
"""STEP 2 - debug ONE episode in the interactive MuJoCo viewer (macOS: run with `mjpython`).
  python scripts/02_run_episode_viewer.py --method qp --seed 3
  python scripts/02_run_episode_viewer.py --method dls --seed 1 --stress limits --slow 2
The episode is executed live; planned TCP waypoints are NOT drawn, but the TCP (green) and cube are visible.  When it ends the
result is printed and the viewer stays open (close the window to exit)."""
import os, sys, time, argparse
import numpy as np, mujoco
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pick_place.config import Config, load_design
from pick_place.stress import apply_stress
from pick_place.sim_env import SimEnv
from pick_place.scene import build_scene
from pick_place.planner import Planner
from pick_place.episode import run_episode


class LiveRecorder:
    """duck-typed Recorder: syncs the viewer every captured tick (paces to real time x slow)"""
    def __init__(self, viewer, slow, dt): self.v, self.slow, self.dt, self.label = viewer, slow, dt, ""; self.frames = []
    def maybe_capture(self, env):
        self.v.sync(); time.sleep(self.dt * self.slow)
    def save_png(self, *a, **k): pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--robot", default="heal", choices=["heal", "franka"]); ap.add_argument("--method", default="qp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stress", default="none"); ap.add_argument("--slow", type=float, default=1.0); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = Config.load(overrides=a.set, robot=a.robot); design = load_design(cfg); cfg, design = apply_stress(cfg, design, a.stress)
    env = SimEnv(cfg, design); psc = build_scene(cfg, design, for_planning=True); pl = Planner(cfg, psc, a.method)
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [design["cx"] * 0.8, design["cy"], design["H"] + 0.1]; v.cam.distance = 1.2; v.cam.azimuth = 150; v.cam.elevation = -25
        live = LiveRecorder(v, a.slow, cfg.ik_dt)
        l = run_episode(env, pl, cfg, design, 0, a.seed, a.method, recorder=live)
        print("SUCCESS" if l.success else f"FAIL: {l.failure_reason} in {l.failure_phase}  ({l.failure_detail})")
        print({k: v for k, v in l.to_dict().items() if k not in ("ik_iters_by_phase",)})
        while v.is_running(): v.sync(); time.sleep(0.05)


if __name__ == "__main__":
    main()
