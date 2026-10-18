#!/usr/bin/env python3
"""Render a PNG of the initial scene (robot, table, tray, cube) - handy as a report figure.
  MUJOCO_GL=egl python scripts/08_scene_snapshot.py --robot franka        ->  report/figures/<robot>_scene.png"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pick_place.config import Config, load_design
from pick_place.sim_env import SimEnv
from pick_place.episode import sample_cube
from pick_place.viz import Recorder

ap = argparse.ArgumentParser(); ap.add_argument("--robot", default="heal", choices=["heal", "franka"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=450); a = ap.parse_args()
cfg = Config.load(robot=a.robot); design = load_design(cfg); env = SimEnv(cfg, design)
x, y, yaw = sample_cube(design, a.seed, cfg); env.reset(np.array(design["q_home"]), (x, y), yaw)
rec = Recorder(env, a.width, a.height)
if a.robot == "franka": rec.cam.lookat[:] = [0.1, design["cy"], design["H"] + 0.1]; rec.cam.distance = 1.6
out = os.path.join(os.path.dirname(__file__), "..", "report", "figures", f"{a.robot}_scene.png"); rec.save_png(out, f"{cfg.robot} home pose"); print("saved", os.path.abspath(out))
