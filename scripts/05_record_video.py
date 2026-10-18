#!/usr/bin/env python3
"""STEP 5 - one video per IK method (the deliverable asks for 3 min = 1 min per method, 'different trials').
  MUJOCO_GL=egl python scripts/05_record_video.py --seconds 60            # headless Linux (try osmesa if egl fails)
  python scripts/05_record_video.py --methods qp --seconds 60 --seeds 0,3,7,11,15   # specific trials
Episodes (different cube poses) are recorded back-to-back until ~--seconds of video exist; frames carry a text overlay
(method, episode, phase).  Output: results/videos/<method>.mp4  (+ videos/all_methods.mp4 if ffmpeg concat is possible)."""
import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pick_place.config import Config, load_design
from pick_place.stress import apply_stress
from pick_place.sim_env import SimEnv
from pick_place.scene import build_scene
from pick_place.planner import Planner
from pick_place.episode import run_episode
from pick_place.viz import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="heal", choices=["heal", "franka"])
    ap.add_argument("--methods", default="mink,dls,qp"); ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seeds", default=None, help="comma list; default 0,1,2,...")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--stress", default="none")
    ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = Config.load(overrides=a.set, robot=a.robot); design = load_design(cfg); cfg, design = apply_stress(cfg, design, a.stress)
    out = a.out or os.path.join(cfg.out_dir, "videos"); os.makedirs(out, exist_ok=True)
    env = SimEnv(cfg, design); psc = build_scene(cfg, design, for_planning=True)
    rec = Recorder(env, a.width, a.height, a.fps)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(range(1000))
    for method in a.methods.split(","):
        pl = Planner(cfg, psc, method); n_frames = 0; k = 0
        for ep, seed in enumerate(seeds):
            if n_frames / a.fps >= a.seconds: break
            l = run_episode(env, pl, cfg, design, ep, seed, method, recorder=rec)
            n_frames = len(rec.frames); k += 1
            print(f"[{method}] ep{ep} seed={seed} {'OK' if l.success else 'FAIL ' + l.failure_reason}  video so far {n_frames / a.fps:.0f}s")
        path = os.path.join(out, f"{method}.mp4"); n = rec.save_video(path); rec.next_t = 0.0
        print(f"saved {path} ({n / a.fps:.0f}s, {k} trials)")


if __name__ == "__main__":
    main()
