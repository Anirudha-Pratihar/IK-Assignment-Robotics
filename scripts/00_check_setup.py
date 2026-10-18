#!/usr/bin/env python3
"""STEP 0 - verify the environment before running anything:  python scripts/00_check_setup.py"""
import os, sys, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ok = True
def chk(name, cond, msg=""):
    global ok; ok &= bool(cond); print(f"[{'OK' if cond else '!!'}] {name} {msg}")
for mod in ("numpy", "scipy", "mujoco", "mink", "qpsolvers", "daqp", "matplotlib", "pandas", "imageio"):
    try: m = importlib.import_module(mod); chk(mod, True, getattr(m, "__version__", ""))
    except Exception as e: chk(mod, False, f"-> pip install {mod}  ({e})")
import qpsolvers; chk("QP solver 'daqp' available", "daqp" in qpsolvers.available_solvers, str(qpsolvers.available_solvers))
from pick_place.config import Config, load_design
from pick_place.gripper import robotiq_available
cfg = Config()
chk("HEAL xml", os.path.isfile(cfg.arm_xml), cfg.arm_xml)
chk("HEAL meshes", os.path.isfile(os.path.join(os.path.dirname(cfg.arm_xml), "heal_meshes", "link_1.STL")))
rq = robotiq_available(cfg); print(f"[{'OK' if rq else '--'}] Robotiq 2F-85 assets {'found' if rq else 'NOT found -> built-in fallback gripper will be used (same interface)'}")
chk("table design", os.path.isfile(cfg.design_json), "run scripts/01_workspace_analysis.py if missing")
try:
    from pick_place.sim_env import SimEnv
    env = SimEnv(cfg, load_design(cfg)); chk("scene compiles", True, f"gripper={env.sc.gripper.kind} nq={env.model.nq}")
except Exception as e: chk("scene compiles", False, str(e))
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>"); r = mujoco.Renderer(m, 32, 32)
    chk("offscreen rendering (screenshots / videos)", True, os.environ.get("MUJOCO_GL", "default backend"))
except Exception as e: chk("offscreen rendering", False, f"set MUJOCO_GL=egl (or osmesa/glfw). screenshots/videos disabled until fixed. ({str(e)[:80]})")
print("\nALL GOOD" if ok else "\nfix the [!!] items above")
