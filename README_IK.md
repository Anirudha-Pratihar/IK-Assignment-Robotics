# Inverse-kinematics pick-and-place lab (MuJoCo, HEAL 6-DOF)

Randomized pick-and-place of a cube into a tray, driven by inverse kinematics, with **three IK methods compared on identical episode seeds**:

| key | method | limits | collision handling |
|---|---|---|---|
| `mink` | off-the-shelf [mink](https://github.com/kevinzakka/mink) `FrameTask` (baseline) | `ConfigurationLimit`, `VelocityLimit` | `CollisionAvoidanceLimit` |
| `dls` | closed-loop damped least squares (adaptive λ) | position clipping only | potential-field repulsion |
| `qp` | closed-loop QP (DAQP via `qpsolvers`) | joint-position + velocity bounds in the QP | signed-distance inequality rows |

It implements Tutorial 4 end to end: workspace analysis -> table design -> randomized episodes -> logs -> failure analysis -> method comparison -> report/video.

## 1. Install
```bash
pip install -r requirements.txt
python scripts/00_check_setup.py          # verifies packages, solver, assets, scene compile, offscreen GL
```
Headless Linux rendering (screenshots/videos): `export MUJOCO_GL=egl` (or `osmesa`). macOS viewer: use `mjpython` instead of `python`.

## 2. Workflow (run in order)
| step | command | produces |
|---|---|---|
| 0 | `python scripts/00_check_setup.py` | environment report |
| 1 | `python scripts/01_workspace_analysis.py` | `configs/table_design.json` (L, W, H, tray, spawn region, home q), `report/figures/workspace_*.png`, `report/tables/table_design.tex` |
| 2 | `python scripts/02_run_episode_viewer.py --method qp --seed 3` | interactive single-episode debugging |
| 3 | `python scripts/03_run_batch.py --tag main --n 25` | `results/main/episodes_{mink,dls,qp}.jsonl` (+ `--traces`, `--screenshots`) |
| 4 | `python scripts/04_analyze_results.py --tag main` | `summary.csv/md`, paired tests, `failures.csv`, `report/figures/main_*.png`, `report/tables/main_*.tex` |
| 5 | `python scripts/05_record_video.py --seconds 60` | `results/videos/{mink,dls,qp}.mp4` (1 min per method = the 3-min deliverable) |
| 6 | `python scripts/06_failure_screenshots.py --tag stress --method qp` | `report/figures/fail_*.png` (re-runs logged failures exactly) |
| 7 | `python scripts/07_make_report_numbers.py` then `cd report && latexmk -pdf main.tex` | `report/numbers.tex` (macros used by the prose), `report/main.pdf` |

`./run_all_experiments.sh` reproduces every experiment in the report (about 1 min each). Run its lines one by one while debugging.

## 3. Experiments
| tag | what | how |
|---|---|---|
| `main` | nominal, designed table | `--tag main` |
| `stress` | reach (table x2.2), cubes next to tray, joint ranges 55 % | `--stress all` |
| `stress_retry4` | mitigation: alternate grasp yaws | `--stress all --set grasp_yaw_retries=4` |
| `lowcarry_noCA` / `_ca` / `_smart` | collision-avoidance ablation at a 7.5 cm carry height | `--set lift_h=0.075 smart_lift=false use_collision_avoidance=false` / `... smart_lift=false` / `--set lift_h=0.075` |
| `fast` | velocity-limit stress | `--set v_cart=0.9 w_cart=3.0 vmax=1.5` |

Reference results (built-in fallback gripper, 25 episodes, Mink / DLS / QP success): main 25 / 25 / 25, stress 16 / 15 / 16, stress_retry4 17 / 16 / 17,
lowcarry_noCA 0 / 0 / 0, lowcarry_ca 0 / 3 / 0, lowcarry_smart 25 / 25 / 25, fast 25 / 24 / 25. **Re-run with your final gripper; numbers will change.**

## 4. Code map
```
pick_place/
  config.py       every tunable number (SI units); override with --set key=value or --config file.json
  gripper.py      Robotiq 2F-85 if robot_descriptions/robotiq_2f85_v4/assets/*.stl exist, else a primitive parallel-jaw fallback (same interface)
  scene.py        MjSpec scene builder: arm (servo actuators) + gripper + table + tray + cube; plan-model variant adds a carried-cube geom
  sim_env.py      physics: position servo + gravity/damping feed-forward, gripper, contact classification, clearance
  collision.py    geom groups per phase, signed-distance queries, QP rows, DLS repulsion
  ik_solvers.py   DLSStepper / QPStepper / MinkStepper  (one interface: dq = step(q, p_target, R_target))
  planner.py      straight-line min-jerk Cartesian segments tracked by a stepper; per-waypoint convergence solve
  episode.py      state machine, failure taxonomy, logging
  workspace.py    task / dexterous workspace + table design search
  metrics.py viz.py stress.py geometry.py
ik_lab/           earlier standalone IK code (Jacobian via mj_jacSite, pose error); reused by the steppers
scripts/          numbered entry points (above)
report/           main.tex (<=5 pages), figures/, tables/, numbers.tex (auto)
```

## 5. Key design choices (read before changing things)
- **Execution is plan-then-replay.** Each phase is planned on a private copy of the model (kinematic, 50 Hz), then replayed by a joint position servo in the physics simulation. Approach+descend are dry-run first so alternative yaws can be tried.
- **Same gain for all methods:** each tick closes a fraction `ik_gain` (0.8) of the pose error. Mink's `task.gain` is a per-step fraction (1.0 = close it in one step); the QP uses `v = gain*e/dt`.
- **Metrics.** `ik_iters` = iterations a fixed-target solve needs per waypoint (solver cost); `track_ticks` = reference-tracking ticks (set by segment timing, not by the solver). `plan_time_s` is wall time inside the solvers. `limit_events` / `vel_events` count ticks where the *raw* update left the joint range / exceeded `vmax` (DLS clips afterwards, so look at events, not final violations).
- **Failure labels** (`ik_limits` vs `ik_infeasible`) are assigned from the physical state: a joint sitting on its bound at stall = limits.
- **Geom ids differ between the sim and plan models** (the plan model adds `carried_cube`). Never reuse ids across models; build groups by name.
- **Collision keep-out distances are per group** (`collision.py`): pads-vs-table uses 6 mm because the grasp intentionally gets within ~13 mm of the table.
- **Smart carry height** (`smart_lift`): lift to at least `tray_wall_h + carry_clearance + grasp_h`. Reactive constraints alone stall in front of the wall (see `lowcarry_ca`).

## 6. Using the real Robotiq 2F-85
Copy the meshes to `robot_descriptions/robotiq_2f85_v4/assets/` (the zip we received had none). `gripper_kind=auto` then selects it. Check, in this order:
1. `python scripts/00_check_setup.py` says `Robotiq ... found`.
2. TCP offset `tcp_z_robotiq` (0.13 m, pad centre read from `2f85.xml`) and that jaws close along tool x.
3. Actuator `gripper/fingers_actuator` (ctrl 0 open .. 255 closed) and pad geoms `gripper/{left,right}_pad{1,2}`.
4. Servo gains (`servo_kp`, `servo_kv`) - the Robotiq is heavier than the fallback gripper.
5. `SimEnv.finger_travel()` assumes joints named `*driver*`/`*slide*`; it is only a diagnostic.
The Robotiq path was only built (names, attach, actuator), never run with real meshes - expect to debug grasp physics.

## 7. Troubleshooting
- `ValueError ... actuator name(s) missing`: wrong gripper/actuator naming.
- Screenshots fail with a GL error: set `MUJOCO_GL=egl` (or `osmesa`); the offscreen buffer is set to 1920x1080 in `scene.py`.
- QP returns `None` often: raise `d_min` tolerance (collision rows infeasible) or check `qp_fail` in the logs; try `--set solver=quadprog`.
- Cubes never grasped: check `grasp_h` (pads must clear the table, d_min pads-table = 6 mm) and cube friction.
- Workspace analysis is slow: increase `--res`, reduce `--yaws`; it parallelises over height levels.
- A background job that stalls on one machine: run the batch in the foreground; episodes are logged incrementally and `--tag` resumes.

## 8. Deliverables checklist (tutorial section 15)
- [x] code + this README
- [x] workspace plots and table-design justification (`report/figures/workspace_*`, `table_design.tex`)
- [x] per-episode logs and summary tables (`results/<tag>/`, `report/tables/`)
- [x] failure analysis with screenshots (`06_failure_screenshots.py`)
- [x] comparative analysis with plots and paired tests
- [ ] 3-min video (1 min per method): `05_record_video.py` (needs a GPU/EGL machine; not produced in the reference run)
- [ ] fill the `\todo{}` items in `report/main.tex` (name, repo link, final gripper, your observations)
