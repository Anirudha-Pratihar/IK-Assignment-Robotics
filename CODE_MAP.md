# Code map: what belongs to HEAL, what belongs to Franka, what is shared

One rule: **robot-specific names live only in `pick_place/robots.py`** (+ per-robot overrides in `config.py` via `Config.load(robot=...)`).
Everything else is shared and takes `--robot heal|franka` (default `heal`, so all old HEAL commands behave exactly as before).
Checked: HEAL results are identical to the saved logs after the Franka changes (success, iterations, clearances, placement error: max difference < 1e-6).

## A. SHARED (robot-agnostic code, used by both)
| file | role |
|---|---|
| `pick_place/config.py` | all tunables; `Config.load(robot=...)` applies the robot's overrides |
| `pick_place/scene.py` | builds arm + gripper + table + tray + cube (sim and plan models) |
| `pick_place/sim_env.py` | physics, servo, contact classification |
| `pick_place/collision.py` | collision groups, distance constraints, DLS repulsion |
| `pick_place/ik_solvers.py` | DLS / QP / Mink steppers (6- and 7-joint, null-space posture for redundant arms) |
| `pick_place/planner.py`, `episode.py` | segment planner, pick-and-place state machine, failure taxonomy |
| `pick_place/workspace.py`, `geometry.py`, `metrics.py`, `viz.py`, `stress.py` | workspace/table design, helpers, logs, rendering, stress presets |
| `ik_lab/` | Jacobian, pose error, standalone DLS/QP/mink solvers (used by the workspace analysis) |
| `scripts/00..08_*.py` | all entry points; every one accepts `--robot` (08 = scene snapshot) |
| `scripts/07_make_report_numbers.py`, `report/main.tex` | report (HEAL = main study, Franka = combined set) |

## B. HEAL 6-DOF only
| item | location |
|---|---|
| robot spec | `pick_place/robots.py` -> `ROBOTS["heal"]` |
| model | `robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml`, `heal_meshes/` |
| gripper | `pick_place/gripper.py` (Robotiq 2F-85 via `robotiq_2f85_v4/`, or the built-in fallback; attached at site `right_center`) |
| table design | `configs/table_design.json` |
| results (25 per method, 7 experiments) | `results/<tag>/` (`main`, `stress`, `stress_retry4`, `lowcarry_*`, `fast`) |
| report files | `report/figures/` and `report/tables/` **without** a `franka_` prefix |
| run it | `./run_heal.sh` (= `run_all_experiments.sh`) |

## C. Franka Panda 7-DOF only
| item | location |
|---|---|
| robot spec | `pick_place/robots.py` -> `ROBOTS["franka"]` (joints `joint1..7`, actuators `actuator1..7`, hand body `hand`, base at (-0.3, 0, 0.8)) |
| model | `robot_descriptions/franka/` (`scene_fk.xml` includes `panda.xml`; separate zip `franka_assets.zip`) |
| gripper | built-in hand, actuator `actuator8`, **ctrl 255 = open, 0 = closed** (opposite of Robotiq); pads = colliding geoms of `left_finger/right_finger`; TCP 0.103 m along hand z |
| Franka-only config overrides | `grasp_h=0.025`, `release_h=0.050`, `posture_gain=0.5` (null space), `spawn_tray_margin=0.14` (wide hand), `pads_tray_band=0.004`, `franka_grip_kp=1000` |
| table design | `configs/franka/table_design.json` (H 0.75 m, 0.30 x 0.72 m) |
| results (25 episodes COMBINED: 9 / 8 / 8) | `results/franka/main/` |
| report files | `report/figures/franka_*.png`, `report/tables/franka_*.tex` |
| run it | `./run_franka.sh` |

## D. First lab (forward kinematics, separate folder `ITR_mujoco_fk_lab/`)
`scripts/fk_lab/` supports `panda7`, `panda6` (joint 7 locked) and `heal6`: `python run_fk_sim.py --robot panda7|panda6|heal6`.

## E. Output locations by robot
| | HEAL | Franka |
|---|---|---|
| logs | `results/<tag>/episodes_<method>.jsonl` | `results/franka/<tag>/episodes_<method>.jsonl` |
| figures | `report/figures/<tag>_*.png` | `report/figures/franka_<tag>_*.png` |
| tables | `report/tables/<tag>_*.tex` | `report/tables/franka_<tag>_*.tex` |
| videos | `results/videos/` | `results/franka/videos/` |
