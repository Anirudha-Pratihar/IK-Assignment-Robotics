#!/usr/bin/env bash
# Franka Panda 7-DOF.  Step 1-3 reproduce the submitted COMBINED run (25 episodes shared by the 3 methods: 9/8/8).
# Step 4 completes the study later (remaining 16/17/17 = 50 episodes -> 25 per method, then paired statistics).
# Run the steps one at a time.
set -e
cd "$(dirname "$0")"; PY=${PY:-python}

# 1. workspace + table design for the Franka (writes configs/franka/table_design.json, report/figures/franka_workspace_*.png)
$PY scripts/01_workspace_analysis.py --robot franka --res 0.06 --yaws 6

# 2. combined 25: each seed is run by exactly ONE method (round robin: mink 0,3,6.. / dls 1,4,7.. / qp 2,5,8..)
$PY scripts/03_run_batch.py --robot franka --tag main --n 25 --split --fresh

# 3. statistics (unpaired tests while seeds differ) + coverage report (how many episodes are left per method)
$PY scripts/04_analyze_results.py --robot franka --tag main --target 25
$PY scripts/07_make_report_numbers.py

# 4. LATER - complete the remaining 50 episodes (do NOT use --fresh or --split; finished episodes are kept, only missing seeds run)
#    $PY scripts/03_run_batch.py --robot franka --tag main --n 25
#    $PY scripts/04_analyze_results.py --robot franka --tag main --target 25     # now paired (McNemar / Wilcoxon)

# optional extras (each adds failures worth analysing; 25 seeds, combined or full):
#    $PY scripts/03_run_batch.py --robot franka --tag stress --n 25 --split --stress all --fresh
#    $PY scripts/03_run_batch.py --robot franka --tag fast   --n 25 --split --set v_cart=0.9 w_cart=3.0 vmax=1.5 --fresh
#    MUJOCO_GL=egl $PY scripts/05_record_video.py --robot franka --seconds 60
#    MUJOCO_GL=egl $PY scripts/08_scene_snapshot.py --robot franka
