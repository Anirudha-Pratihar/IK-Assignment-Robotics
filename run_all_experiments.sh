#!/usr/bin/env bash
# Reproduces every experiment used in the report (25 episodes x 3 methods each, identical seeds per method).
# ~1 min per experiment on a laptop.  Run one line at a time while debugging.
# For failure screenshots add  --screenshots  (needs MUJOCO_GL=egl) to the 'stress' line.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
EXPS="main stress stress_retry4 lowcarry_noCA lowcarry_ca lowcarry_smart fast"
$PY scripts/03_run_batch.py --tag main           --n 25 --fresh                                                   # nominal: designed table
$PY scripts/03_run_batch.py --tag stress         --n 25 --stress all --fresh                                      # reach + tray + limits stress -> failures
$PY scripts/03_run_batch.py --tag stress_retry4  --n 25 --stress all --set grasp_yaw_retries=4 --fresh            # mitigation: try alternate grasp yaws
$PY scripts/03_run_batch.py --tag lowcarry_noCA  --n 25 --set lift_h=0.075 smart_lift=false use_collision_avoidance=false --fresh   # ablation: no avoidance
$PY scripts/03_run_batch.py --tag lowcarry_ca    --n 25 --set lift_h=0.075 smart_lift=false --fresh               # reactive avoidance only
$PY scripts/03_run_batch.py --tag lowcarry_smart --n 25 --set lift_h=0.075 --fresh                                # reactive + geometry-aware carry height
$PY scripts/03_run_batch.py --tag fast           --n 25 --set v_cart=0.9 w_cart=3.0 vmax=1.5 --fresh              # velocity-limit stress
for t in $EXPS; do $PY scripts/04_analyze_results.py --tag $t --extra main,stress,stress_retry4,lowcarry_noCA,lowcarry_ca,lowcarry_smart,fast > results/$t/analysis.txt; done
echo "done - see results/*/summary.md and report/"
