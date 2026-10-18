#!/usr/bin/env bash
# HEAL 6-DOF: all experiments (25 episodes x 3 methods each, identical seeds) + analysis.  ~10 min.
cd "$(dirname "$0")" && exec ./run_all_experiments.sh
