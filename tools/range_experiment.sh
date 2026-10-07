#!/usr/bin/env bash
# Does putting the range reward inside arm's reach make the policy punch?
#
# tools/gate_calibration.py measured the geometry: a scripted jab connects 0 times
# at 1.15 m, 26 times at 0.9 m and 53 times at 0.7 m. The range term's optimum was
# at 0.9 m, so the reward was telling both boxers to hold a distance from which
# hitting is rare. This retrains the frozen arm with the range target at 0.7 m.
#
# Prediction, written before running: gated punches per round should rise from the
# 0.009-0.304 band the current policies sit in. The risk is the opposite side of
# the same coin -- closing distance is also what a lunge does -- so down_red is
# read alongside it rather than hoped away. If punches rise and falls also rise,
# the range term is not the blocker and the blocker is stepping.
#
#   wsl bash tools/range_experiment.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

while pgrep -f "boxer.train|boxer.arena|boxer.rollout" > /dev/null; do sleep 20; done

mkdir -p runs/range
"$PY" -u -m boxer.train --mode frozen --opponent passive \
  --target-range 0.7 --spawn-distance 1.3 \
  --total-steps 2621440 --num-evals 5 \
  --worlds 256 --batch-size 4096 --num-minibatches 8 \
  --output runs/range > runs/range/train.log 2>&1
echo "range experiment exited $?"
