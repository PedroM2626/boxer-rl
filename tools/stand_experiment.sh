#!/usr/bin/env bash
# Can a boxer learn to stand when it cannot reach anybody?
#
# A zero-action boxer topples in 1.08 s, and that is the expected behaviour of a
# position-controlled humanoid whose PD error starts at exactly zero -- so the
# question is not whether the stance is broken, it is whether *standing* is
# learnable at this scale before adding an opponent. The opponent is pushed to
# 2.2 m and the range target set to the same distance, so the policy gets no
# reward for closing in and no contact is available: balance, facing and guard
# are the only things left to earn.
#
# Predicted outcome, written before running it: if down_red falls well below the
# ~0.45-0.72 seen with an opponent in reach, standing is learnable and the
# boxing coupling is what breaks the policy. If it stays above ~0.5, standing is
# the wall, and no reward shape will move it.
#
#   wsl bash tools/stand_experiment.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

# One GPU job at a time: 6 GB, and overlapping them is what killed WSL before.
while pgrep -f "boxer.train|boxer.arena|boxer.rollout" > /dev/null; do sleep 20; done

mkdir -p runs/stand
"$PY" -u -m boxer.train --mode frozen --opponent passive \
  --spawn-distance 2.2 --target-range 2.2 \
  --total-steps 2621440 --num-evals 5 \
  --worlds 256 --batch-size 4096 --num-minibatches 8 \
  --output runs/stand > runs/stand/train.log 2>&1
echo "stand experiment exited $?"
