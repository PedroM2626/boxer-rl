#!/usr/bin/env bash
# The long run, with the intermediate target written down before starting it.
#
# Why this is now the right spend, after three null results:
#
#   arm                       down_red   hits_red   contacts
#   standing only (2.0 m)      0.82       0.00       3.0     at 2.62M steps
#   frozen, opponent in reach  0.72       0.42      50.2     at 2.62M steps
#   frozen, range at 0.7 m     0.76       0.20      40.7     at 2.62M steps
#
# The first row is the one that matters: with an opponent placed out of reach and
# essentially no contact happening, the learner still falls in 82% of rounds. So
# falling is not caused by the opponent, by the range reward's distance, or by the
# knockout term's magnitude -- all three of those were tested and were null. It is
# the base skill, and 2.6M steps does not have enough of it. The 12.45M arm is
# still descending at its last checkpoint (0.80 -> 0.46), which is the evidence
# that more steps buy something here after all.
#
# What would make this a wasted night, stated in advance so it is detectable and
# not just hoped away: if down_red has not gone below ~0.3 by 50M learner steps,
# then scale is not the blocker either and the problem is the control interface
# (50 Hz position actuators on 29 joints, action_scale 0.25) rather than the
# optimiser's budget. Checkpoints land every 5,242,880 steps, so that is readable
# after roughly 40 minutes and the run can be stopped with --resume still working.
#
#   wsl bash tools/long_run.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

while pgrep -f "boxer.train|boxer.arena|boxer.rollout" > /dev/null; do sleep 20; done

mkdir -p runs/frozen_200m
"$PY" -u -m boxer.train --mode frozen --opponent passive \
  --total-steps 200000000 --num-evals 40 \
  --worlds 256 --batch-size 4096 --num-minibatches 8 \
  --output runs/frozen_200m > runs/frozen_200m/train.log 2>&1
echo "long run exited $?"
