#!/usr/bin/env bash
# Run the whole suite, and only if it is green, train the event-reward arm.
#
# The reward changed shape (a punch is one event, embrace costs a step), so the
# existing checkpoints are no longer optimising what this environment pays for.
# The new arm uses the same budget as the 12.45M frozen arm on purpose: same
# worlds, same batch geometry, same opponent, so the comparison is about the
# objective and not about how long each one ran.
#
#   wsl bash tools/verify_then_train.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

"$PY" -m pytest tests/ -q > /home/pedro/suite_before_event_arm.log 2>&1
status=$?
tail -2 /home/pedro/suite_before_event_arm.log
if [ "$status" -ne 0 ]; then
  echo "suite is not green; not spending GPU hours on a changed reward"
  exit "$status"
fi

mkdir -p runs/frozen_event
"$PY" -u -m boxer.train --mode frozen --opponent passive \
  --total-steps 10000000 --num-evals 20 \
  --worlds 256 --batch-size 4096 --num-minibatches 8 \
  --output runs/frozen_event > runs/frozen_event/train.log 2>&1
echo "event arm exited $?"
