#!/usr/bin/env bash
# Run the whole suite, and only if it is green, start the long run.
#
# This is the gate, and it has to be structural. The trainer's own wait loop checks
# for the *absence* of a competing process, not for a passing suite, so queueing it
# behind a suite that then fails would spend 26 GPU hours on a red build -- which is
# what nearly happened tonight, and "I will remember to kill it" is not an
# instrument. The suite's exit status is captured directly here rather than through
# a pipe, because `pytest | tail` reports the tail's status and hides the failure.
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
  echo "suite is not green (exit $status); not spending GPU hours on it"
  exit "$status"
fi

exec bash tools/long_run_event.sh
