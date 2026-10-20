#!/usr/bin/env bash
# Run the suite in halves, and only if it is green, say so.
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

# Two jobs, not one. The whole suite in a single process kills the WSL VM at
# 20-25 minutes every time -- four reboots in one night, `uptime` reading 0 min and
# the log stopping mid-progress with no summary -- while a trainer ran 20.6 h
# uninterrupted on the same card. So the suite is run in halves and each half
# reports its own exit status; a gate that cannot tell "passed" from "the VM died"
# is not a gate.
"$PY" -m pytest tests/test_env.py -q > /home/pedro/suite_half1.log 2>&1
status1=$?
tail -2 /home/pedro/suite_half1.log
"$PY" -m pytest tests/ -q --ignore=tests/test_env.py > /home/pedro/suite_half2.log 2>&1
status2=$?
tail -2 /home/pedro/suite_half2.log
if [ "$status1" -ne 0 ] || [ "$status2" -ne 0 ]; then
  echo "suite is not green (halves exit $status1/$status2); not spending GPU hours on it"
  exit 1
fi

# It stops at the gate on purpose. This used to `exec bash tools/long_run_event.sh`,
# which is an MJX-era launcher for a different arm -- a gate that silently starts the
# wrong 11-hour job is worse than no gate. Print the command, let the human aim it.
echo "suite is green; launch with:"
echo "  PYTHONPATH=$PROJECT JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\"
echo "    nohup /home/pedro/venvs/warp_test/bin/python -u -m boxer.warp_train \\"
echo "      --worlds 2048 --imitation runs/reference/jab_targets_cmu.json --output runs/warpN &"
