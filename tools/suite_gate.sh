#!/usr/bin/env bash
# The suite gate, run in two halves.
#
# One process per half rather than one pass over tests/: a single process has
# rebooted the WSL VM somewhere around the twentieth minute of XLA compilation,
# and a reboot leaves a truncated log with no traceback to read. The trainer
# survives 20 h in one process, the suite does not, so the suite is split.
#
# Each half logs to /home/pedro/suite_<half>.log and its exit status lands in
# /home/pedro/suite_gate.status. Nothing here starts a training run: a red build
# must not spend GPU hours, and the decision to launch stays a separate step.
#
#   wsl bash -lc 'setsid nohup bash /mnt/e/boxer-agent/tools/suite_gate.sh > /home/pedro/suite_gate.out 2>&1 < /dev/null &'
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

STATUS=/home/pedro/suite_gate.status
: > "$STATUS"

run_half () {
  half="$1"
  shift
  "$PY" -m pytest "$@" -q > "/home/pedro/suite_${half}.log" 2>&1
  status=$?
  {
    echo "${half} exit=${status}"
    tail -2 "/home/pedro/suite_${half}.log"
  } >> "$STATUS"
}

run_half half_a tests/test_env.py tests/test_scene.py
run_half half_b tests/test_policy.py tests/test_arena.py tests/test_alternating.py \
                tests/test_selfplay.py tests/test_train.py
echo "gate finished" >> "$STATUS"
