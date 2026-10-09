#!/usr/bin/env bash
# The second long run: same budget as the first, in a fixed instrument.
#
# Everything trained before this point saw three things that were wrong:
#
#   * a guard pose that held the fists at chin height by breaking both wrists,
#     which is 29 observation dims and the action's own offset (measured cost:
#     the 204M arm goes from 0.938 wins against a statue to 0.139 when the pose
#     is corrected underneath it, so that 200M is not re-usable, only re-playable
#     through --guard-pose broken-wrist);
#   * a hit counter that accumulated per physics substep, so a fist resting on the
#     opponent paid 185 times a round and the best policy was to close and topple;
#   * no notion of an embrace at all, so the collapse race had no cost.
#
# Those are fixed now, so the objective this run optimises is the one the README
# claims it always had. The budget is deliberately the same as the first long run
# because the reason for the budget has not changed: standing is the base skill and
# on this card it needed ~100M steps, which is roughly what MuJoCo Playground spends
# on the G1 just to walk.
#
# What would make this a wasted night, stated before starting it rather than after:
# read the arena at the 100M checkpoint (256 rounds, corners swapped). If the arm is
# upright and still lands fewer than ~1 punch per round, then scale is not what buys
# punching either, and the remaining suspect is the arm's own pathway -- wrist
# actuators limited to 5 N.m at kp 20, saturating at a quarter radian of error, with
# no reward term that can see the difference between a punch and a palm that happens
# to be near a face. `python -m boxer.arena runs/frozen_event_200m passive --rounds 256`
#
#   wsl bash tools/long_run_event.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

# One job on the card at a time; a second process asking for its own 4.5 GiB pool is
# what has taken the WSL VM down before -- and the test suite counts, because it
# compiles the same physics. Waiting here means the suite can be started first and
# this picks up the card the moment it is free.
while pgrep -f "boxer.train|boxer.arena|boxer.rollout|pytest" > /dev/null; do
  echo "waiting for the GPU to be free"; sleep 30
done

mkdir -p runs/frozen_event_200m
# The card has taken the WSL VM down three times in one night, and a trainer that
# dies at hour 20 should not lose the run. Every attempt after the first resumes
# from the newest checkpoint, which lands every 5,242,880 steps, so the worst case
# cost of a crash is ~40 minutes. This does not survive the *VM* dying -- a loop
# inside WSL cannot outlive it -- so a reboot still needs the script re-run.
resume=""
for attempt in 1 2 3 4 5 6; do
  "$PY" -u -m boxer.train --mode frozen --opponent passive --anchor-opponent \
    --total-steps 200000000 --num-evals 40 \
    --worlds 256 --batch-size 4096 --num-minibatches 8 $resume \
    --output runs/frozen_event_200m >> runs/frozen_event_200m/train.log 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "event-reward long run finished cleanly"
    break
  fi
  echo "trainer exited $rc on attempt $attempt; resuming from the last checkpoint"
  resume=--resume
  sleep 30
done
