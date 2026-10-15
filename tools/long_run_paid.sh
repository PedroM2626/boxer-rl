#!/usr/bin/env bash
# The fifth long run: the same statue, with a punch that actually pays.
#
# The fourth arm (runs/frozen_scored_200m, stopped at 161M steps by decision) learned
# to stand -- it fell in 100% of episodes early on and in 1% by the end -- and scores
# 0.000 punches across 256 arena rounds while making 403 contacts a round. That is not
# the counter being wrong again; it is the reward being invisible. The hit channel paid
# the *depth* of a contact, and the arm's measured damage was 0.013 per episode against
# a round return of 829, while merely standing and being alive pays 1.659 every step. A
# policy that grinds its gloves against a statue is reading those numbers correctly.
#
# One change: the hit channel pays the event (w_hit 3.0 -> 10.0 on `landed_hits`), with
# depth kept as a small extra (w_damage 1.0 on the old `landed`). A landed punch is now
# worth about six steps of standing, and the rearm window caps a boxer at 2.5 punches a
# second, so landing them becomes the best use of a round by a wide margin.
# tests/test_env.py pins that ratio so it cannot quietly decay back into the noise floor.
#
# It starts from the fourth arm's weights (--init-from), which is a transplant rather
# than a continuation -- brax 0.14 reloads the networks with a fresh optimizer and a
# restarted step counter -- and the reason is arithmetic too: the standing phase costs
# the first ~60M steps of any frozen arm, and paying 7 GPU hours to re-learn it would
# buy nothing that runs/frozen_scored_200m has not already bought.
#
# What this run decides: whether a visible hit channel is enough for the statue setup to
# produce punching. If it scores, the sixth arm faces something that moves and answers.
# If it lands 0.000 again with the channel worth six steps of standing, then the frozen
# statue is definitively the limit of what this project can learn from, and the sixth arm
# is not optional.
#
# Read it at the end, not in the middle:
#   python -m boxer.arena runs/frozen_paid_200m/checkpoint passive --rounds 256 --length 500
#
#   wsl bash tools/suite_gate.sh            # the gate; do not spend GPU hours on a red build
#   wsl bash tools/long_run_paid.sh         # only once the suite is green
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

# The pattern must not match this script's own command line -- "tools/" does, and a
# waiter that waits for itself never starts the run.
while pgrep -f "boxer.train|boxer.arena|boxer.rollout|pytest|hand_cost|jab_probe|fist_probe" > /dev/null; do
  echo "waiting for the GPU to be free"; sleep 30
done

RUN=runs/frozen_paid_200m
SEED=runs/frozen_scored_200m/checkpoint
if [ -d "$RUN/checkpoint" ] && [ "$(ls "$RUN/checkpoint" | wc -l)" -gt 30 ]; then
  echo "$RUN already holds a finished arm; not writing into it"
  exit 1
fi

# Same rule the fourth arm learned: a host reboot takes this script with it, and a
# relaunch without --resume would restart at step 0 and overwrite saved weights.
resume=""
if [ -n "$(ls -A "$RUN/checkpoint" 2>/dev/null)" ]; then
  echo "$RUN already has checkpoints; resuming their weights with a fresh horizon"
  resume=--resume
fi
mkdir -p "$RUN"
TOTAL_STEPS=${TOTAL_STEPS:-200000000}
echo "horizon for this attempt: $TOTAL_STEPS learner steps"

init=""
[ -n "$resume" ] || init="--init-from $SEED"

for attempt in 1 2 3 4 5 6; do
  "$PY" -u -m boxer.train --mode frozen --opponent passive --anchor-opponent \
    --total-steps "$TOTAL_STEPS" --num-evals 40 \
    --worlds 256 --batch-size 4096 --num-minibatches 8 $resume $init \
    --output "$RUN" >> "$RUN/train.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "paid-punch long run finished cleanly"
    break
  fi
  echo "trainer exited $rc on attempt $attempt; resuming from the last checkpoint"
  resume=--resume
  init=""
  sleep 30
done
