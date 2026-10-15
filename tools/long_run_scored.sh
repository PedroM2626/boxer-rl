#!/usr/bin/env bash
# The fourth long run: the same frozen arm, under a scorecard instead of a counter.
#
# runs/frozen_event2_200m (199.2M steps) was trained with every target above the
# waist paying something, and rendered it is not boxing: the learner walks to one
# spot, drops its guard and flaps both hands at the opponent's chest and gloves at
# about 6 Hz. Under the counter it scored 31.3 hits in a 10 s round; re-measured in
# this scorecard the same checkpoint lands 0.000 punches with the same 126 contacts a
# round (256 rounds, and 8 and 20 steps of window read identically), so every one of
# those 31.3 was a hand on a hand -- a glove paid 0.25 and that was the
# cheapest scoring contact in the scene.
#
# What changes here is one question -- what counts as a scored punch -- and the
# answer moves in five places at once:
#   * arms 0.25 -> 0.0 (they stay collision geoms, so blocking is still physical);
#   * head 1.0 > torso 0.6 -> 0.5 > legs 0.4 -> 0.35, the order a scorecard uses;
#   * a zero-weight target leaves the hit count as well as the damage, so the
#     metric and the reward cannot disagree about a round again;
#   * rearm_steps 8 -> 20, i.e. 0.4 s, because 0.16 s permitted the 6 Hz flap;
#   * presence and score are separate quantities. Presence -- any striker inside the
#     margin of any target -- drives the rising edge and the contacts denominator;
#     score drives damage, hits and the window. Spending the window on any touch was
#     tried first and is unusable: at fighting distance the corners are in contact on
#     well over half the steps and never stay off each other for 20 in a row, so a
#     window spent by touch stays shut and the round scores nothing at all.
# The opponent stays a passive anchored statue, the weights stay untouched, and the
# guard pose stays the fitted one, so "did the aim change" has one candidate cause.
#
# What this run can say that the last one could not: whether paying nothing for a
# glove moves the aim off the gloves. If the learner still presses and flaps with
# a scorecard that refuses to buy it, the frozen-statue setup is the limit, not the
# weights -- and the next arm has to be against something that moves and answers.
#
# Read it at the end, not in the middle:
#   python -m boxer.arena runs/frozen_scored_200m passive --rounds 256 --length 500
#
#   wsl bash tools/verify_then_train.sh    # the gate: suite green, then the arm
#   wsl bash tools/long_run_scored.sh      # only if the suite is already green
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

# One job on the card at a time; a second process asking for its own pool is what
# has taken the WSL VM down before -- and the test suite counts, because it
# compiles the same physics. The pattern must not match this script's own command
# line: "tools/" does, and a waiter that waits for itself never starts the run.
while pgrep -f "boxer.train|boxer.arena|boxer.rollout|pytest|hand_cost|jab_probe" > /dev/null; do
  echo "waiting for the GPU to be free"; sleep 30
done

RUN=runs/frozen_scored_200m
# The horizon is a parameter because this arm has already been restarted once: the
# host powered off at 23:29 on 2026-10-14 with 45.9M steps logged and its last
# checkpoint at 41.9M, and brax 0.14 cannot continue a run -- a resume is a fresh
# horizon fine-tuned from those weights. Running the full 200M again would give this
# arm ~246M steps against the third arm's ~204M, which makes compute a second
# difference between them and stops the comparison being one-variable. So the
# relaunch asks for the remainder, and lands within ~2M steps of the other arm.
TOTAL_STEPS=${TOTAL_STEPS:-160000000}
echo "horizon for this attempt: $TOTAL_STEPS learner steps"
if [ -d "$RUN/checkpoint" ] && [ "$(ls "$RUN/checkpoint" | wc -l)" -gt 30 ]; then
  echo "$RUN already holds a finished arm; not writing into it"
  exit 1
fi

# If this run already holds checkpoints, the previous attempt did not end in a
# trainer crash: a host reboot takes this script with it, and a first attempt
# without --resume would start at step 0 and overwrite the weights that were saved.
# Detect them instead of relying on someone remembering the flag -- that hazard
# already cost the 204M arm its checkpoints once.
resume=""
if [ -n "$(ls -A "$RUN/checkpoint" 2>/dev/null)" ]; then
  echo "$RUN already has checkpoints; resuming their weights with a fresh horizon"
  resume=--resume
fi
mkdir -p "$RUN"
# brax 0.14 cannot continue a run: --resume restores the networks and restarts the
# optimizer and the step counter, so an attempt after a crash is a fresh horizon
# fine-tuned from the last checkpoint. That bounds the loss of a crash to the steps
# since the last checkpoint (~39 min), not to the whole arm.
for attempt in 1 2 3 4 5 6; do
  "$PY" -u -m boxer.train --mode frozen --opponent passive --anchor-opponent \
    --total-steps "$TOTAL_STEPS" --num-evals 40 \
    --worlds 256 --batch-size 4096 --num-minibatches 8 $resume \
    --output "$RUN" >> "$RUN/train.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "scorecard long run finished cleanly"
    break
  fi
  echo "trainer exited $rc on attempt $attempt; resuming from the last checkpoint"
  resume=--resume
  sleep 30
done
