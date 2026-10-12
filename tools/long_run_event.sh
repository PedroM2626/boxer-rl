#!/usr/bin/env bash
# The third long run: the same arm, retrained under a counter that counts punches.
#
# runs/frozen_event_200m finished 204.5M steps in a broken instrument, and reading
# it back is the reason this one exists. Three separate bugs made "hits" mean
# something other than punches, each found by measuring rather than by tuning:
#
#   * an event paid ~17 hits, because raw_landed sums how many contact slots
#     crossed the margin over the five substeps (fixed: hits and head shots are
#     indicators, damage still scales with how much connected);
#   * the retraction gate never bound, because it measured distance to the torso
#     *site* at the body's centre, so a fist resting on a chest counted as
#     retracted (fixed: rearm_steps, and contact spends the window);
#   * the arena anchored the blue corner while the corner swap puts the learner
#     there for half the rounds, so a_win sat at exactly 0.500 with the statue
#     falling by itself in the swapped half (fixed: the anchor follows the corner
#     holding the opponent).
#
# One variable changes from the previous arm: the counter. The reward weights are
# deliberately untouched, because this project has already spent three null results
# learning that weight tuning buys nothing while the instrument is wrong -- and
# because moving the head bonus at the same time would make this run unable to say
# which of the two did anything.
#
# What this run can say that the last one could not: the previous arm, re-measured
# in the fixed instrument, lands 19.4 punches in a 10 s round (1.96/s) and 0.000 to
# the head, and the head is reachable (+0.55 m of fist against +0.43 m) -- so if this
# arm also never aims high, it is shaping, and the head term is the next thing to
# change with a measurement behind it.
#
# Read it at the end, not in the middle: `python -m boxer.arena
# runs/frozen_event2_200m passive --rounds 256 --length 500`. Running the arena
# alongside a trainer is what has taken the VM down.
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

# Same hazard that nearly cost the 204M arm its checkpoints: a second trainer
# writing into a finished run's directory overwrites the evidence.
if [ -d runs/frozen_event2_200m/checkpoint ] && \
   [ "$(ls runs/frozen_event2_200m/checkpoint | wc -l)" -gt 30 ]; then
  echo "runs/frozen_event2_200m already holds a finished arm; not writing into it"
  exit 1
fi

mkdir -p runs/frozen_event2_200m
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
    --output runs/frozen_event2_200m >> runs/frozen_event2_200m/train.log 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "event-reward long run finished cleanly"
    break
  fi
  echo "trainer exited $rc on attempt $attempt; resuming from the last checkpoint"
  resume=--resume
  sleep 30
done
