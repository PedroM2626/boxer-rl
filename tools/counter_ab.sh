#!/usr/bin/env bash
# Does the fixed counter actually kill the flicker, or did I just describe it better?
#
# Same checkpoint, same environment, same sampler -- the only difference is whether
# the two gates are on. Comparing against last night's 43 punches/second would not
# be a test: that number came from the training loop's own episodic sums, with
# exploration noise and 256 worlds, and the arena is deterministic. The claim needs
# one measurement with the gates off and one with them on, from the same code.
#
# Read it this way. `a_hits_per_round` with the gates off is what the boxer really
# does to a welded target, which is expected to be a large nonsense number. With the
# gates on it should fall to something an arm could produce: a round is ~10 s and a
# person throws maybe one to two punches a second at most. `hits_per_contact` is the
# ratio framing -- how much of the contact was punch.
#
#   wsl bash tools/counter_ab.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

ARM=runs/frozen_event_200m/checkpoint

while pgrep -f "boxer.train|boxer.arena|boxer.rollout|pytest" > /dev/null; do
  echo "waiting for the card"; sleep 30
done

cell() {
  echo "=== counter: $1 ==="
  "$PY" -u -m boxer.arena "$ARM" passive --rounds 256 --length 500 $2 2>&1 |
    grep -E "^env:|a_win|b_win|round_seconds|a_hits_per_round|hits_per_contact|contacts_per_round|heads_per_round|dropped"
}

cell "gates OFF (as trained)" "--rearm-gap 0 --impact-ref 0"
cell "gates ON (the fix)" ""
echo "=== done ==="
