#!/usr/bin/env bash
# Judge the 200M arm by looking at it, then by playing it against everything else.
#
# The training log says down_red fell from 0.79 to ~0.05, so standing is solved and
# the scale hypothesis was right. It also says hits_red reached ~250 per 65-step
# episode, which is 3.8 per control step -- and hits accumulate per physics substep,
# so that is a fist resting on the opponent for most of every step, not punching.
# The render decides which of the two readings is the one that matters.
#
#   wsl bash tools/judge_200m.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
export MUJOCO_GL=egl
cd "$PROJECT"

NEW=runs/frozen_200m/checkpoint

while pgrep -f "boxer.train" > /dev/null; do sleep 20; done
echo "training done, rendering"

render() {
  echo "=== render seed $2, first $1 steps ==="
  "$PY" -u -m boxer.rollout "$NEW" --opponent passive --format sheet --sheet-tiles 4 \
    --length "$1" --seed "$2" --output "runs/judge_200m_$1_$2" 2>&1 |
    grep -E "final metrics|wrote"
}

render 60 0
render 60 7
render 240 0

cell() {
  echo "=== $1 vs $2 (256 rounds) ==="
  "$PY" -u -m boxer.arena "$1" "$2" --rounds 256 --length 300 2>&1 |
    grep -E "a_win|b_win|draw|round_seconds|hits|heads|dropped"
}

cell "$NEW" passive
cell "$NEW" jabber
cell "$NEW" runs/frozen/checkpoint
cell "$NEW" runs/frozen_v2/checkpoint
echo "=== all done ==="
