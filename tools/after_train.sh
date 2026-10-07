#!/usr/bin/env bash
# Waits for the running frozen_v2 training to finish, then measures and renders.
#
# Chained on purpose: the GPU is a 6 GB card and one job at a time is the rule
# that keeps WSL alive, so the arena cells and the renders have to queue behind
# training rather than overlap it.
#
#   wsl bash tools/after_train.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
export MUJOCO_GL=egl
cd "$PROJECT"

OLD=runs/frozen/checkpoint
NEW=runs/frozen_v2/checkpoint

while pgrep -f "boxer.train" > /dev/null; do sleep 20; done
echo "training finished, measuring"

cell() {
  echo "=== $1 vs $2 (256 rounds, new gate) ==="
  "$PY" -u -m boxer.arena "$1" "$2" --rounds 256 --length 300 2>&1 |
    grep -E "a_win|b_win|draw|round_seconds|hits|heads|dropped"
}

cell "$OLD" passive
cell "$NEW" passive
cell "$NEW" "$OLD"

render() {
  echo "=== rendering $1 (seed $3, first $2 steps) ==="
  "$PY" -u -m boxer.rollout "$1" --opponent passive --format sheet --sheet-tiles 4 \
    --length "$2" --seed "$3" --output "runs/render_$(basename "$(dirname "$1")")_$3" 2>&1 |
    grep -E "final metrics|wrote"
}

render "$NEW" 80 0
render "$OLD" 80 0
echo "=== all done ==="
