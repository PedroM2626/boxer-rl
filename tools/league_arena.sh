#!/usr/bin/env bash
# Arena cells for the best-response league, at 256 rounds per cell.
#
# 256 rather than 64 because 64 was shown to be noise: the same cells at the same
# seeds disagreed by up to 0.11 in win rate there. One process per cell, because
# loading several checkpoints in one process OOMs a 6 GB card.
#
#   wsl bash tools/league_arena.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

run() {
  echo "=== $1 vs $2 (256 rounds) ==="
  "$PY" -u -m boxer.arena "$1" "$2" --rounds 256 --length 300 2>&1 |
    grep -E "a_win|b_win|draw|round_seconds|hits|heads"
}

run runs/alt/red/checkpoint passive
run runs/alt/blue/checkpoint passive
run runs/alt/red/checkpoint runs/frozen/checkpoint
run runs/alt/red/checkpoint runs/alt/blue/checkpoint
echo "=== done ==="
