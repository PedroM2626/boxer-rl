#!/usr/bin/env bash
# Arena cells for the mode comparison, one subprocess per matchup.
#
# Each cell forks: an RTX 3060 laptop carries 6 GB, and the 12-cell matrix driven
# from a single process died with CUDA_ERROR_OUT_OF_MEMORY on the eleventh cell
# because every loaded checkpoint keeps its own compiled graphs alive. One
# process per cell pays ~30 s of JAX startup and gives the memory back.
#
#   wsl bash tools/arena_matrix.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

FROZEN=runs/frozen/checkpoint
SELFPLAY=runs/selfplay/checkpoint

run() {
  echo "=== $1 vs $2 ==="
  "$PY" -u -m boxer.arena "$1" "$2" --rounds 64 --length 300 2>&1 |
    grep -E "a_win|b_win|draw|undecided|round_seconds|hits_per_round|heads_per_round|damage_per_round"
}

run "$FROZEN" passive
run "$FROZEN" jabber
run "$SELFPLAY" passive
run "$SELFPLAY" jabber
run "$FROZEN" "$SELFPLAY"
echo "=== done ==="
