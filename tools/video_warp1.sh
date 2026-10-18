#!/usr/bin/env bash
# Render the two ends of the warp1 run: the first saved checkpoint (iteration 15,
# std 0.75) and the last one before the stop (iteration 375, std ~16).
#
# The comparison exists because the training log says the policy's action std
# blew up (0.71 -> 27.4) while its "hits" rose fivefold -- so the question the
# video answers is whether the late policy boxes or flails. Each cell is its own
# python process: a loaded checkpoint keeps its compiled graphs on the card for
# the life of the interpreter, and the trainer's warp batch is already gone.
#
#   bash tools/video_warp1.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT"

LATE=runs/warp1
EARLY=runs/warp1/checkpoint/step-4915200.pkl

run() {  # run <checkpoint> <format> <outdir> [label]
  echo "=== $2 $1 -> $4"
  "$PY" -u -m boxer.rollout "$1" --guard-pose natural --format "$2" \
      --length 500 --output "$3" 2>&1 |
    grep -vE "^\[|Deprecation|warnings.warn|UserWarning|Please see|For a host|^\s*$" |
    tail -8
}

run "$LATE" sheet runs/video_warp1_late late_sheet
run "$LATE" mp4 runs/video_warp1_late late_mp4
run "$EARLY" sheet runs/video_warp1_early early_sheet
run "$EARLY" mp4 runs/video_warp1_early early_mp4

cp runs/video_warp1_late/rollout.mp4  fights/warp1_it375_selfplay.mp4 &&
  echo "wrote fights/warp1_it375_selfplay.mp4"
cp runs/video_warp1_early/rollout.mp4 fights/warp1_it15_selfplay.mp4 &&
  echo "wrote fights/warp1_it15_selfplay.mp4"
echo "=== done ==="
