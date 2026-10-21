#!/usr/bin/env bash
# Reproduce the assisted self-play run. Extra arguments override these defaults.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
boxing_python="${BOXER_PYTHON:-$HOME/venvs/warp_test/bin/python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
exec "$boxing_python" -u -m boxer.warp_train \
  --environment skills --worlds 256 --total-steps 2097152 \
  --batch-size 256 --num-minibatches 4 --unroll-length 20 \
  --num-updates-per-batch 4 --policy-layers 64 64 --value-layers 128 128 \
  --learning-rate 0.0003 --discounting 0.99 --round-steps 600 \
  --damage-cap 6 --save-every 10 --output runs/skills2 "$@"
