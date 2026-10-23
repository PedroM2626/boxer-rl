#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
exec "$HOME/venvs/boxing_curriculum/bin/python" -u tools/replay_directed_feasibility.py --run "$HOME/boxing-runs/striking-directed-holdout-1441792" --output "$HOME/boxing-runs/striking-directed-holdout-1441792/replay.json" "$@"
