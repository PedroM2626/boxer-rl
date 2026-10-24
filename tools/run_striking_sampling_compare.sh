#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
exec "$HOME/venvs/boxing_curriculum/bin/python" -u tools/striking_sampling_compare.py "$@"
