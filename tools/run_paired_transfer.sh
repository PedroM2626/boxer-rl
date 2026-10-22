#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
exec "${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}" -u \
    tools/evaluate_paired_foundation.py "$@"
