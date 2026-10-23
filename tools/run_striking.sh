#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
# Local Linux fsync keeps the per-action consumed ledger affordable. A complete
# checkpoint/state mirror is copied into the repository at every PPO iteration.
exec "${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}" -u \
    -m boxer.striking_train \
    --output "${BOXER_STRIKING_OUTPUT:-$HOME/boxing-runs/striking-pilot-v1}" \
    --publish-dir runs/striking-pilot "$@"
