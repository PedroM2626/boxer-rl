#!/usr/bin/env bash
# Run under a hidden persistent Windows WSL host, like run_curriculum.sh.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
boxing_python="${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}"
exec "$boxing_python" -u -m boxer.curriculum_compare "$@"
