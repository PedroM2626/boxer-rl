#!/usr/bin/env bash
# Run a boxer module with the curriculum GPU stack on WSL, e.g.
#   bash tools/run_spar.sh boxer.curriculum_eval checkpoints/curriculum/approach --stage spar ...
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
boxing_python="${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
module="$1"
shift
exec "$boxing_python" -u -m "$module" "$@"
