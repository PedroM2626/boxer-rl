#!/usr/bin/env bash
# Keep this foreground process alive with start_curriculum.ps1 on WSL.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
boxing_python="${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
windows_git='/mnt/c/Program Files/Git/cmd/git.exe'
if [[ -n "${WSL_INTEROP:-}" && -f "$windows_git" ]]; then
  set -- --git-executable "$windows_git" --git-worktree "$(wslpath -w "$project_dir")" "$@"
fi
exec "$boxing_python" -u -m boxer.curriculum_job "$@"
