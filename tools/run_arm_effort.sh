#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
windows_git='/mnt/c/Program Files/Git/cmd/git.exe'
if [[ -n "${WSL_INTEROP:-}" && -f "$windows_git" ]]; then
  set -- --git-executable "$windows_git" --git-worktree "$(wslpath -w "$PWD")" "$@"
fi
exec "$HOME/venvs/boxing_curriculum/bin/python" -u tools/striking_arm_effort_supervisor.py "$@"
