#!/usr/bin/env bash
# Durable, resumable 200M continuation. Publishing requires the explicit --publish flag.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "${1:-}" == "--background" ]]; then
  if [[ -n "${WSL_INTEROP:-}" ]]; then
    printf '%s\n' 'Use tools/start_skills_200m.ps1 from PowerShell to keep the WSL host alive.' >&2
    exit 2
  fi
  shift
  mkdir -p runs
  nohup bash "$project_dir/tools/run_skills_200m.sh" "$@" \
    >> runs/skills_200m_supervisor.log 2>&1 < /dev/null &
  supervisor_pid=$!
  printf '%s\n' "$supervisor_pid" > runs/skills_200m_supervisor.pid
  printf 'Started supervisor PID %s; log: runs/skills_200m_supervisor.log\n' "$supervisor_pid"
  exit 0
fi
boxing_python="${BOXER_PYTHON:-$HOME/venvs/warp_test/bin/python}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
# The project remote uses the Windows SSH host alias. Git for Windows retains
# that configuration when invoked from WSL; Linux Git has a separate SSH config.
windows_git='/mnt/c/Program Files/Git/cmd/git.exe'
if [[ -n "${WSL_INTEROP:-}" && -f "$windows_git" ]]; then
  set -- --git-executable "$windows_git" --git-worktree "$(wslpath -w "$project_dir")" "$@"
fi
exec "$boxing_python" -u -m boxer.long_train "$@"
