#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
exec "$HOME/venvs/boxing_curriculum/bin/python" -u tools/striking_directed_feasibility.py --target head --advance-only --seeds 2000003 2000004 2000005 2000006 2000007 2000008 2000009 2000010 2000011 2000012 2000013 2000014 2000015 2000016 2000017 2000018 "$@"
