#!/usr/bin/env bash
# Install the isolated GPU stack and the exact upstream robot assets.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
boxing_env="${BOXER_CURRICULUM_VENV:-$HOME/venvs/boxing_curriculum}"
boxing_uv="${BOXER_UV:-$(command -v uv || true)}"
if [[ -z "$boxing_uv" && -x "$HOME/venvs/boxer/bin/uv" ]]; then
  boxing_uv="$HOME/venvs/boxer/bin/uv"
fi
if [[ -n "$boxing_uv" ]]; then
  if [[ ! -x "$boxing_env/bin/python" ]]; then
    "$boxing_uv" venv --python 3.12 "$boxing_env"
  fi
  "$boxing_uv" pip install --python "$boxing_env/bin/python" -r requirements-curriculum.txt
else
  if [[ ! -x "$boxing_env/bin/python" ]]; then
    python3.12 -m venv "$boxing_env"
  fi
  "$boxing_env/bin/python" -m pip install -r requirements-curriculum.txt
fi
boxing_assets="${BOXER_CURRICULUM_MENAGERIE:-$HOME/.cache/boxing-curriculum/menagerie}"
boxing_revision=1b86ece576591213e2b666ebf59508454200ca97
if [[ ! -d "$boxing_assets" ]]; then
  mkdir -p "$boxing_assets"
  git -C "$boxing_assets" init
  git -C "$boxing_assets" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
  git -C "$boxing_assets" fetch --depth 1 --filter=blob:none origin "$boxing_revision"
  git -C "$boxing_assets" sparse-checkout init --cone
  git -C "$boxing_assets" sparse-checkout set unitree_g1
  git -C "$boxing_assets" checkout --detach FETCH_HEAD
fi
test "$(git -C "$boxing_assets" rev-parse HEAD)" = "$boxing_revision"
printf '%s\n' 'Pinned curriculum runtime and G1 assets are ready.'
