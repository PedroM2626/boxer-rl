#!/usr/bin/env bash
# Does the refitted guard pose change what the 204M arm does?
#
# The stance is not decoration: it is baked into the observation (the 6 guard-error
# dims and the 29 `qpos - stance` dims) and into the `guard` term of the reward. A
# checkpoint trained under the old pose therefore sees a shifted input distribution
# when it is loaded into the fixed one, so a video of it can be read two ways --
# "the agent looks better now" or "the agent got worse". Play it and find out.
#
# The comparison baseline is the table in the README (256 rounds, corners swapped,
# action_scale 0.25): 0.938 wins against the statue, 0.947 against the metronome,
# 0.754 against the 12.45M arm. `a_win` and `round_seconds` are still comparable
# across the reward change because winning only depends on who is standing; the
# `hits` columns are NOT, since hits became one event per contact instead of one
# count per physics substep.
#
#   wsl bash tools/stance_ab.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
export MUJOCO_GL=egl
cd "$PROJECT"

ARM=runs/frozen_200m/checkpoint

cell() {
  echo "=== $1 vs $2 (256 rounds, new stance) ==="
  "$PY" -u -m boxer.arena "$1" "$2" --rounds 256 --length 300 2>&1 |
    grep -E "^env:|a_win|b_win|draw|round_seconds|hits|heads|dropped"
}

video() {
  echo "=== video: $1 vs $2, $4 steps, seed $3 ==="
  "$PY" -u -m boxer.rollout "$1" --opponent "$2" --format mp4 \
    --length "$4" --seed "$3" --output "runs/stance_ab_$5" 2>&1 |
    grep -E "final metrics|wrote"
  cp "runs/stance_ab_$5/rollout.mp4" "fights/$5" 2>/dev/null
}

cell "$ARM" passive
cell "$ARM" jabber
cell "$ARM" runs/frozen/checkpoint

video "$ARM" passive 0 200 204M_vs_statue_newguard.mp4
video "$ARM" runs/frozen/checkpoint 0 200 204M_vs_12M_newguard.mp4
echo "=== all done ==="
