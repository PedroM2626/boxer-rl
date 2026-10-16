#!/usr/bin/env bash
# Render the fifth arm once the trainer has let go of the card.
#
# The user's order: keep the run going, then produce the video. So this waits rather
# than competes -- two JAX processes on a 6 GB mobile 3060 is the failure mode that
# has taken the WSL VM down more than once, and a rollout holds its own compiled
# graphs on top of the trainer's 3.9 GB.
#
# It detaches itself on the first invocation, because asking wsl.exe to background a
# job from the Windows command line has already failed twice without leaving a trace
# (see tools/launch_paid.sh): the launcher printed "launched" and never created its
# own log file. Here the re-exec happens inside the distro, and the parent prints the
# process table it can see.
#
# Each cell is its own process on purpose. A loaded checkpoint keeps its compiled
# backward graphs alive for the life of the interpreter, so two cells in one python
# is two pools on the card at once.
set -uo pipefail

OUT="$HOME/arm5_video.out"
if [ -z "${DETACHED:-}" ]; then
  DETACHED=1 setsid nohup bash "$0" > "$OUT" 2>&1 < /dev/null &
  disown
  sleep 3
  echo "armed; follow $OUT"
  pgrep -af "video_after_arm" | grep -v pgrep
  exit 0
fi

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

ARM=runs/frozen_paid_200m/checkpoint

while pgrep -f "boxer.train|boxer.arena|boxer.rollout|pytest" > /dev/null; do
  sleep 60
done
echo "=== card free at $(date) ==="

if [ ! -d "$ARM" ]; then
  echo "no checkpoint at $ARM -- the run did not produce one; nothing to render"
  exit 1
fi
echo "checkpoint files: $(ls "$ARM" | wc -l)"

render() {
  echo "=== rollout vs $1, corner $2 anchored ==="
  "$PY" -u -m boxer.rollout "$ARM" --opponent "$1" --anchor "$2" --format mp4 \
    --length 500 --seed "$3" --output "runs/video_arm5_$4" 2>&1 |
    grep -vE "^\[|Deprecation|warnings.warn|UserWarning|Please see|For a host|^\s*$" | tail -12
  cp "runs/video_arm5_$4/rollout.mp4" "fights/$4" && echo "wrote fights/$4"
}

# The statue first: it is the opponent this arm was trained against, so it is the
# fair read of what the run learned. The jabber second, because a scripted answer
# shows whether the form survives something that moves.
render passive blue 0 arm5_vs_pinned_statue.mp4
render jabber blue 1 arm5_vs_jabber.mp4

echo "=== arena read, 256 rounds against the passive statue ==="
"$PY" -u -m boxer.arena "$ARM" passive --rounds 256 --length 500 2>&1 |
  grep -E "^env:|a_win|b_win|round_seconds|a_hits_per_round|hits_per_contact|contacts_per_round|heads_per_round|dropped"

echo "=== done at $(date) ==="
