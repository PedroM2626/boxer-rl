#!/usr/bin/env bash
# Start the fifth arm so it outlives the shell that asked for it.
#
# Doing the setsid/nohup on a command line that travels Windows -> wsl.exe -> bash
# has just failed twice: the launcher exited before it created its own log file, so
# nothing ran and the only evidence was a missing file. A script in the repo is
# parsed by one shell, and the redirect target is created before the trainer is
# trusted. Then it reads the process table back, because "launched" printed by the
# caller is not evidence -- the caller printed that both times it launched nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$HOME/arm5.out"

setsid nohup bash tools/long_run_paid.sh > "$OUT" 2>&1 < /dev/null &
disown
sleep 5

echo "=== $OUT ==="
cat "$OUT"
echo "=== processes ==="
pgrep -af "long_run_paid|boxer.train" | grep -v pgrep || echo "NOTHING RUNNING"
