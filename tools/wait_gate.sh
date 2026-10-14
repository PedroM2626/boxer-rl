#!/usr/bin/env bash
# Blocks until tools/suite_gate.sh writes "gate finished", then prints both logs.
#
# A waiter rather than a poll: the gate runs 30-60 minutes and the point is to be
# told when it ends instead of asking. Bounded so a VM reboot -- which kills the
# pytest process and leaves the status file unfinished -- ends the wait with the
# logs instead of hanging forever.
#
#   wsl bash tools/wait_gate.sh
set -uo pipefail

STATUS=/home/pedro/suite_gate.status
tries=0
while [ "$tries" -lt 240 ]; do
  if grep -q "gate finished" "$STATUS" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "pyt[e]st" > /dev/null 2>&1 && ! grep -q "gate finished" "$STATUS" 2>/dev/null; then
    # No pytest running and no finished marker: either between the two halves or
    # the VM took the run down. Give the second half a chance to start, then stop.
    sleep 60
    if ! pgrep -f "pyt[e]st" > /dev/null 2>&1; then
      echo "=== no pytest running and the gate never finished ==="
      break
    fi
  fi
  sleep 30
  tries=$((tries + 1))
done

echo "=== status ==="
cat "$STATUS" 2>/dev/null
for half in half_a half_b; do
  echo "=== ${half} ==="
  tail -30 "/home/pedro/suite_${half}.log" 2>/dev/null
done
