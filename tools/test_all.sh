#!/usr/bin/env bash
# Run the test suite one file per process, with the curriculum runtime on WSL.
#   bash tools/test_all.sh                      # every tests/test_*.py
#   bash tools/test_all.sh tests/test_spar.py   # only these files
# The whole suite in one pytest process exhausts the WSL VM's memory; one
# process per file keeps the peak to a single file's. Prints one line per
# file and a total, and exits non-zero if any file fails or times out.
set -uo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
boxing_python="${BOXER_CURRICULUM_PYTHON:-$HOME/venvs/boxing_curriculum/bin/python}"
per_file_timeout="${BOXER_TEST_TIMEOUT:-1800}"     # seconds
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
ulimit -c 0
if [ "$#" -gt 0 ]; then files=("$@"); else files=(tests/test_*.py); fi
failed=() passed=0 started=$SECONDS
for f in "${files[@]}"; do
  summary=$(timeout "$per_file_timeout" "$boxing_python" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -1)
  status=${PIPESTATUS[0]}
  echo "$f: $summary"
  if [ "$status" -ne 0 ]; then
    failed+=("$f")
  else
    passed=$((passed + $(sed -n 's/.*\b\([0-9]\+\) passed.*/\1/p' <<<"$summary")))
  fi
done
echo "files: ${#files[@]}, failed: ${#failed[@]}, tests passed: $passed, $(( (SECONDS - started) / 60 )) min"
for f in "${failed[@]}"; do echo "FAILED: $f"; done
[ "${#failed[@]}" -eq 0 ]
