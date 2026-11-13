#!/usr/bin/env bash
# Run the test suite one file per process, with the curriculum runtime on WSL.
#   bash tools/test_all.sh                      # every tests/test_*.py
#   bash tools/test_all.sh tests/test_spar.py   # only these files
#   bash tools/test_all.sh --fast               # skip the files in tools/slow_tests.txt
#   bash tools/test_all.sh --times FILE         # also write "seconds<TAB>file" lines to FILE
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
fast=0 times=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --fast) fast=1; shift ;;
    --times) times="$2"; shift 2 ;;
    *) break ;;
  esac
done
if [ "$#" -gt 0 ]; then files=("$@"); else files=(tests/test_*.py); fi
if [ "$fast" -eq 1 ]; then
  kept=()
  for f in "${files[@]}"; do
    grep -qxF "$f" tools/slow_tests.txt || kept+=("$f")
  done
  echo "fast mode: skipping $(( ${#files[@]} - ${#kept[@]} )) slow files (tools/slow_tests.txt)"
  files=("${kept[@]}")
fi
[ -n "$times" ] && : > "$times"
failed=() passed=0 started=$SECONDS
for f in "${files[@]}"; do
  file_started=$SECONDS
  summary=$(timeout "$per_file_timeout" "$boxing_python" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -1)
  status=${PIPESTATUS[0]}
  echo "$f: $summary"
  [ -n "$times" ] && printf '%s\t%s\n' "$(( SECONDS - file_started ))" "$f" >> "$times"
  if [ "$status" -ne 0 ]; then
    failed+=("$f")
  else
    passed=$((passed + $(sed -n 's/.*\b\([0-9]\+\) passed.*/\1/p' <<<"$summary")))
  fi
done
echo "files: ${#files[@]}, failed: ${#failed[@]}, tests passed: $passed, $(( (SECONDS - started) / 60 )) min"
for f in "${failed[@]}"; do echo "FAILED: $f"; done
[ "${#failed[@]}" -eq 0 ]
