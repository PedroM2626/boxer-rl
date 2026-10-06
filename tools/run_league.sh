#!/usr/bin/env bash
# Runs the best-response league from the checkpoint the frozen arm produced.
#
# Both corners are transplanted from the frozen net and fight that same net in
# iteration 0, so the league starts from a policy that already boxes instead of
# from two random nets discovering balance the hard way. The measurement this
# exists to produce is whether a corner keeps what it learned while its opponent
# is a stale copy of it -- symmetric self-play lost 85% of its rounds to a statue
# at the same physics budget, and that is the failure this mode is meant to dodge.
#
#   wsl bash tools/run_league.sh
set -uo pipefail

PROJECT=/mnt/e/boxer-agent
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

SEED="${SEED:-runs/frozen/checkpoint}"
mkdir -p runs/alt

"$PY" -u -m boxer.alternating \
  --output runs/alt \
  --iterations "${ITERATIONS:-2}" \
  --steps-per-iter "${STEPS_PER_ITER:-4000000}" \
  --opponent "$SEED" \
  --seed-checkpoint "$SEED" \
  --worlds "${WORLDS:-256}" \
  --batch-size 4096 --num-minibatches 8 --num-evals 16 \
  > runs/alt/league.log 2>&1
echo "league exited $?"
