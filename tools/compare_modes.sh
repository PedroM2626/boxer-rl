#!/usr/bin/env bash
# Compares the training modes under equalised physics, then ranks them in the arena.
#
# Equalising on *world* steps, not learner steps: self-play counts two learners
# per physics world while frozen mode counts one, so an equal learner budget
# would silently give the frozen run twice the simulated sparring. Both runs here
# see WORLD_STEPS physics steps.
#
#   wsl bash tools/compare_modes.sh
set -uo pipefail

PROJECT="${PROJECT:-/mnt/e/boxer-agent}"
PY="${PY:-/home/pedro/venvs/boxer313/bin/python}"
WORLD_STEPS="${WORLD_STEPS:-10000000}"
WORLDS="${WORLDS:-256}"
export PYTHONPATH="$PROJECT"
export JAX_PLATFORMS=cuda,cpu
cd "$PROJECT"

mkdir -p runs/frozen runs/selfplay

# self-play: 2 learners per world -> 2 * WORLD_STEPS learner steps
# frozen:    1 learner  per world ->     WORLD_STEPS learner steps
COMMON=(--worlds "$WORLDS" --batch-size 4096 --num-minibatches 8 --num-evals 20)

echo "=== [1/2] frozen mode vs a passive opponent ($WORLD_STEPS world steps) ==="
"$PY" -u -m boxer.train --mode frozen --opponent passive \
  --total-steps "$WORLD_STEPS" "${COMMON[@]}" --output runs/frozen \
  > runs/frozen/train.log 2>&1 || echo "frozen training FAILED, see runs/frozen/train.log"

echo "=== [2/2] symmetric self-play ($WORLD_STEPS world steps) ==="
"$PY" -u -m boxer.train --mode selfplay \
  --total-steps $(( WORLD_STEPS * 2 )) "${COMMON[@]}" --output runs/selfplay \
  > runs/selfplay/train.log 2>&1 || echo "selfplay training FAILED, see runs/selfplay/train.log"

echo "=== arena round-robin ==="
"$PY" -u - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from boxer import arena
from boxer.env import BoxingEnv

env = BoxingEnv()
entries = {
    "frozen-trained": "runs/frozen/checkpoint",
    "selfplay-trained": "runs/selfplay/checkpoint",
    "passive": "passive",
    "jabber": "jabber",
}
names = list(entries)
print(f"{'':18}" + "".join(f"{n[:14]:>16}" for n in names))
for a in names:
    row = f"{a:18}"
    for b in names:
        if a == b:
            row += f"{'-':>16}"
            continue
        r = arena.matchup(env, entries[a], entries[b], rounds=64, length=300, seed=0)
        row += f"{r['a_win'] - r['b_win']:>16.3f}"
    print(row, flush=True)
print("\n(cell = A's win rate minus B's; positive means the row beats the column)")

for a in names[:2]:
    r = arena.matchup(env, entries[a], "passive", rounds=64, length=300, seed=0)
    print(f"{a}: vs passive win={r['a_win']:.3f} loss={r['b_win']:.3f} "
          f"round={r['round_seconds']:.2f}s punches={r['hits_per_round']:.2f} "
          f"heads={r['heads_per_round']:.2f}")
PY
