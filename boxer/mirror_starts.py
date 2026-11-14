"""Starting states taken just before a fall in a policy's bouts against itself.

Run 6 (``checkpoints/whole-body``) falls in mirror bouts, mostly after a
minute, in body contact. ``collect`` plays it against itself in native
MuJoCo (restart rule, 3 knockdowns) and, whenever a boxer falls, keeps the
pair's state 0.5 to 3 s before (every 0.5 s) with ``risen`` set to the corner
that fell. ``native_finetune --mirror-bank`` starts a share of training bouts
from them against the same policy, the learner mostly in the falling corner.
The format is ``boxer.getup_starts``'s.

    python -m boxer.mirror_starts --output runs/mirror_starts.npz --bouts 60
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from boxer.getup_starts import FIELDS, snapshot

BEFORE = (.5, 1., 1.5, 2., 2.5, 3.)     # seconds before the fall kept


class FallWatcher:
    """A ``spar_match.bout`` observer: snapshots every ``every`` seconds and
    keeps those ``BEFORE`` a fall."""

    def __init__(self, every=.5, before=BEFORE):
        self.every, self.before = every, before
        self.recent = deque(maxlen=int(round(max(before) / every)) + 1)
        self.states = []

    def __call__(self, elapsed, pair, outcome):
        if outcome['done']:
            fallen = np.flatnonzero(outcome['fallen'])
            if len(fallen) == 1:
                for when, state in self.recent:
                    if any(abs(elapsed - when - b) < self.every / 2 for b in self.before):
                        self.states.append(dict(state, risen=int(fallen[0]), before=elapsed - when))
            self.recent.clear()          # a restart follows
            return
        if round(elapsed / pair.dt) % round(self.every / pair.dt) == 0:
            self.recent.append((elapsed, snapshot(pair)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", type=Path, default=Path("checkpoints/whole-body"))
    parser.add_argument("--arm", type=Path, default=Path("checkpoints/arm-skill"))
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar-native"))
    parser.add_argument("--output", type=Path, default=Path("runs/mirror_starts.npz"))
    parser.add_argument("--bouts", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)
    import jax
    from boxer.arm_skill import ArmSkill
    from boxer.curriculum_env import UnassistedG1
    from boxer.curriculum_train import inference, load_policy_params
    from boxer.paired_foundation import PairedFoundation
    from boxer.spar_match import bout, load_brain
    foundation = UnassistedG1('spar')
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation, ring=True)
    watcher = FallWatcher()
    brains = {i: load_brain(args.policy) for i in range(2)}
    skills = {i: ArmSkill(load_policy_params(args.arm)[0]) for i in range(2)}
    for k in range(args.bouts):
        result, _, _ = bout(pair, motor, args.seed + k, args.seconds, brains=brains, knockdowns=3,
                      arm_skills=skills, observer=watcher)
        print(json.dumps(dict(bout=k, knockdowns=result['knockdowns_suffered'], kept=len(watcher.states))),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    states = watcher.states
    np.savez_compressed(args.output, **{k: np.stack([s[k] for s in states]) for k in (*FIELDS, "risen", "before")})
    print(json.dumps(dict(output=str(args.output), states=len(states))), flush=True)


if __name__ == "__main__":
    main()
