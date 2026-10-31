"""What does a punch actually touch? Native fist-contact census per move.

Red throws only one move (scripted); blue is the scripted brain. Every
physics substep, contacts involving red's fists are recorded by the
opponent geom they touch, with peak normal force. This separates misses
(nothing touched) from blocked punches (arms or guard first) and from
landed contacts that the scorecard rejects.
"""
import argparse
import collections
import json
from pathlib import Path

import jax
import mujoco
import numpy as np

from boxer import moves
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import RingBrain, bout


def census(pair, motor, move, reach, defender_reach, high, seeds, seconds):
    touched = collections.Counter()
    force = collections.defaultdict(float)
    fists = set(map(int, pair.fists[0]))
    original = pair._physics_step

    def step():
        original()
        seen = set()
        for k, contact in enumerate(pair.data.contact):
            geoms = set(map(int, contact.geom))
            fist = geoms & fists
            if not fist:
                continue
            other = next(iter(geoms - fist), None)
            name = pair.model.geom(other).name if other is not None else "?"
            if not name.startswith("blue/"):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(pair.model, pair.data, k, f)
            seen.add(name)
            force[name] = max(force[name], float(f[0]))
        for name in seen:
            touched[name] += 1
    pair._physics_step = step
    totals = dict(thrown=0, landed=0)
    try:
        for seed in seeds:
            brains = {0: RingBrain(np.random.default_rng([seed, 0]), reach=reach, move=move),
                      1: RingBrain(np.random.default_rng([seed, 1]), reach=defender_reach, high=high)}
            result, _, _ = bout(pair, motor, seed, seconds, brains=brains)
            totals["thrown"] += result["punches_thrown"][0]
            totals["landed"] += int(result["clean_punches"][0])
    finally:
        pair._physics_step = original
    return dict(totals, substeps_touching={k[len("blue/"):]: v for k, v in touched.most_common()},
                peak_force={k[len("blue/"):]: round(v, 1) for k, v in force.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--reach", type=float, default=.55)
    parser.add_argument("--defender-reach", type=float, default=.55)
    args = parser.parse_args()
    foundation = UnassistedG1("spar")
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation)
    for name, move in (("straight", moves.STRAIGHT), ("hook", moves.HOOK)):
        for high in (False, True):
            row = census(pair, motor, move, args.reach, args.defender_reach, high, args.seeds, args.seconds)
            print(json.dumps(dict(move=name, defender_guard="high" if high else "normal", **row)), flush=True)


if __name__ == "__main__":
    main()
