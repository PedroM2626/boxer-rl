"""How often each punch type lands against each guard, in native physics.

Red is a scripted attacker that throws only one move; blue is the scripted
brain holding either the normal or the high guard. Both use the current
sparring motor. Reports red's clean punches per punch thrown, for a range of
attacker reach settings (the root distance the attacker tries to hold).
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from boxer import moves
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import RingBrain, bout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--reaches", type=float, nargs="+", default=[.62, .52])
    parser.add_argument("--defender-reach", type=float, default=.62,
                        help="distance the scripted defender holds; lower it to test infighting")
    parser.add_argument("--moves", nargs="+", default=["straight", "hook"], choices=["straight", "hook"])
    parser.add_argument("--output", type=Path, default=Path("runs/moves/effectiveness.json"))
    args = parser.parse_args()
    foundation = UnassistedG1("spar")
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation)
    rows = []
    for reach in args.reaches:
        for move in [dict(straight=moves.STRAIGHT, hook=moves.HOOK)[m] for m in args.moves]:
            for high in (False, True):
                landed = thrown = fell = 0
                for seed in args.seeds:
                    brains = {0: RingBrain(np.random.default_rng([seed, 0]), reach=reach, move=move),
                              1: RingBrain(np.random.default_rng([seed, 1]), reach=args.defender_reach,
                                           high=high)}
                    result, _, _ = bout(pair, motor, seed, args.seconds, brains=brains)
                    landed += result["clean_punches"][0]
                    thrown += result["punches_thrown"][0]
                    fell += result["fallen"][0]
                row = dict(reach=reach, defender_reach=args.defender_reach, move="hook" if move == moves.HOOK else "straight",
                           defender_guard="high" if high else "normal", bouts=len(args.seeds),
                           thrown=thrown, landed=landed, land_rate=landed / max(thrown, 1),
                           attacker_falls=int(fell))
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
