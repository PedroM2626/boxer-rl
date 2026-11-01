"""What happens in the second before a knockdown? Native fall-context census.

Plays long bouts under knockdown rules and records, every control step, the
robots' distance, rope contact, body contact between the robots (any
cross-robot contact not involving a fist), clean punches received and
whether each boxer's own punch is in flight. For every knockdown it reports
which of these occurred in the last second before the fall, next to how
often they occur at any moment (the base rate).
"""
import argparse
import collections
import json
from pathlib import Path

import jax
import numpy as np

from boxer import spar_match
from boxer.arm_skill import ArmSkill
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation


class Arms(spar_match.PunchArm):
    live = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Arms.live = (Arms.live + [self])[-2:]


def fall_reason(pair, corner):
    """Which fall rule fired: low pelvis, tilted torso, or a self-trip contact."""
    m = pair.maps[corner]
    if pair.data.qpos[m['root'] + 2] < .55:
        return "pelvis low"
    if pair.data.sensor(m['side'] + 'upvector_torso').data[2] < .6:
        return "torso tilted"
    for g1, g2 in pair.data.contact.geom:
        if pair._trip.get(frozenset((int(g1), int(g2)))) == corner:
            return "self-trip (own feet/shin touching)"
    return "other"


def body_contacts(pair):
    """Cross-robot contacts without a fist, as sorted (red part, blue part) names."""
    fists = set(map(int, pair.fists.ravel()))
    found = set()
    for g1, g2 in pair.data.contact.geom:
        g1, g2 = int(g1), int(g2)
        c1, c2 = pair._corner_of.get(g1), pair._corner_of.get(g2)
        if c1 is not None and c2 is not None and c1 != c2 and g1 not in fists and g2 not in fists:
            red, blue = (g1, g2) if c1 == 0 else (g2, g1)
            found.add((pair.model.geom(red).name.split('/', 1)[1], pair.model.geom(blue).name.split('/', 1)[1]))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red-tactics", type=Path, default=Path("checkpoints/tactics"))
    parser.add_argument("--blue-tactics", type=Path, default=Path("checkpoints/tactics-ring-v1"))
    parser.add_argument("--arm", type=Path, default=Path("checkpoints/arm-skill"))
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--output", type=Path, default=Path("runs/knockdown_causes.json"))
    args = parser.parse_args()
    foundation = UnassistedG1("spar")
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation, ring=True)
    tactics = [spar_match.PolicyBrain(load_policy_params(p)[0]) for p in (args.red_tactics, args.blue_tactics)]
    skills = {i: ArmSkill(load_policy_params(args.arm)[0]) for i in range(2)} if args.arm else {}
    history, falls = [], []
    original = pair.step

    def step(action):
        outcome = original(action)
        arms = Arms.live
        history.append(dict(distance=float(np.linalg.norm(pair.data.qpos[pair.maps[1]['root']:pair.maps[1]['root'] + 2]
                                                           - pair.data.qpos[pair.maps[0]['root']:pair.maps[0]['root'] + 2])),
                            ropes=pair.on_ropes().tolist(), parts=body_contacts(pair),
                            scored=np.asarray(outcome["scored"]).tolist(),
                            punching=[bool(a.age >= 0) for a in arms] if len(arms) == 2 else [False, False]))
        history[-1]["body"] = bool(history[-1]["parts"])
        if outcome["done"]:
            window = history[-50:]
            for corner in np.flatnonzero(outcome["fallen"]):
                corner = int(corner)
                falls.append(dict(corner=corner, reason=fall_reason(pair, corner),
                                  ropes=any(h["ropes"][corner] for h in window),
                                  body_contact=any(h["body"] for h in window),
                                  parts=sorted({tuple(p) for h in window for p in h["parts"]}),
                                  punches_taken=int(sum(h["scored"][1 - corner] for h in window)),
                                  own_punch=any(h["punching"][corner] for h in window),
                                  distance=round(window[-1]["distance"], 2)))
        return outcome
    pair.step = step
    spar_match.PunchArm = Arms
    for swapped in (False, True):
        for seed in args.seeds:
            brains = {0: tactics[1], 1: tactics[0]} if swapped else {0: tactics[0], 1: tactics[1]}
            spar_match.bout(pair, motor, seed, args.seconds, brains=brains, knockdowns=3, arm_skills=skills)
    n = len(history)
    base = dict(ropes=sum(any(h["ropes"]) for h in history) / n, body_contact=sum(h["body"] for h in history) / n)
    # Base rates over 1 s windows, comparable with the per-fall windows.
    windows = [history[k:k + 50] for k in range(0, n - 50, 50)]
    base_window = dict(ropes=np.mean([any(any(h["ropes"]) for h in w) for w in windows]),
                       body_contact=np.mean([any(h["body"] for h in w) for w in windows]),
                       punch_taken=np.mean([any(sum(h["scored"]) for h in w) for w in windows]))
    def kind(parts):
        legs = ("foot", "shin", "thigh", "hip", "pelvis")
        return "legs" if any(any(w in p for w in legs) for pair_ in parts for p in pair_) else "upper body"
    all_parts = collections.Counter(kind([tuple(p) for p in h["parts"]]) for h in history if h["parts"])
    fall_parts = collections.Counter(kind(f["parts"]) for f in falls if f["parts"])
    summary = dict(knockdowns=len(falls), steps=n,
                   contact_kind_all_steps=dict(all_parts), contact_kind_before_falls=dict(fall_parts),
                   share_of_falls={k: float(np.mean([f[k] for f in falls])) for k in ("ropes", "body_contact", "own_punch")}
                   if falls else {},
                   falls_after_punch_taken=float(np.mean([f["punches_taken"] > 0 for f in falls])) if falls else 0.,
                   base_rate_per_step=base, base_rate_per_1s_window={k: float(v) for k, v in base_window.items()},
                   fall_reasons=collections.Counter(f["reason"] for f in falls),
                   distance_at_fall=collections.Counter(round(f["distance"], 1) for f in falls), falls=falls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "falls"}, default=float))


if __name__ == "__main__":
    main()
