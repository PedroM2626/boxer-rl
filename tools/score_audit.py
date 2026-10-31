"""Why do fist-on-target contacts score or not? Native scorecard audit per move.

Every contact onset of a fist on a legal target (the moment the scorecard
opens an impact window) is logged with the move in flight, the scorecard's
gate values (closing and advancing speed along the fist -> target-centre
line) and the same speeds along the contact normal. After the 80 ms window
closes the outcome is classified by the first gate that failed:
cooldown, closing, advancing, impulse, or scored.

Scripted rows pit a single-move attacker against the scripted brain with a
normal or high guard; ``--tactics`` adds learned-vs-learned bouts.
"""
import argparse
import collections
import json
from pathlib import Path

import jax
import numpy as np

from boxer import moves, spar_match
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.impact_scorecard import ImpactScorecard
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import PolicyBrain, RingBrain

MOVE_NAMES = {moves.STRAIGHT: "straight", moves.HOOK: "hook"}


class Arms(spar_match.PunchArm):
    """PunchArm that registers itself, so contacts can be tagged with the move."""
    live = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Arms.live = (Arms.live + [self])[-2:]


class Audit:
    """Passive ``diagnostic_punch_trace`` hook; reads state, changes nothing."""

    def __init__(self):
        self.rows, self.open = [], {}

    def before_substep(self, pair, positions, velocities, torso_motion):
        self.velocities, self.positions = velocities, positions
        self.torso_motion = torso_motion
        self.engaged = pair.scorecard.engaged.copy()
        self.last_score = pair.scorecard.last_score.copy()

    def after_substep(self, pair, present, impulse, closing, advancing):
        card = pair.scorecard
        for key, row in list(self.open.items()):
            row["impulse"] = float(card.impulse[key])
            row["scored"] |= bool(card.last_score[key] > self.last_score[key])
            if card.age[key] >= card.contact_window or not card.engaged[key]:
                self._close(key)
        onsets = present & ~self.engaged
        for i, hand in zip(*np.nonzero(onsets)):
            key = (int(i), int(hand))
            if key in self.open:
                self._close(key)
            arm = Arms.live[i] if len(Arms.live) == 2 else None
            move = "idle"
            if arm is not None and arm.age >= 0 and arm.hand == hand:
                move = MOVE_NAMES[arm.move]
            normal_closing, normal_advancing, target = self._normal(pair, i, hand)
            self.open[key] = dict(
                corner=key[0], move=move, target=target,
                closing=float(closing[key]), advancing=float(advancing[key]),
                normal_closing=normal_closing, normal_advancing=normal_advancing,
                cooldown_ok=bool(pair.data.time - self.last_score[key] >= card.cooldown_seconds),
                eligible=bool(card.eligible[key]), impulse=float(card.impulse[key]),
                scored=bool(card.last_score[key] > self.last_score[key]))

    def _normal(self, pair, i, hand):
        """Largest closing/advancing speed along any fist-target contact normal."""
        fist = int(pair.fists[i][hand])
        best_c, best_a, name = -np.inf, -np.inf, "?"
        origin, velocity = self.torso_motion[i]
        for contact in pair.data.contact:
            geoms = [int(g) for g in contact.geom]
            if fist not in geoms:
                continue
            other = geoms[1] if geoms[0] == fist else geoms[0]
            if other not in pair.targets[1 - i]:
                continue
            n = contact.frame[:3] * (1. if geoms[0] == fist else -1.)   # fist -> target
            v_fist, v_target = self.velocities[fist], self.velocities[other]
            torso = velocity[3:] + np.cross(velocity[:3], self.positions[fist] - origin)
            c, a = float((v_fist - v_target) @ n), float((v_fist - torso) @ n)
            if c > best_c:
                best_c, best_a = c, a
                name = pair.model.geom(other).name.split("/", 1)[-1]
        return best_c, best_a, name

    def _close(self, key):
        row = self.open.pop(key)
        card = ImpactScorecard
        if row["scored"]:
            reason = "scored"
        elif not row["cooldown_ok"]:
            reason = "cooldown"
        elif row["closing"] < card.minimum_closing_speed:
            reason = "closing"
        elif row["advancing"] < card.minimum_fist_speed:
            reason = "advancing"
        elif row["impulse"] < card.minimum_impulse:
            reason = "impulse"
        else:
            reason = "other"
        row["reason"] = reason
        # Would the gates pass if speeds were measured along the contact normal?
        row["normal_gates_ok"] = bool(row["normal_closing"] >= card.minimum_closing_speed
                                      and row["normal_advancing"] >= card.minimum_fist_speed)
        self.rows.append(row)


def run(pair, motor, brains_for, seeds, seconds):
    audit = Audit()
    pair.diagnostic_punch_trace = audit
    original = spar_match.PunchArm
    spar_match.PunchArm = Arms
    try:
        for seed in seeds:
            spar_match.bout(pair, motor, seed, seconds, brains=brains_for(seed))
            for key in list(audit.open):
                audit._close(key)
    finally:
        spar_match.PunchArm = original
        pair.diagnostic_punch_trace = None
    return audit.rows


def summarize(rows, label):
    out = []
    for move in ("straight", "hook", "idle"):
        sub = [r for r in rows if r["move"] == move]
        if not sub:
            continue
        reasons = collections.Counter(r["reason"] for r in sub)
        gate_fail = [r for r in sub if r["reason"] in ("closing", "advancing")]
        rescued = sum(r["normal_gates_ok"] for r in gate_fail)
        out.append(dict(
            bouts=label, move=move, contacts=len(sub), reasons=dict(reasons.most_common()),
            targets=dict(collections.Counter(r["target"] for r in sub).most_common(4)),
            median_closing=round(float(np.median([r["closing"] for r in sub])), 2),
            median_normal_closing=round(float(np.median([r["normal_closing"] for r in sub])), 2),
            median_advancing=round(float(np.median([r["advancing"] for r in sub])), 2),
            median_impulse=round(float(np.median([r["impulse"] for r in sub])), 2),
            gate_failures_passing_on_normal=rescued))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--tactics", type=Path, default=Path("checkpoints/tactics"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rows", type=Path, help="write every contact row as JSON lines")
    args = parser.parse_args()
    foundation = UnassistedG1("spar")
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation)
    every = []
    for name, move in (("straight", moves.STRAIGHT), ("hook", moves.HOOK)):
        for high in (False, True):
            def brains(seed, move=move, high=high):
                return {0: RingBrain(np.random.default_rng([seed, 0]), move=move),
                        1: RingBrain(np.random.default_rng([seed, 1]), high=high)}
            rows = [dict(r, bouts=f"scripted {name} vs {'high' if high else 'normal'} guard")
                    for r in run(pair, motor, brains, args.seeds, args.seconds) if r["corner"] == 0]
            every += rows
            for line in summarize(rows, rows[0]["bouts"] if rows else name):
                print(json.dumps(line), flush=True)
    if args.tactics:
        params = load_policy_params(args.tactics)[0]
        rows = [dict(r, bouts="learned vs learned")
                for r in run(pair, motor, lambda seed: {0: PolicyBrain(params), 1: PolicyBrain(params)},
                             args.seeds, args.seconds)]
        every += rows
        for line in summarize(rows, "learned vs learned"):
            print(json.dumps(line), flush=True)
    if args.rows:
        args.rows.write_text("".join(json.dumps(r) + "\n" for r in every), encoding="utf-8")


if __name__ == "__main__":
    main()
