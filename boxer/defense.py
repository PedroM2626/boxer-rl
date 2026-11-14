"""Punches that miss, and the ones the target slipped.

A clean punch lands on the head or torso (``ImpactScorecard``); a blocked
one lands on the arms or gloves under the same rules (``PairedFoundation``'s
``blocked``). A miss touches neither: the fist drives at the opponent's head
or torso (closing at ``attack_speed`` or faster within ``start_gap`` of it),
comes within ``near`` of its surface, and pulls back without touching a
target or the guard. It is *evaded* if the target moved away from the fist
at ``away_speed`` or faster at some point during the attack: a slip or a
step back, not just a punch thrown wide.

``MissTracker.update`` runs once per control step on the pair's state and
returns, per attacking corner, the punches that missed and those evaded
(both credited to the attacker's index, like ``scored`` and ``blocked``).
"""
from __future__ import annotations

import mujoco
import numpy as np


class MissTracker:
    def __init__(self, pair, attack_speed=1., start_gap=.35, near=.10, away_speed=.3):
        self.attack_speed, self.start_gap, self.near, self.away_speed = attack_speed, start_gap, near, away_speed
        self.fists = pair.fists
        self.targets = [sorted(t) for t in pair.targets]
        self.reset()

    def reset(self):
        self.active = np.zeros((2, 2), bool)
        self.closest = np.full((2, 2), np.inf)
        self.touched = np.zeros((2, 2), bool)
        self.away = np.zeros((2, 2))

    def _velocity(self, model, data, geom):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(geom), velocity, 0)
        return velocity[3:]

    def measure(self, model, data, i, hand):
        """(gap, closing speed, target speed away from the fist) for ``i``'s ``hand``
        and the nearest of the opponent's head and torso."""
        fist = int(self.fists[i][hand])
        best = None
        for target in self.targets[1 - i]:
            fromto = np.zeros(6)
            gap = mujoco.mj_geomDistance(model, data, fist, int(target), 1., fromto)
            if best is None or gap < best[0]:
                best = (gap, target, fromto)
        gap, target, fromto = best
        line = fromto[3:] - fromto[:3]
        if np.linalg.norm(line) < 1e-6:
            line = data.geom_xpos[target] - data.geom_xpos[fist]
        line = line / max(np.linalg.norm(line), 1e-8)
        fist_v, target_v = self._velocity(model, data, fist), self._velocity(model, data, target)
        return float(gap), float((fist_v - target_v) @ line), float(target_v @ line)

    def update(self, model, data, touched):
        """``touched``: (2, 2) hands that touched a target or the guard this step."""
        missed, evaded = np.zeros(2), np.zeros(2)
        for i in range(2):
            for hand in range(2):
                gap, closing, away = self.measure(model, data, i, hand)
                if not self.active[i, hand]:
                    if closing >= self.attack_speed and gap <= self.start_gap and not touched[i, hand]:
                        self.active[i, hand] = True
                        self.closest[i, hand], self.touched[i, hand], self.away[i, hand] = gap, False, away
                    continue
                self.closest[i, hand] = min(self.closest[i, hand], gap)
                self.touched[i, hand] |= bool(touched[i, hand])
                self.away[i, hand] = max(self.away[i, hand], away)
                if closing < 0. or gap > self.start_gap + .1:
                    self.active[i, hand] = False
                    if not self.touched[i, hand] and self.closest[i, hand] <= self.near:
                        missed[i] += 1
                        evaded[i] += self.away[i, hand] >= self.away_speed
        return missed, evaded
