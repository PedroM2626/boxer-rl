"""Independent fist-only impact events, not proximity or sustained force reward."""
from __future__ import annotations

import numpy as np


class ImpactScorecard:
    """One event per hand contact, requiring active pre-impact fist motion.

    A separated hand rearms after 100 ms; a 200 ms cooldown limits chatter.
    Contact impulses are integrated over the first 80 ms, never accumulated
    indefinitely by leaning. Thresholds are fixed before the boxing pilot.
    """

    minimum_impulse = .5  # N s
    minimum_closing_speed = .3  # m/s, relative to the legal target
    minimum_fist_speed = .2  # m/s toward target relative to own torso
    contact_window = .08
    release_seconds = .10
    cooldown_seconds = .20

    def __init__(self):
        self.reset()

    def reset(self):
        self.engaged = np.zeros((2, 2), bool)
        self.eligible = np.zeros((2, 2), bool)
        self.scored = np.zeros((2, 2), bool)
        self.impulse = np.zeros((2, 2))
        self.age = np.zeros((2, 2))
        self.absent = np.full((2, 2), self.release_seconds)
        self.last_score = np.full((2, 2), -np.inf)

    def update(self, present, impulses, closing, advancing, time, dt, both_up=True):
        present = np.asarray(present, bool)
        impulses, closing, advancing = map(np.asarray, (impulses, closing, advancing))
        if any(a.shape != (2, 2) for a in (present, impulses, closing, advancing)):
            raise ValueError('scorecard inputs must have shape (2, 2)')
        if not all(np.isfinite(a).all() for a in (impulses, closing, advancing)):
            raise FloatingPointError('non-finite impact input')
        if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0 or np.any(impulses < 0):
            raise ValueError('invalid impact timing or impulse')
        if not present.any():
            # Most substeps: no hand touches a target. Same result as below.
            self.absent += dt
            self.engaged &= self.absent < self.release_seconds
            return np.zeros(2, np.float32)
        self.absent = np.where(present, 0., self.absent + dt)
        rearm = ~present & (self.absent >= self.release_seconds)
        self.engaged[rearm] = False
        onset = present & ~self.engaged
        self.impulse[onset] = 0.
        self.age[onset] = 0.
        self.scored[onset] = False
        self.eligible[onset] = ((closing >= self.minimum_closing_speed)
                                & (advancing >= self.minimum_fist_speed)
                                & (time - self.last_score >= self.cooldown_seconds)
                                & bool(both_up))[onset]
        self.engaged |= present
        within = present & (self.age < self.contact_window)
        self.impulse += np.where(within, impulses, 0.)
        self.age += present * dt
        events = (within & self.eligible & ~self.scored
                  & (self.impulse >= self.minimum_impulse) & bool(both_up))
        self.scored |= events
        self.last_score[events] = time
        return events.sum(axis=1).astype(np.float32)
