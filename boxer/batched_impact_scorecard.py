"""Vectorized independent impact-window state, preserving the scalar scorecard."""
import numpy as np
from boxer.impact_scorecard import ImpactScorecard


class BatchedImpactScorecard:
    fields = ('engaged', 'eligible', 'scored', 'impulse', 'age', 'absent', 'last_score')

    def __init__(self, worlds):
        if not isinstance(worlds, int) or worlds <= 0:
            raise ValueError('positive integer worlds required')
        self.worlds = worlds
        for key in self.fields:
            value = getattr(ImpactScorecard(), key)
            setattr(self, key, np.repeat(value[None], worlds, axis=0))

    def reset(self, rows=None):
        rows = np.arange(self.worlds) if rows is None else np.asarray(rows, int)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self.worlds) or len(set(rows.tolist())) != len(rows):
            raise ValueError('unique valid reset rows required')
        fresh = ImpactScorecard()
        for key in self.fields:
            getattr(self, key)[rows] = getattr(fresh, key)

    def update(self, present, impulses, closing, advancing, time, dt, both_up, active=None):
        shape = (self.worlds, 2, 2)
        present = np.asarray(present, bool)
        impulses, closing, advancing = map(np.asarray, (impulses, closing, advancing))
        time, both_up = np.asarray(time), np.asarray(both_up, bool)
        if any(a.shape != shape for a in (present, impulses, closing, advancing)) or time.shape != (self.worlds,) or both_up.shape != (self.worlds,):
            raise ValueError('exact batch shape required')
        if not all(np.isfinite(a).all() for a in (impulses, closing, advancing, time)):
            raise FloatingPointError('nonfinite batched impact input')
        if not np.isfinite(dt) or dt <= 0 or np.any(impulses < 0):
            raise ValueError('positive timestep and nonnegative impulses required')
        active = np.ones(self.worlds, bool) if active is None else np.asarray(active, bool)
        if active.shape != (self.worlds,): raise ValueError('per-world activity required')
        selected = np.flatnonzero(active)
        result = np.zeros((self.worlds, 2), np.float32)
        if not len(selected): return result
        p, force, close, advance = (x[selected] for x in (present, impulses, closing, advancing))
        engaged, eligible, scored, impulse, age, absent, last = (getattr(self, k)[selected].copy() for k in self.fields)
        t, up = time[selected, None, None], both_up[selected, None, None]
        ref = ImpactScorecard
        absent = np.where(p, 0., absent+dt)
        engaged[~p & (absent >= ref.release_seconds)] = False
        onset = p & ~engaged
        impulse[onset] = 0.; age[onset] = 0.; scored[onset] = False
        eligible[onset] = ((close >= ref.minimum_closing_speed) & (advance >= ref.minimum_fist_speed)
                          & (t-last >= ref.cooldown_seconds) & up)[onset]
        engaged |= p
        within = p & (age < ref.contact_window)
        impulse += np.where(within, force, 0.)
        age += p*dt
        events = within & eligible & ~scored & (impulse >= ref.minimum_impulse) & up
        scored |= events
        last = np.where(events, t, last)
        for key, value in zip(self.fields, (engaged, eligible, scored, impulse, age, absent, last)):
            getattr(self, key)[selected] = value
        result[selected] = events.sum(axis=2).astype(np.float32)
        return result
