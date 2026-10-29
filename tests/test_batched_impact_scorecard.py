"""Batched impact state must match independent scalar windows and selective resets."""
import numpy as np
from boxer.impact_scorecard import ImpactScorecard
from boxer.batched_impact_scorecard import BatchedImpactScorecard


def test_mixed_world_windows_pauses_and_resets_match_scalar_scorecards():
    worlds = 5
    batched = BatchedImpactScorecard(worlds)
    scalar = [ImpactScorecard() for _ in range(worlds)]
    rng = np.random.default_rng(73)
    for step in range(400):
        present = rng.uniform(size=(worlds, 2, 2)) > (.95 if step % 100 < 55 else .1)
        impulse = rng.uniform(0., .25, present.shape)
        close, advance = (rng.uniform(-.4, 1., present.shape) for _ in range(2))
        up = rng.uniform(size=worlds) > .1
        active = rng.uniform(size=worlds) > .15
        time = step*.002+np.arange(worlds)*.1
        actual = batched.update(present, impulse, close, advance, time, .002, up, active)
        expected = np.zeros((worlds, 2))
        for w in range(worlds):
            if active[w]: expected[w] = scalar[w].update(present[w], impulse[w], close[w], advance[w], time[w], .002, up[w])
        np.testing.assert_array_equal(actual, expected)
        if step % 79 == 0:
            rows = [step % worlds]
            batched.reset(rows)
            scalar[rows[0]].reset()
        for key in batched.fields:
            np.testing.assert_array_equal(getattr(batched, key), [getattr(s, key) for s in scalar])
