import numpy as np


class Pair:
    fists = np.array([[0, 1], [2, 3]])
    targets = [{4, 5}, {6, 7}]


def run(tracker, frames, touched_at=()):
    """frames: (gap, closing, away) of red's left hand per step; other hands idle."""
    totals = np.zeros((2, 2))
    for t, frame in enumerate(frames):
        tracker.measure = lambda model, data, i, hand: frame if (i, hand) == (0, 0) else (1., 0., 0.)
        touched = np.zeros((2, 2), bool)
        touched[0, 0] = t in touched_at
        missed, evaded = tracker.update(None, None, touched)
        totals += [missed, evaded]
    return totals[:, 0]


def test_a_punch_that_comes_close_and_pulls_back_untouched_misses_and_is_evaded_if_the_target_moved_away():
    from boxer.defense import MissTracker
    tracker = MissTracker(Pair())
    slipped = [(.3, 2., 0.), (.15, 2., .5), (.05, 1., .4), (.12, -1., 0.)]
    assert run(tracker, slipped).tolist() == [1., 1.]
    wide = [(.3, 2., 0.), (.08, 1., 0.), (.2, -1., 0.)]              # target did not move: a miss only
    assert run(tracker, wide).tolist() == [1., 0.]
    short = [(.3, 2., .5), (.2, 1., .5), (.25, -1., 0.)]             # never within 10 cm: nothing
    assert run(tracker, short).tolist() == [0., 0.]
    blocked = [(.3, 2., .5), (.05, 1., .5), (.1, -1., 0.)]           # touched the guard: not a miss
    assert run(tracker, blocked, touched_at={1}).tolist() == [0., 0.]
    slow = [(.3, .5, .5), (.05, .5, .5), (.2, -1., 0.)]              # never an attack
    assert run(tracker, slow).tolist() == [0., 0.]
