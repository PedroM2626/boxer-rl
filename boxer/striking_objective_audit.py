"""Read-only accounting of the difference between own-fall and pair-fall cost."""
import numpy as np


def uncharged_pair_fall_cost(fallen,learner_corner):
    flags=np.asarray(fallen)
    if flags.shape!=(2,) or flags.dtype.kind!='b' or learner_corner not in (0,1):
        raise ValueError('two boolean first-fall flags and a learner corner required')
    return 5. if flags[1-learner_corner] and not flags[learner_corner] else 0.
