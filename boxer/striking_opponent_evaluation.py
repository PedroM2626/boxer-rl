"""Explicit learned-policy corner routing for independent mixed-policy evaluation."""
import numpy as np


def route_opponent_action(actions,opponent_action,evaluated_corner):
    original=np.asarray(actions);other=np.asarray(opponent_action)
    if evaluated_corner not in (0,1) or original.shape!=(2,17) or other.shape!=(17,):
        raise ValueError('explicit evaluated corner and two compatible 17-action policies required')
    if not np.isfinite(original).all() or not np.isfinite(other).all() or np.any(np.abs(other)>1.00001):
        raise ValueError('finite bounded opposing-policy actions required')
    result=original.copy();result[1-evaluated_corner]=other;return result
