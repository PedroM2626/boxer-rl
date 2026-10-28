"""Check learner-only arm cost and unchanged source action authority."""
import numpy as np
import pytest
from boxer.striking_arm_effort import PROFILE, arm_effort_config, effort_cost


def test_arm_effort_is_bounded_and_uses_only_the_learner():
    residual = np.zeros((2, 14))
    residual[0] = .5
    residual[1] = 1.
    assert effort_cost({'arm_residual': residual}, 0, .02) == pytest.approx(.01)
    assert effort_cost({'arm_residual': residual}, 1, .02) == pytest.approx(.04)
    assert PROFILE['action_authority_unchanged'] and PROFILE['full_cycle_bonus'] == 10.
    np.testing.assert_array_equal(residual[0], .5)


def test_invalid_or_out_of_range_residuals_cannot_enter_reward():
    with pytest.raises(ValueError):
        effort_cost({'arm_residual': np.full((2, 14), np.nan)}, 0, .02)
    with pytest.raises(ValueError):
        effort_cost({'arm_residual': np.full((2, 14), 1.1)}, 0, .02)
    with pytest.raises(ValueError):
        arm_effort_config({'version': 15, 'contact_rebalance_ppo': False})
