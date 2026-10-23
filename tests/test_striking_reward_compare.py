"""Training shaping remains bounded, finite and disabled after a fall."""
import numpy as np
import pytest

from tools.striking_reward_compare import reach_bonus, motion_bonus


def test_reach_signal_is_bounded_and_prefers_actual_hand_proximity():
    bonus = reach_bonus([0., .4], .02, True)
    assert bonus[0] == .005
    assert 0 < bonus[1] < bonus[0]
    np.testing.assert_array_equal(reach_bonus([0., 0.], .02, False), [0., 0.])
    np.testing.assert_allclose(reach_bonus([.1, .2], .01, True) * 2,
                               reach_bonus([.1, .2], .02, True))


@pytest.mark.parametrize('distances,dt', [([np.nan, 0], .02), ([-.1, 0], .02),
                                       ([0], .02), ([0, 0], 0)])
def test_invalid_shaping_inputs_fail_closed(distances, dt):
    with pytest.raises(ValueError):
        reach_bonus(distances, dt, True)


def test_motion_reward_requires_active_extension_and_similarity_without_a_clock():
    reference = np.zeros((3, 4))
    angles = np.zeros((2, 4))
    np.testing.assert_array_equal(motion_bonus(angles, reference, [0, -1], .02, True), [0, 0])
    np.testing.assert_array_equal(motion_bonus(angles, reference, [1, 20], .02, True), [.02, .02])
    np.testing.assert_array_equal(motion_bonus(angles, reference, [1, 1], .02, False), [0, 0])
    assert np.all(motion_bonus(angles + 1, reference, [1, 1], .02, True) < .001)
    with pytest.raises(ValueError):
        motion_bonus(angles, np.zeros((0, 4)), [1, 1], .02, True)
