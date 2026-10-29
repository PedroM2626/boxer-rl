"""Observation ordering must match the native policy before GPU adoption."""
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mujoco_playground')

from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_features import PairedFeatures, snapshot
from boxer.striking_contact import ContactAwareStrikingEnv


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('distance', [.58, .45])
def test_all_policy_inputs_match_native_with_asymmetric_history(swap, distance):
    foundation = UnassistedG1('approach', impl='jax')
    pair = PairedFoundation(foundation)
    env = ContactAwareStrikingEnv(foundation, None, pair=pair, motor=lambda *x: None,
                                 engagement_distance=distance)
    pair.reset(seed=47, swap=swap)
    rng = np.random.default_rng(9)
    pair.phase = rng.uniform(-np.pi, np.pi, (2, 2))
    pair.last_action = rng.uniform(-1, 1, (2, 29))
    env.previous = rng.uniform(-1, 1, (2, 17))
    pair.scorecard.last_score[0, 1] = 1.
    pair.scorecard.last_score[1, 0] = 2.
    features = PairedFeatures(pair, engagement_distance=distance)
    state = snapshot(pair.data)
    # Duplicate the world, checking that the implementation does not mix corners
    # or world axes. Different seeds and the swap exercise rotated local frames.
    state = {k: np.repeat(v, 2, axis=0) for k, v in state.items()}
    last = np.repeat(pair.last_action[None], 2, axis=0)
    phase = np.repeat(pair.phase[None], 2, axis=0)
    previous = np.repeat(env.previous[None], 2, axis=0)
    flags = np.repeat(np.isfinite(pair.scorecard.last_score)[None], 2, axis=0)
    offsets = rng.uniform(-1, 1, (2, 3))
    command = features.commands(state, offsets=np.repeat(offsets[None], 2, axis=0))
    np.testing.assert_allclose(command[0], env.commands(np.column_stack([offsets, np.zeros((2, 14))])), atol=1e-7)
    np.testing.assert_allclose(features.motor_observation(state, command, last, phase)[0],
                               pair.observation(command[0]), atol=1e-6)
    actual = features.striking_observation(state, last, phase, previous, flags)
    np.testing.assert_allclose(actual[0], env.observation(), atol=1e-6)
    np.testing.assert_array_equal(actual[0], actual[1])
