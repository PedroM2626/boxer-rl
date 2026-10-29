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
    rng = np.random.default_rng(9)
    features = PairedFeatures(pair, engagement_distance=distance)
    states, last, phase, previous, flags, offsets = [], [], [], [], [], []
    expected_commands, expected_motor, expected_striking = [], [], []
    # Distinct states/history/rotations in each world detect accidental world
    # broadcasting and corner/world swaps that identical duplicate rows miss.
    for world, seed in enumerate((47, 73)):
        pair.reset(seed=seed, swap=swap if world == 0 else not swap)
        pair.phase = rng.uniform(-np.pi, np.pi, (2, 2))
        pair.last_action = rng.uniform(-1, 1, (2, 29))
        env.previous = rng.uniform(-1, 1, (2, 17))
        pair.scorecard.last_score[world, 1-world] = 1.
        offset = rng.uniform(-1, 1, (2, 3))
        command = env.commands(np.column_stack([offset, np.zeros((2, 14))]))
        states.append(snapshot(pair.data))
        last.append(pair.last_action.copy()); phase.append(pair.phase.copy())
        previous.append(env.previous.copy()); flags.append(np.isfinite(pair.scorecard.last_score))
        offsets.append(offset); expected_commands.append(command)
        expected_motor.append(pair.observation(command)); expected_striking.append(env.observation())
    state = {k: np.concatenate([s[k] for s in states]) for k in states[0]}
    last, phase, previous, flags, offsets = map(np.asarray, (last, phase, previous, flags, offsets))
    command = features.commands(state, offsets=offsets)
    np.testing.assert_allclose(command, expected_commands, atol=1e-7)
    np.testing.assert_allclose(features.motor_observation(state, command, last, phase), expected_motor, atol=1e-6)
    actual = features.striking_observation(state, last, phase, previous, flags)
    np.testing.assert_allclose(actual, expected_striking, atol=1e-6)
    assert not np.array_equal(actual[0], actual[1])
