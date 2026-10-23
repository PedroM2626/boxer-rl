"""Role slicing must preserve the original two-body physics and rewards."""
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_partner import PartnerStrikingEnv
from boxer.striking_env import StrikingEnv
from boxer.paired_foundation import PairedFoundation
from test_paired_foundation import foundation


@pytest.mark.parametrize('seed,swap', [(22, False), (23, True)])
def test_partner_routes_only_learner_and_preserves_physics(foundation, seed, swap):
    def motor(obs, rng):
        return jp.zeros((2, 29)), {}
    env = PartnerStrikingEnv(foundation, None, motor=motor)
    reference = StrikingEnv(foundation, None, motor=motor,
                           pair=PairedFoundation(foundation))
    obs = env.reset(seed, swap)
    expected = reference.reset(seed, swap)
    i = seed % 2
    np.testing.assert_array_equal(obs, expected[i:i+1])
    action = np.linspace(-.1, .1, 17)[None]
    paired = np.zeros((2, 17)); paired[i] = action[0]
    for _ in range(3):
        obs, reward, done, outcome = env.step(action)
        other, rewards, stop, _ = reference.step(paired)
        np.testing.assert_array_equal(env.env.pair.data.qpos, reference.pair.data.qpos)
        np.testing.assert_array_equal(obs, other[i:i+1])
        np.testing.assert_array_equal(reward, rewards[i:i+1])
        assert done == stop and outcome['learner_corner'] == i
        np.testing.assert_array_equal(env.env.previous[1-i], np.zeros(17))
    assert env.env.pair.model.neq == 0
    assert not np.any(env.env.pair.model.body_gravcomp)
