"""Native two-robot fine-tuning of the leg motor (boxer.motor_native)."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.curriculum_env import LEFT_ARM, UnassistedG1, stage_config
from boxer.motor_native import (BRAINS, LEGS, REWARD, VALUE_SIZE, MotorBout, leg_log_prob, motor_reward,
                                split_logits)
from boxer.paired_foundation import PairedFoundation


@pytest.fixture(scope='module')
def pair():
    config = stage_config('spar', impl='jax')
    config.noise_config.level = 0.
    return PairedFoundation(UnassistedG1('spar', config=config), ring=True)


def test_reward_tracks_commands_and_charges_falls():
    obs = np.zeros(103)
    obs[0:2] = [.3, 0.]
    still = np.zeros(29)
    on_track = motor_reward(obs, still, still, np.array([.3, 0., 0.]), False, .02)
    assert on_track == pytest.approx(.02 * (REWARD["linear"] + REWARD["yaw"]))
    assert motor_reward(obs, still, still, np.array([0., 0., 0.]), False, .02) < on_track
    jerk = still.copy()
    jerk[LEFT_ARM:] = 1.                                  # arm changes are not the motor's
    assert motor_reward(obs, still, jerk, np.array([.3, 0., 0.]), False, .02) == pytest.approx(on_track)
    assert motor_reward(obs, still, still, np.array([.3, 0., 0.]), True, .02) == pytest.approx(on_track - REWARD["fall"])


def test_log_prob_counts_only_legs_and_waist():
    logits = jax.random.normal(jax.random.PRNGKey(0), (3, 58))
    loc, scale = split_logits(logits)
    raw = loc + scale * jax.random.normal(jax.random.PRNGKey(1), loc.shape)
    moved_arms = raw.at[:, LEGS:].add(5.)
    np.testing.assert_allclose(leg_log_prob(loc, scale, raw), leg_log_prob(loc, scale, moved_arms))
    gaussian = jax.scipy.stats.norm.logpdf(raw[:, :LEGS], loc[:, :LEGS], scale[:, :LEGS]).sum(-1)
    np.testing.assert_allclose(leg_log_prob(loc, scale, raw), gaussian, rtol=1e-5)


def test_bout_replaces_arms_and_ends_on_a_fall(pair):
    bout = MotorBout(pair, np.random.default_rng(0), {3: None}, None, 50)
    assert bout.obs.shape == (2, 103) and bout.values.shape == (2, VALUE_SIZE)
    assert bout.kinds == [3, 3] and bout.values[0, 106 + 3] == 1 and bout.values[0, 106 + len(BRAINS) + 3] == 1
    standing = np.zeros((2, 29))
    standing[:, LEFT_ARM:] = 9.                           # never reaches the robots
    rewards, terminal, truncated, next_values, info = bout.step(standing)
    assert np.abs(pair.last_action[:, LEFT_ARM:]).max() < 2
    assert not terminal.any() and not truncated.any() and np.isfinite(rewards).all()
    for _ in range(400):                                  # legs gone limp: someone falls
        rewards, terminal, truncated, next_values, info = bout.step(-np.ones((2, 29)))
        if terminal.any() or truncated.any():
            break
    assert terminal.any() and (terminal | truncated).all()
    assert rewards[terminal].max() < -REWARD["fall"] / 2 and next_values.shape == (2, VALUE_SIZE)


def test_brain_weights_set_how_often_each_corner_draws_a_brain(pair):
    bout = MotorBout(pair, np.random.default_rng(0), {0: None, 3: None}, None, 50, weights={0: 0., 3: 1.})
    for _ in range(5):
        bout.reset()
        assert bout.kinds == [3, 3]
