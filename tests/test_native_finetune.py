"""Learner math for native-physics PPO fine-tuning."""
import jax
import jax.numpy as jp
import numpy as np

from boxer.native_finetune import floor_std, gae, tactics_net


def test_gae_stops_at_falls_and_bootstraps_only_through_clock_endings():
    rewards = np.array([1., 0., 2., 1.])
    values = np.array([.5, .5, .5, .5])
    next_values = np.array([.5, 9., .5, 3.])
    terminal = np.array([0., 1., 0., 0.])
    truncated = np.array([False, False, False, True])
    advantage, returns = gae(rewards, values, next_values, terminal, truncated, gamma=.9, lam=1.)
    # Step 1 fell: no bootstrap from next_values[1] = 9, and nothing leaks back past it.
    np.testing.assert_allclose(advantage[1], 0 - .5)
    np.testing.assert_allclose(advantage[0], (1 + .9 * .5 - .5) + .9 * advantage[1])
    # Step 3 hit the clock: bootstrap from its own next value only.
    np.testing.assert_allclose(advantage[3], 1 + .9 * 3 - .5)
    np.testing.assert_allclose(advantage[2], (2 + .9 * .5 - .5) + .9 * advantage[3])
    np.testing.assert_allclose(returns, advantage + values)


def test_std_floor_raises_only_small_scales_and_keeps_the_mean():
    net, dist = tactics_net()
    logits = jp.array([[.3, -.2, 0., .1, .5, -9., -9., 2., -9., -9.]])
    floored = floor_std(logits, .15)
    np.testing.assert_allclose(floored[:, :5], logits[:, :5])
    scale = jax.nn.softplus(floored[:, 5:]) + .001
    assert float(scale.min()) >= .15 - 1e-4
    np.testing.assert_allclose(floored[0, 7], 2.)             # already wider than the floor
    assert floor_std(logits, 0) is logits
    raw = dist.sample_no_postprocessing(floored, jax.random.PRNGKey(0))
    assert np.isfinite(float(dist.log_prob(floored, raw)[0]))
