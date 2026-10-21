"""PPO termination and inference contracts, without a physics rollout."""
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("flax")

import jax
import jax.numpy as jp
import optax
from boxer.warp_train import MLP, make_update, normal_logp, numpy_forward, split_params


@pytest.mark.parametrize("timeout", [0.0, 1.0])
def test_respawn_value_never_bootstraps_a_finished_round(timeout):
    policy, critic = MLP((), 2), MLP((), 1)
    pp = {"params": {"Dense_0": {"kernel": jp.zeros((1, 2)),
                                 "bias": jp.zeros(2)}}}
    pv = {"params": {"Dense_0": {"kernel": jp.zeros((1, 1)),
                                 "bias": jp.array([10.0])}}}
    tx = optax.sgd(0)
    update = make_update(policy, critic, tx, .2, 0, .5, .99, .95, (.05, 2.0))
    loc, scale = split_params(policy.apply(pp, jp.zeros((1, 1))))
    batch = dict(obs=jp.zeros((1, 1, 1)), raw_action=jp.zeros((1, 1, 1)),
                 logp_old=normal_logp(loc, scale, jp.zeros((1, 1))),
                 reward=jp.array([[2.0]]), done=jp.ones((1, 1)),
                 timeout=jp.full((1, 1), timeout), terminal_obs=jp.ones((1, 1)))
    _, _, _, metrics = update(pp, pv, tx.init((pp, pv)), batch)
    # target=2, predicted=10, squared error=64. A respawn bootstrap gives 3.61.
    assert float(metrics["value_loss"]) == pytest.approx(64)
    assert float(metrics["clip_frac"]) == 0


def test_numpy_inference_matches_flax_and_sampling_has_its_own_seed():
    policy = MLP((8,), 12)
    params = policy.init(jax.random.PRNGKey(3), jp.zeros((1, 5)))
    norm = {"mean": np.arange(5) * .1, "var": np.arange(5) + 1}
    cfg = dict(policy_layers=[8], action_size=6)
    obs = np.random.default_rng(2).normal(size=(4, 5)).astype(np.float32)
    loc, _ = split_params(policy.apply(params, (obs - norm["mean"]) /
                                       np.sqrt(norm["var"] + 1e-8)))
    predict = numpy_forward(cfg, params, norm)
    np.testing.assert_allclose(predict(obs), np.tanh(loc), atol=1e-6)
    a = numpy_forward(cfg, params, norm, stochastic=True, seed=9)
    b = numpy_forward(cfg, params, norm, stochastic=True, seed=9)
    np.testing.assert_array_equal(a(obs), b(obs))
    assert not np.array_equal(a(obs), b(obs + .1))


def test_points_reading_settles_only_standing_timeouts():
    from boxer.warp_arena import outcomes, points_decisions
    done = np.zeros((4, 3))
    done[-1] = 1
    out = np.zeros((4, 3, 2))
    out[-1, 2, 1] = 1  # Physical knockout still wins, regardless of points.
    damage = np.zeros_like(out)
    damage[-1] = [[1, 2], [2, 1], [3, 1]]
    clock = np.zeros_like(done)
    clock[-1] = 1
    original = outcomes(done, out)
    assert original[0].tolist() == [False, False, True]
    a_points, b_points, a_judged, b_judged, draw = points_decisions(done, out, damage, clock)
    assert a_points.tolist() == [True, False, False]
    assert b_points.tolist() == [False, True, False]
    assert a_judged.tolist() == [True, False, True]
    assert b_judged.tolist() == [False, True, False]
    assert not draw.any()


def test_corner_pooling_preserves_signed_margins_and_points_wins():
    from boxer.warp_arena import pool
    forward = dict(rounds=10, a_punch_value_margin=2., b_punch_value_margin=-2.,
                   a_win_with_points=.7, b_win_with_points=.3)
    reverse = dict(rounds=10, a_punch_value_margin=1., b_punch_value_margin=-1.,
                   a_win_with_points=.6, b_win_with_points=.4)
    result = pool(forward, reverse)
    assert result["a_punch_value_margin"] == pytest.approx(.5)
    assert result["b_punch_value_margin"] == pytest.approx(-.5)
    assert result["a_win_with_points"] == pytest.approx(.55)
    assert result["a_win_with_points"] + result["b_win_with_points"] == pytest.approx(1)
