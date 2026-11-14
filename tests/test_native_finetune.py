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


def test_scripted_high_guard_opponents_hold_the_guard():
    from boxer.native_finetune import NativeBout

    class Pair:
        pose, action_scale, dt = np.zeros(29), 1., .02

        def reset(self, seed):
            pass

    holds = []
    for fraction in (0., 1.):
        bout = NativeBout(Pair(), None, np.random.default_rng(0), [], 1., 10, 30., fraction)
        holds.append(bout.opponent.high)
    assert holds == [False, True]


def test_run_config_writes_paths_and_path_lists_as_text():
    import argparse
    import json
    from pathlib import Path
    from boxer.native_finetune import run_config
    args = argparse.Namespace(output=Path("runs/x"), pool=[Path("a"), Path("b")], long_gate_opponents=[],
                              steps=3, ring=True)
    config = run_config(args)
    assert json.loads(json.dumps(config)) == dict(output="runs/x", pool=["a", "b"], long_gate_opponents=[],
                                                  steps=3, ring=True)


def test_stack_fall_penalty_applies_only_against_non_whole_body_opponents():
    from boxer.native_finetune import NativeBout
    bout = NativeBout.__new__(NativeBout)
    bout.fall_penalty, bout.stack_fall_penalty = 60., 120.
    bout.opponent = type("Stack", (), {})()
    assert bout.learner_fall_penalty() == 120.
    bout.opponent = type("WholeBody", (), {"legs": None})()
    assert bout.learner_fall_penalty() == 60.


def test_banked_starts_restore_a_state_and_mostly_put_the_learner_in_the_risen_corner(monkeypatch):
    import boxer.getup_starts as getup_starts
    from boxer.native_finetune import NativeBout

    class Pair:
        pose, action_scale, dt = np.zeros(29), 1., .02

        def reset(self, seed):
            self.restored = None

    monkeypatch.setattr(getup_starts, "restore", lambda pair, state: setattr(pair, "restored", state))
    bank = [dict(risen=1)]
    bout = NativeBout(Pair(), None, np.random.default_rng(0), [], 1., 10, 30.,
                      start_bank=bank, start_bank_fraction=1.)
    learners = []
    for _ in range(400):
        bout.reset()
        assert bout.banked and bout.pair.restored is bank[0]
        learners.append(bout.learner)
    assert .65 < np.mean(learners) < .85
    bout.start_bank_fraction = 0.
    bout.reset()
    assert not bout.banked and bout.pair.restored is None


def test_mirror_starts_face_the_mirror_opponent_with_the_learner_mostly_in_the_falling_corner(monkeypatch):
    import boxer.getup_starts as getup_starts
    from boxer.native_finetune import NativeBout

    class Pair:
        pose, action_scale, dt = np.zeros(29), 1., .02

        def reset(self, seed):
            self.restored = None

    monkeypatch.setattr(getup_starts, "restore", lambda pair, state: setattr(pair, "restored", state))
    count, mirror, itself, other = [dict(risen=0)], [dict(risen=1)], object(), object()
    bout = NativeBout(Pair(), None, np.random.default_rng(0), [other], 0., 10, 30.,
                      start_bank=count, start_bank_fraction=.25, mirror_bank=mirror, mirror_bank_fraction=.25,
                      mirror_opponent=itself)
    kinds, learners = [], []
    for _ in range(800):
        bout.reset()
        if bout.mirrored:
            assert bout.pair.restored is mirror[0] and bout.opponent is itself
            learners.append(bout.learner)
            kinds.append("mirror")
        elif bout.banked:
            assert bout.pair.restored is count[0] and bout.opponent is other
            kinds.append("count")
        else:
            assert bout.pair.restored is None and bout.opponent is other
            kinds.append("fresh")
    assert .2 < kinds.count("mirror") / 800 < .3 and .2 < kinds.count("count") / 800 < .3
    assert .65 < np.mean(learners) < .85
