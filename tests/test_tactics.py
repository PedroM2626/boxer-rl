"""Learned tactics see and act the same way in batched training and native bouts."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer import moves
from boxer.curriculum_train import load_policy_params
from boxer.spar_match import PunchArm
from boxer.tactics_env import ACTION_SIZE, OBS_SIZE, REACH, TacticsEnv, decode, observe


def bodies(rng):
    def body():
        yaw = rng.uniform(-np.pi, np.pi)
        return dict(pos=rng.uniform(-1, 1, 2), vel=rng.uniform(-.3, .3, 2),
                    rot=np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]),
                    up=rng.uniform(.8, 1.), height=rng.uniform(-.05, .05))
    return body(), body()


def test_observation_and_decoding_match_between_numpy_and_jax():
    rng = np.random.default_rng(0)
    for _ in range(20):
        own, opp = bodies(rng)
        arm, other = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
        clock = rng.uniform()
        native = observe(np, own, opp, arm, other, clock)
        batched = observe(jp, *(jax.tree.map(jp.asarray, (own, opp))), jp.asarray(arm),
                          jp.asarray(other), jp.asarray(clock))
        assert native.shape == (OBS_SIZE,)
        np.testing.assert_allclose(native[26:], [arm[4], arm[5], other[4], other[5]])
        np.testing.assert_allclose(native, np.asarray(batched), rtol=1e-5, atol=1e-5)
        tactics = rng.uniform(-1.5, 1.5, ACTION_SIZE)
        distance, local = np.linalg.norm(opp["pos"] - own["pos"]), own["rot"].T @ (opp["pos"] - own["pos"])
        a = decode(tactics, distance, local)
        b = decode(jp.asarray(tactics), jp.asarray(distance), jp.asarray(local))
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float), rtol=1e-5, atol=1e-5)


def test_decoding_turns_toward_the_opponent_and_respects_motor_limits():
    command, want, hand, aim, move, high = decode(np.array([-1, 1, 1, 1, 0., .5, .5]), 2.0,
                                                  np.array([0., 2.]))
    assert command[2] == .5 and want and hand == 1 and aim == 1 and move == moves.HOOK and high
    # Legacy five-number tactics: straight punches and the normal guard.
    command, want, _, _, move, high = decode(np.array([-1, -1, 0, -1, 1.]), .5, np.array([.5, 0.]))
    assert move == moves.STRAIGHT and not high
    assert command[0] < 0 and command[1] > 0 and not want
    assert np.all(command >= [-.35, -.25, -.5]) and np.all(command <= [.55, .25, .5])


def test_native_arm_features_follow_a_punch():
    arm = PunchArm(np.zeros(29), .5, .02)
    np.testing.assert_allclose(arm.features(), 0)
    arm.launch(1, .25, np.random.default_rng(0))
    arm.actions()
    age, active, hand, aim, hook, high = arm.features()
    assert 0 < age < 1 and active == 1 and hand == 1 and aim == .25 and hook == 0 and high == 0
    arm = PunchArm(np.zeros(29), .5, .02)
    arm.launch(0, 1., np.random.default_rng(0), moves.HOOK)
    arm.actions(high=True)
    assert arm.features()[4] == 1 and arm.features()[5] == 1


def test_native_arms_hold_the_high_guard_and_hook_through_the_wind_up():
    arm = PunchArm(np.zeros(29), .5, .02)
    rest = arm.actions(high=True)
    np.testing.assert_allclose(rest[:4], moves.HIGH_GUARD / .5)
    np.testing.assert_allclose(rest[7:11], moves.mirror(np, moves.HIGH_GUARD, 1) / .5)
    arm.launch(0, 1., np.random.default_rng(0), moves.HOOK)
    arm.speed = 1.0
    poses = [arm.actions(high=False)[:4] * .5 for _ in range(40)]
    extend, retract = moves.durations(np, moves.HOOK, .02)
    # Step 4 of 13 is 5/13 of the extension: 96 % of the way from guard (zero) to the wind-up.
    np.testing.assert_allclose(poses[4], 5 / extend / moves.WINDUP_FRACTION * moves.WINDUP_HEAD,
                               atol=1e-9)
    assert np.abs(poses[int(extend) - 1] - moves.HOOK_HEAD).max() < .07   # jitter <= 0.06
    assert arm.ready and np.allclose(poses[-1], 0)


def test_moves_match_between_numpy_and_jax():
    for move in (moves.STRAIGHT, moves.HOOK):
        for age in (0., 3.4, 7., 12.5, 20.):
            for high in (False, True):
                rest = moves.rest_pose(np, np.array([-.2, .1, .1, -.5]), high)
                target = moves.strike_pose(np, move, .3)
                a = moves.punch_pose(np, move, .3, target, age, rest, .02)
                b = moves.punch_pose(jp, jp.asarray(move), jp.asarray(.3), jp.asarray(target),
                                     jp.asarray(age), jp.asarray(rest), .02)
                np.testing.assert_allclose(a, np.asarray(b), rtol=1e-5, atol=1e-5)


def test_widened_legacy_policy_acts_like_the_original():
    from boxer.tactics_env import policy_sizes, tactics_policy, widen_legacy
    legacy = load_policy_params("checkpoints/tactics-batched")[0]
    assert policy_sizes(legacy) == (26, 5)
    wide = widen_legacy(legacy)
    assert policy_sizes(wide) == (OBS_SIZE, ACTION_SIZE)
    old, new = tactics_policy(legacy, True), tactics_policy(wide, True)
    obs = jax.random.normal(jax.random.PRNGKey(0), (16, OBS_SIZE))
    a, _ = old(obs, jax.random.PRNGKey(1))
    b, _ = new(obs, jax.random.PRNGKey(1))
    assert a.shape == b.shape == (16, ACTION_SIZE)
    np.testing.assert_allclose(np.asarray(a)[:, :5], np.asarray(b)[:, :5], atol=1e-5)
    # Legacy padding (-1) and the widened net's silent outputs (0) both mean
    # "straight punch, normal guard": only positive values switch.
    assert np.all(np.asarray(a)[:, 5:] == -1) and np.all(np.asarray(b)[:, 5:] <= 0)


def test_batched_bout_runs_with_finite_observations_and_scripted_tactics():
    motor, _ = load_policy_params("checkpoints/spar")
    env = TacticsEnv(motor, worlds_hint=2, episode_seconds=1.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    # Spawns are 0.9-1.7 m apart: outside punching range, by obs[3] = distance - reach.
    assert state.obs.shape == (OBS_SIZE,) and .9 - REACH <= float(state.obs[3]) <= 1.7 - REACH
    step = jax.jit(env.step)
    for t in range(50):
        action, wait = env._scripted(state.obs, jax.random.PRNGKey(t), state.info["scripted_wait"])
        state = step(state, action)
    assert np.isfinite(np.asarray(state.obs)).all()
    assert float(state.done) == 1.0          # one-second episode ended on the clock
    assert float(state.metrics["fell"]) == 0


def test_native_style_starts_and_an_opponent_pool():
    motor, _ = load_policy_params("checkpoints/spar")
    from boxer.tactics_env import tactics_network_factory
    from brax.training.acme import running_statistics
    net = tactics_network_factory()(OBS_SIZE, ACTION_SIZE)
    key = jax.random.PRNGKey(0)
    params = (running_statistics.init_state(jax.ShapeDtypeStruct((OBS_SIZE,), jp.float32)),
              net.policy_network.init(key), net.value_network.init(key))
    env = TacticsEnv(motor, [params, params], scripted_fraction=0.0, worlds_hint=2,
                     episode_seconds=.5, native_reset_fraction=1.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    assert abs(float(state.obs[2]) - 2.0) < 1e-3          # PairedFoundation spawn distance
    assert int(state.info["opponent"]) in (0, 1) and not bool(state.info["scripted"])
    step = jax.jit(env.step)
    for _ in range(10):
        state = step(state, jp.zeros(ACTION_SIZE))
    assert np.isfinite(np.asarray(state.obs)).all()


def test_native_scoring_counts_events_per_hand_and_rearms():
    motor, _ = load_policy_params("checkpoints/spar")
    env = TacticsEnv(motor, worlds_hint=2, episode_seconds=1.0, scoring="native")
    state = jax.jit(env.reset)(jax.random.PRNGKey(2))
    assert state.info["engaged"].shape == (2, 2)
    step = jax.jit(env.step)
    total = 0.
    for t in range(40):
        action, _ = env._scripted(state.obs, jax.random.PRNGKey(t), state.info["scripted_wait"])
        state = step(state, action)
        total += float(state.metrics["hits"])
        assert float(state.metrics["hits"]) in (0., 1., 2.)   # whole events, at most one per hand
    assert np.isfinite(np.asarray(state.obs)).all() and total >= 0


def test_domain_randomization_batches_physics_and_zero_strength_is_nominal():
    motor, _ = load_policy_params("checkpoints/spar")
    env = TacticsEnv(motor, worlds_hint=2)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    model, in_axes = env.randomizer(1.0)(env.mjx_model, keys)
    assert model.body_mass.shape == (3, env.mj_model.nbody) and in_axes.body_mass == 0
    ratio = np.asarray(model.body_mass) / np.maximum(env.mj_model.body_mass, 1e-9)
    light = env.mj_model.body_mass > 1.5            # +-1 kg on torsos dominates light links
    assert np.all(ratio[:, light] > .5) and np.all(ratio[:, light] < 1.6)
    friction = np.asarray(model.pair_friction)[:, :, 0]
    assert friction.min() >= .3 - 1e-6 and friction.max() <= 1.0 + 1e-6
    assert not np.allclose(friction[0], friction[1])
    nominal, _ = env.randomizer(0.0)(env.mjx_model, keys)
    np.testing.assert_allclose(np.asarray(nominal.actuator_gainprm)[0], env.mj_model.actuator_gainprm, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(nominal.body_mass)[1], env.mj_model.body_mass, rtol=1e-6)


def test_decode_stops_circling_and_turns_when_facing_away():
    circle = np.array([-1, -1, 0, 0, 1., -1, -1])
    facing, *_ = decode(circle, .6, np.array([.6, 0.]))
    behind, *_ = decode(circle, .6, .6 * np.array([np.cos(2.), np.sin(2.)]))
    assert abs(facing[1]) == pytest.approx(.2) and facing[2] == 0
    assert np.allclose(behind[:2], 0) and behind[2] == pytest.approx(.5)
