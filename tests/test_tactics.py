"""Learned tactics see and act the same way in batched training and native bouts."""
import jax
import jax.numpy as jp
import numpy as np

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
        arm, other = rng.uniform(-1, 1, 4), rng.uniform(-1, 1, 4)
        clock = rng.uniform()
        native = observe(np, own, opp, arm, other, clock)
        batched = observe(jp, *(jax.tree.map(jp.asarray, (own, opp))), jp.asarray(arm),
                          jp.asarray(other), jp.asarray(clock))
        assert native.shape == (OBS_SIZE,)
        np.testing.assert_allclose(native, np.asarray(batched), rtol=1e-5, atol=1e-5)
        tactics = rng.uniform(-1.5, 1.5, ACTION_SIZE)
        distance, local = np.linalg.norm(opp["pos"] - own["pos"]), own["rot"].T @ (opp["pos"] - own["pos"])
        a = decode(tactics, distance, local)
        b = decode(jp.asarray(tactics), jp.asarray(distance), jp.asarray(local))
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float), rtol=1e-5, atol=1e-5)


def test_decoding_turns_toward_the_opponent_and_respects_motor_limits():
    command, want, hand, aim = decode(np.array([-1, 1, 1, 1, 0.]), 2.0, np.array([0., 2.]))
    assert command[2] == .5 and want and hand == 1 and aim == 1
    command, want, _, _ = decode(np.array([-1, -1, 0, -1, 1.]), .5, np.array([.5, 0.]))
    assert command[0] < 0 and command[1] > 0 and not want
    assert np.all(command >= [-.35, -.25, -.5]) and np.all(command <= [.55, .25, .5])


def test_native_arm_features_follow_a_punch():
    arm = PunchArm(np.zeros(29), .5, .02)
    np.testing.assert_allclose(arm.features(), 0)
    arm.launch(1, .25, np.random.default_rng(0))
    arm.actions()
    age, active, hand, aim = arm.features()
    assert 0 < age < 1 and active == 1 and hand == 1 and aim == .25


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
