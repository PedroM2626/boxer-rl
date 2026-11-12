"""One policy for legs, waist and arms (boxer.whole_body)."""
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics

from boxer.curriculum_env import LEFT_ARM, UnassistedG1, stage_config
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import PunchArm, bout
from boxer.striker import StrikerArm, to_unit
from boxer.whole_body import (LEGS, WHOLE_BODY_ACTION_SIZE, WHOLE_BODY_OBS_SIZE, WholeBodyBrain, apply_arms,
                              body_observation, is_whole_body, whole_body_net, whole_body_observe)


@pytest.fixture(scope='module')
def pair():
    config = stage_config('approach', impl='jax')
    config.noise_config.level = 0.
    return PairedFoundation(UnassistedG1('approach', config=config), ring=True)


def random_params(seed=0):
    net, _ = whole_body_net()
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((WHOLE_BODY_OBS_SIZE,), jp.float32))
    return norm, net.policy_network.init(k1), net.value_network.init(k2)


def test_observation_joins_the_striker_view_and_the_body_without_its_command(pair):
    pair.reset(seed=0)
    arms = [StrikerArm(pair.pose, pair.action_scale, pair.dt), PunchArm(pair.pose, pair.action_scale, pair.dt)]
    obs = whole_body_observe(pair, 0, arms, 0.)
    assert obs.shape == (WHOLE_BODY_OBS_SIZE,) and np.isfinite(obs).all()
    full = pair.observation(np.ones((2, 3)))[1]
    np.testing.assert_allclose(body_observation(pair, 1), np.r_[full[:9], full[12:]], atol=1e-6)
    assert LEGS == LEFT_ARM and WHOLE_BODY_ACTION_SIZE == LEGS + 8


def test_arm_outputs_move_the_arms_and_leg_outputs_pass_through(pair):
    arm = StrikerArm(pair.pose, pair.action_scale, pair.dt)
    action = np.r_[np.full(LEGS, .2), to_unit(arm.arm_pose).ravel()]
    np.testing.assert_allclose(apply_arms(arm, action), .2)
    np.testing.assert_allclose(arm.actions(), 0, atol=1e-12)       # asked to hold the guard


def test_a_whole_body_corner_sets_its_own_legs_in_a_bout(pair):
    params = random_params()
    assert is_whole_body(params)
    brain = WholeBodyBrain(params)
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    result, _, _ = bout(pair, motor, 0, .2, brains={0: brain})
    assert result['seconds'] > 0
    assert np.abs(brain.legs).max() > 0 and np.abs(pair.last_action[0, :LEGS] - brain.legs).max() < 1e-6
    assert np.abs(pair.last_action[1, :LEGS]).max() == 0          # the scripted corner keeps the motor's


def test_native_bout_learns_the_whole_body(pair):
    from boxer.native_finetune import NativeBout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    bout_ = NativeBout(pair, motor, np.random.default_rng(0), [], 1., 50, 30., whole_body=True)
    assert bout_.observe().shape == (WHOLE_BODY_OBS_SIZE,)
    action = np.r_[np.full(LEGS, .3), to_unit(bout_.arms[bout_.learner].arm_pose).ravel()]
    reward, terminal, truncated, info = bout_.step(action)
    np.testing.assert_allclose(bout_.executed[bout_.learner, :LEGS], .3)
    assert np.abs(bout_.executed[1 - bout_.learner, :LEGS]).max() == 0 and np.isfinite(reward)


def test_tilt_penalty_uses_the_motors_upright_weights():
    from boxer.native_finetune import tilt_penalty
    obs = np.zeros(103)
    obs[6:9] = [0., 0., -1.]                                  # upright, still
    assert tilt_penalty(obs) == 0.
    obs[6:9] = [.3, .4, -.87]
    obs[3:6] = [1., 2., 5.]                                   # yaw rate costs nothing
    assert np.isclose(tilt_penalty(obs), 2. * .25 + .15 * 5.)


def test_tilt_cost_lowers_the_whole_body_reward_by_the_learners_tilt(pair):
    from boxer.native_finetune import NativeBout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    rewards, tilts = [], []
    for cost in (0., 5.):
        bout_ = NativeBout(pair, motor, np.random.default_rng(0), [], 1., 50, 30., whole_body=True, tilt_cost=cost)
        action = np.r_[np.full(LEGS, .3), to_unit(bout_.arms[bout_.learner].arm_pose).ravel()]
        for _ in range(5):
            reward, _, _, info = bout_.step(action)
        rewards.append(reward)
        tilts.append(info["tilt"])
    assert tilts[0] == 0. and tilts[1] > 0.
    assert np.isclose(rewards[0] - rewards[1], pair.dt * 5. * tilts[1], rtol=1e-6)


def test_leg_drift_measures_only_the_leg_outputs():
    from boxer.native_finetune import leg_drift
    net, _ = whole_body_net()
    norm, policy, _ = random_params()
    obs = jp.asarray(np.random.default_rng(0).normal(size=(8, WHOLE_BODY_OBS_SIZE)), jp.float32)
    logits = net.policy_network.apply(norm, policy, obs)
    assert float(leg_drift(net, norm, logits, policy, obs)) == 0.
    moved = logits.at[:, LEGS:].add(1.)                        # arm means and every scale
    moved = moved.at[:, WHOLE_BODY_ACTION_SIZE:].add(1.)
    assert float(leg_drift(net, norm, moved, policy, obs)) < 1e-12
    assert float(leg_drift(net, norm, logits.at[:, :LEGS].add(.5), policy, obs)) > 1e-3
