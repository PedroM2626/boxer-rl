"""A striker drives footwork and both arms from one policy (boxer.striker)."""
import jax
import numpy as np
import pytest

from boxer import moves
from boxer.curriculum_env import LEFT_ARM, UnassistedG1, stage_config
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import PunchArm, bout
from boxer.striker import (POSE_HIGH, POSE_LOW, RATE, STRIKER_ACTION_SIZE, STRIKER_OBS_SIZE,
                           StrikerArm, StrikerBrain, from_unit, striker_net, striker_observe, to_unit)


@pytest.fixture(scope='module')
def pair():
    config = stage_config('approach', impl='jax')
    config.noise_config.level = 0.
    return PairedFoundation(UnassistedG1('approach', config=config), ring=True)


def test_pose_box_holds_every_scripted_pose_and_round_trips():
    for pose in (moves.HIGH_GUARD, moves.STRAIGHT_HEAD, moves.STRAIGHT_BODY, *moves.HOOK_PATH_HEAD):
        assert (pose > POSE_LOW).all() and (pose < POSE_HIGH).all()
        np.testing.assert_allclose(from_unit(to_unit(pose)), pose, atol=1e-12)


def test_striker_arm_replays_executed_arm_actions_and_limits_its_rate(pair):
    scripted = PunchArm(pair.pose, pair.action_scale, pair.dt)
    scripted.launch(1, .7, np.random.default_rng(0), moves.HOOK)
    arm = StrikerArm(pair.pose, pair.action_scale, pair.dt)
    np.testing.assert_allclose(arm.actions(), 0, atol=1e-12)          # starts in the guard
    assert not arm.features()[1]
    seen_active = False
    for _ in range(12):
        executed = scripted.actions(False)
        arm.follow(executed)
        np.testing.assert_allclose(arm.actions(), executed, atol=1e-12)
        seen_active |= bool(arm.features()[1]) and arm.features()[2] == 1.
    assert seen_active                                                 # right hand under way
    arm = StrikerArm(pair.pose, pair.action_scale, pair.dt)
    start = arm.arm_pose.copy()
    arm.command(np.ones(8))
    assert np.abs(arm.arm_pose - start).max() == pytest.approx(RATE)


def test_a_striker_corner_boxes_a_short_native_bout(pair):
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    net, _ = striker_net()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((STRIKER_OBS_SIZE,), jp.float32))
    brain = StrikerBrain((norm, net.policy_network.init(k1), net.value_network.init(k2)))
    pair.reset(seed=0)
    arms = [brain.new_arm(pair), PunchArm(pair.pose, pair.action_scale, pair.dt)]
    obs = striker_observe(pair, 0, arms, 0.)
    assert obs.shape == (STRIKER_OBS_SIZE,) and np.isfinite(obs).all()
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    result, _, _ = bout(pair, motor, 0, .4, brains={0: brain})
    assert result['seconds'] > 0 and len(result['punches_thrown']) == 2
    assert len(result['arm_motion_rad']) == 2 and min(result['arm_motion_rad']) >= 0
    assert STRIKER_ACTION_SIZE == 2 + 8


def test_punch_shapes_split_scripted_paths_from_new_ones():
    import importlib.util
    spec = importlib.util.spec_from_file_location("punch_shapes", "tools/punch_shapes.py")
    shapes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shapes)
    guard = np.zeros(3)
    points = dict(straight=(np.zeros((1, 3)), np.array([[.4, 0., .1]])),
                  hook=(np.array([[.3, .3, .1]]), np.array([[.35, -.05, .1]])))
    assert shapes.segment_distance(np.array([.2, 0., .05]), *points["straight"]) == pytest.approx(0)
    stopped_short = np.array([.2, 0., .05])            # on the straight's path
    assert shapes.classify(stopped_short - guard, stopped_short, .0, points) == "straight"
    assert shapes.classify(np.array([.32, .12, .1]), np.array([.32, .12, .1]), .25, points) == "hook"
    rising = np.array([.05, 0., .3])
    assert shapes.classify(rising, rising, 0., points) == "new up"
    path = np.array([[0, 0, 0], [.2, 0, .05], [.4, 0, .1], [.2, 0, .05], [0, 0, 0]], float)
    found = shapes.punches(path, np.zeros((2, 3)), points)
    assert len(found) == 1 and found[0]["kind"] == "straight" and found[0]["reach"] == pytest.approx(.4)


def test_native_striker_pays_for_moving_its_arm_targets(pair):
    from boxer.native_finetune import NativeBout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    rewards, motions = [], []
    for cost in (0., 1.):
        bout = NativeBout(pair, motor, np.random.default_rng(0), [], 1., 50, 30., striker=True,
                          arm_motion_cost=cost)
        start = bout.arms[bout.learner].arm_pose.copy()
        action = np.concatenate([[0., 0.], to_unit(start).ravel() + .2])
        reward, _, _, info = bout.step(action)
        moved = np.abs(bout.arms[bout.learner].arm_pose - start).sum()
        assert info["motion"] == pytest.approx(moved) and moved > 0
        held = bout.step(np.concatenate([[0., 0.], to_unit(bout.arms[bout.learner].arm_pose).ravel()]))
        assert held[3]["motion"] == pytest.approx(0, abs=1e-9)
        rewards.append(reward)
        motions.append(moved)
    assert rewards[0] - rewards[1] == pytest.approx(motions[1])


def test_a_striker_can_be_the_native_opponent(pair):
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from boxer.native_finetune import NativeBout
    net, _ = striker_net()
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((STRIKER_OBS_SIZE,), jp.float32))
    opponent = StrikerBrain((norm, net.policy_network.init(k1), net.value_network.init(k2)))
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    bout = NativeBout(pair, motor, np.random.default_rng(0), [opponent], 0., 50, 30., striker=True)
    assert bout.opponent is opponent and all(isinstance(arm, StrikerArm) for arm in bout.arms)
    reward, terminal, truncated, info = bout.step(np.zeros(STRIKER_ACTION_SIZE))
    assert np.isfinite(reward)
