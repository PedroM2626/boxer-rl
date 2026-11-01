"""The sparring stage drives the arms; the policy keeps legs, waist and balance."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer import moves
from boxer.curriculum_control import STAGES, TRAINABLE_STAGES, commands_for
from boxer.curriculum_env import (LEFT_ARM, PUNCH_EXTEND, PUNCH_RETRACT,
                                  UnassistedG1, stage_config)
from boxer.spar_match import PunchArm, RingBrain


def test_spar_stays_out_of_the_automatic_locomotion_sequence():
    assert "spar" not in STAGES and "spar" in TRAINABLE_STAGES
    assert commands_for("spar")[0] == (0.0, 0.0, 0.0) and len(commands_for("spar")) == 7
    config = stage_config("spar")
    assert config.push_config.enable and config.push_config.magnitude_range[1] > .3


def test_generated_punches_override_only_the_arms_and_return_to_guard():
    env = UnassistedG1("spar")
    state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    step = jax.jit(env.step)
    policy_arms = jp.full(29, .3)
    moved, launched = False, False
    for _ in range(120):
        state = step(state, policy_arms)
        applied = np.asarray(state.info["last_act"])
        np.testing.assert_allclose(applied[:LEFT_ARM], .3)   # legs/waist untouched
        launched |= float(state.info["punch_age"]) >= 0
        moved |= np.abs(applied[LEFT_ARM:]).max() > 1.0
    assert launched and moved
    # Between punches, the arms hold the normal guard (zero offset) or the high guard.
    while float(state.info["punch_age"]) >= 0:
        state = step(state, policy_arms)
    state = step(state, policy_arms)
    if float(state.info["punch_age"]) < 0:
        arms = np.asarray(state.info["last_act"])[LEFT_ARM:]
        if bool(state.info["high_guard"]):
            guard = np.asarray(env._default_pose)[LEFT_ARM:LEFT_ARM + 4]
            np.testing.assert_allclose(arms[:4], (moves.HIGH_GUARD - guard) / .5, atol=1e-5)
        else:
            np.testing.assert_allclose(arms, 0, atol=1e-6)


def test_generator_throws_hooks_and_raises_the_guard():
    env = UnassistedG1("spar")
    state = jax.jit(env.reset)(jax.random.PRNGKey(5))
    step = jax.jit(env.step)
    seen_moves, seen_high = set(), set()
    for _ in range(600):
        state = step(state, jp.zeros(29))
        if float(state.info["punch_age"]) >= 0:
            seen_moves.add(int(state.info["punch_move"]))
        seen_high.add(bool(state.info["high_guard"]))
    assert seen_moves == {moves.STRAIGHT, moves.HOOK} and seen_high == {False, True}


def test_match_punch_arm_extends_one_hand_and_retracts():
    pose = np.zeros(29)
    arm = PunchArm(pose, .5, .02)
    arm.launch(1, 1.0, np.random.default_rng(0))
    peaks = []
    while not arm.ready:
        a = arm.actions()
        assert np.all(a[:7] == 0)                      # left arm stays at guard
        peaks.append(np.abs(a[7:11]).max())
    assert len(peaks) <= int(np.ceil((PUNCH_EXTEND + PUNCH_RETRACT) / .02 / .9)) + 1
    assert max(peaks) > 2 and peaks[-1] < max(peaks) / 3


def test_ring_brain_closes_distance_then_punches_only_in_range():
    brain = RingBrain(np.random.default_rng(0))
    arm = PunchArm(np.zeros(29), .5, .02)
    command, punch = brain.decide(1.6, np.array([1.6, 0.]), arm)
    assert command[0] > .2 and punch is None
    punches = [brain.decide(.6, np.array([.6, 0.]), arm)[1] for _ in range(60)]
    assert any(p is not None for p in punches)


class _FallingPair:
    """Minimal stand-in for PairedFoundation whose boxers fall on schedule."""

    dt, action_scale, pose = .02, 1., np.zeros(29)
    self_trip_falls = True

    def __init__(self, falls):
        self.falls, self.t, self.resets = falls, 0, 0
        self.maps = [dict(root=0, side='red/'), dict(root=7, side='blue/')]

        class Data:
            qpos = np.zeros(14)

            @staticmethod
            def site(name):
                return type('Site', (), {'xmat': np.eye(3).ravel()})
        self.data = Data()
        self.data.qpos[7] = .6

    def reset(self, seed=0):
        self.resets += 1

    def observation(self, commands):
        return np.zeros((2, 4), np.float32)

    def on_ropes(self):
        return np.zeros(2, bool)

    def step(self, action):
        fallen = np.array(self.falls.get(self.t, (False, False)))
        self.t += 1
        return dict(done=fallen.any(), fallen=fallen, scored=np.zeros(2))


def test_knockdowns_restart_the_bout_until_the_stopping_count():
    from boxer.spar_match import KNOCKDOWN_POINTS, bout
    motor = lambda obs, key: (np.zeros((2, 29)), None)
    falls = {10: (False, True), 20: (False, True), 30: (True, False)}
    original, _, _ = bout(_FallingPair(falls), motor, 0, 1.0)
    assert original['method'] == 'knockdown' and original['winner'] == 'red'
    assert original['seconds'] == pytest.approx(.22)
    pair = _FallingPair(falls)
    result, _, _ = bout(pair, motor, 0, 1.0, knockdowns=3)
    assert result['method'] == 'points' and result['seconds'] == pytest.approx(1.0)
    assert result['knockdowns_suffered'] == [1, 2] and len(result['restarts']) == 3
    assert result['cards'] == [2 * KNOCKDOWN_POINTS, KNOCKDOWN_POINTS] and result['winner'] == 'red'
    assert pair.resets == 4
    stopped, _, _ = bout(_FallingPair(falls), motor, 0, 1.0, knockdowns=2)
    assert stopped['method'] == 'tko' and stopped['winner'] == 'red' and stopped['fallen'] == [False, True]


def test_corner_motors_take_each_row_from_that_corners_motor():
    from boxer.spar_match import corner_motors, motor_actions
    red = lambda inputs, key: (inputs['state'][:, :3] + 1, None)
    blue = lambda inputs, key: (inputs['state'][:, :3] - 1, None)
    obs = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.testing.assert_array_equal(motor_actions(corner_motors(red, blue), obs), [[1, 2, 3], [5, 6, 7]])
    np.testing.assert_array_equal(motor_actions(corner_motors(red, red), obs), motor_actions(red, obs))
