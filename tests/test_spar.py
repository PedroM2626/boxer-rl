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

    touching = range(0)

    def bodies_touching(self):
        return self.t in self.touching

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


def test_referee_breaks_clinches_after_the_set_time_without_counting_anything():
    from boxer.spar_match import bout
    motor = lambda obs, key: (np.zeros((2, 29)), None)
    pair = _FallingPair({})
    pair.touching = range(5, 40)              # 0.7 s of unbroken body contact
    free, _, _ = bout(pair, motor, 0, 1.0)
    assert free['clinch_breaks'] == [] and free['clinch_seconds'] == pytest.approx(.7)
    pair = _FallingPair({})
    pair.touching = range(5, 40)
    broken, _, _ = bout(pair, motor, 0, 1.0, clinch_break=.3)
    assert len(broken['clinch_breaks']) == 2 and pair.resets == 3
    assert broken['knockdowns_suffered'] == [0, 0] and broken['restarts'] == []


class _CountedPair(_FallingPair):
    """Red goes down at step 10 and stands from ``up_at`` on; the bout continues."""

    floor_body, stop_on_fall, done = True, True, False

    def __init__(self, up_at):
        super().__init__({10: (True, False)})
        self.up_at, self.actions = up_at, []

    def step(self, action):
        self.actions.append(np.array(action))
        return super().step(action)

    def standing(self):
        return np.array([self.t >= self.up_at, True])

    def fallen(self):
        return np.zeros(2, bool)


def test_with_a_getup_policy_the_referee_counts_instead_of_restarting():
    from boxer.spar_match import COUNT_STEADY, bout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    getup = lambda inputs, key: (np.full((2, 29), 7.), None)
    pair = _CountedPair(up_at=40)
    result, _, _ = bout(pair, motor, 0, 3.0, knockdowns=3, getup=getup, mandatory_count=0.)
    assert pair.resets == 1 and result['restarts'] == []          # only the opening reset
    assert result['knockdowns_suffered'] == [1, 0] and result['method'] != 'ko'
    [count] = result['counts']
    assert count['fallen'] == [True, False]
    # Down at 0.22 s (after step 10); standing from step 40, steady for COUNT_STEADY.
    assert count['seconds_down'] == pytest.approx((40 + COUNT_STEADY / .02) * .02 - .22, abs=.03)
    # Only the fallen boxer's legs and arms come from the get-up policy, and only while down.
    counted = [a for a in pair.actions if a[0, 0] == 7.]
    assert len(counted) == pytest.approx((count['seconds_down']) / .02, abs=2)
    assert all(a[1, 0] == 0. for a in pair.actions)
    assert pair.actions[-1][0, 0] == 0. and pair.stop_on_fall


def test_the_mandatory_count_keeps_a_risen_boxer_waiting():
    from boxer.spar_match import bout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    getup = lambda inputs, key: (np.full((2, 29), 7.), None)
    pair = _CountedPair(up_at=40)
    result, _, _ = bout(pair, motor, 0, 4.0, knockdowns=3, getup=getup, mandatory_count=2.)
    assert result['counts'][0]['seconds_down'] == pytest.approx(2., abs=.03)


def test_a_boxer_still_down_after_the_count_is_knocked_out():
    from boxer.spar_match import bout
    motor = lambda inputs, key: (np.zeros((2, 29)), None)
    getup = lambda inputs, key: (np.zeros((2, 29)), None)
    pair = _CountedPair(up_at=10 ** 6)
    result, _, _ = bout(pair, motor, 0, 5.0, knockdowns=3, getup=getup, count_seconds=1.)
    assert result['method'] == 'ko' and result['winner'] == 'blue'
    assert result['counted_out'] == [True, False] and result['seconds'] == pytest.approx(1.22, abs=.03)


def test_getting_up_needs_floor_colliders():
    from boxer.spar_match import bout
    pair = _FallingPair({})
    pair.floor_body = False
    with pytest.raises(ValueError, match="floor_body"):
        bout(pair, lambda i, k: (np.zeros((2, 29)), None), 0, 1.0, getup=lambda i, k: None)


def test_corner_motors_take_each_row_from_that_corners_motor():
    from boxer.spar_match import corner_motors, motor_actions
    red = lambda inputs, key: (inputs['state'][:, :3] + 1, None)
    blue = lambda inputs, key: (inputs['state'][:, :3] - 1, None)
    obs = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.testing.assert_array_equal(motor_actions(corner_motors(red, blue), obs), [[1, 2, 3], [5, 6, 7]])
    np.testing.assert_array_equal(motor_actions(corner_motors(red, red), obs), motor_actions(red, obs))


def test_rapid_punches_throw_back_to_back_with_bounded_corrections():
    from boxer.curriculum_env import RAPID_PUNCH_RESIDUAL

    def punching_share(config):
        env = UnassistedG1("spar", config=config)
        info = dict(jax.jit(env.reset)(jax.random.PRNGKey(0)).info)
        generate = jax.jit(lambda info: (env.punch_arms(info), info)[1])
        busy, residuals = 0, []
        for _ in range(400):
            info = generate(info)
            busy += float(info["punch_age"]) >= 0
            residuals.append(np.asarray(info["punch_residual"]))
        return busy / 400, np.abs(residuals)

    normal, normal_residual = punching_share(stage_config("spar"))
    config = stage_config("spar")
    config.rapid_punches = True
    rapid, rapid_residual = punching_share(config)
    assert rapid > normal + .25 and rapid > .7
    assert normal_residual.max() == 0
    assert 0 < rapid_residual.max() <= RAPID_PUNCH_RESIDUAL


def test_striker_arm_traces_replay_in_a_share_of_episodes(tmp_path):
    traces = np.random.default_rng(0).uniform(-1, 1, (1200, 14)).astype(np.float32)
    traces[:, [4, 5, 6, 11, 12, 13]] = 0
    np.savez(tmp_path / "arms.npz", arms=traces)
    config = stage_config("spar")
    config.striker_arms = str(tmp_path / "arms.npz")
    env = UnassistedG1("spar", config=config)
    generate = jax.jit(lambda info: (env.punch_arms(info), info))
    replayed = 0
    for seed in range(8):
        info = dict(jax.jit(env.reset)(jax.random.PRNGKey(seed)).info)
        start = int(info["trace_t"])
        assert 0 <= start < len(traces) - config.episode_length
        arms, after = generate(info)
        arms = np.asarray(arms)
        if bool(info["trace_on"]):
            replayed += 1
            np.testing.assert_allclose(arms, traces[start])
            assert int(after["trace_t"]) == start + 1
    assert 0 < replayed < 8
    config.striker_arms = str(tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError):
        UnassistedG1("spar", config=config)


def test_videos_resample_recorded_states_without_blending_restarts():
    import mujoco
    from boxer.spar_match import resample
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>"
        "<body><joint type='hinge'/><geom size='.1'/></body></worldbody></mujoco>")
    frames, times = [], []
    for k in range(6):                                   # 50 Hz, restart after the 4th state
        angle = .1 * k
        qpos = np.array([k, 0, 0, np.cos(angle), 0, 0, np.sin(angle), k * .5])
        frames.append(qpos)
        times.append(((k + 1) * .02, np.zeros(2), np.array([0., float(k >= 4)])))
    out, out_times = resample(model, frames, times, 60)
    assert len(out) == 7                                 # 0.02 .. 0.12 s at 1/60 s
    np.testing.assert_allclose([t[0] for t in out_times], .02 + np.arange(7) / 60)
    np.testing.assert_allclose(out[0], frames[0])
    np.testing.assert_allclose(out[1][0], 50 / 60, atol=1e-9)   # 0.0367 s, between states 0 and 1
    for qpos in out:
        assert np.linalg.norm(qpos[3:7]) == pytest.approx(1)
    held = [q for q, t in zip(out, out_times) if .08 <= t[0] < .1]   # between states 3 and 4: a restart
    assert held and all(np.array_equal(q, frames[3]) for q in held)
