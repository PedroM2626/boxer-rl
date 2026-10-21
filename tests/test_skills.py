"""Physical checks for assisted boxing; run in the mujoco_warp environment."""
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from boxer.skills_env import SkillsBoxing, scripted_actions
from boxer.warp_config import make_env


def build(**kwargs):
    return SkillsBoxing(nworld=4, spawn_jitter=0, yaw_jitter=0,
                        pose_noise=0, **kwargs)


def test_idle_boxers_stand_without_scoring():
    env = build()
    obs = env.reset()
    start = env.physics._qpos_sync.copy()
    assert obs.shape == (4, 2, env.obs_size)
    for _ in range(300):
        obs, reward, done, parts = env.step(np.zeros((4, 2, 6)))
        assert not done.any()
        assert not parts["scored_by"].any()
        assert not parts["on_floor"].any()
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
    # The support is compliant: the solver still moves the roots under load.
    assert np.max(np.abs(env.physics._qpos_sync - start)) > .001


def test_both_corners_land_and_retract_without_falling():
    env = build()
    env.reset()
    landed = np.zeros(2)
    away = np.zeros(2)
    home = np.full(2, np.inf)
    for t in range(600):
        _, _, _, parts = env.step(scripted_actions(env, t))
        landed += (parts["scored_by"] > 0).sum(axis=0)
        away = np.maximum(away, parts["at_home"].max(axis=0))
        if t > 50:
            home = np.minimum(home, parts["at_home"].max(axis=0))
        assert not parts["on_floor"].any()
    assert (landed >= 12).all(), landed
    assert (away > .15).all(), away
    assert (home < .05).all(), home


def test_stun_prevents_launch_and_nan_is_rejected():
    env = build()
    env.reset()
    env.physics._stun[:] = 10
    action = np.zeros((4, 2, 6))
    action[..., 0] = 1
    _, _, _, parts = env.step(action)
    assert not parts["launched"].any()
    assert (env.age == -1).all()
    action[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        env.step(action)


def test_timeout_resets_only_finished_worlds_and_keeps_terminal_render_state():
    env = build(round_steps=40)
    env.reset()
    env.physics.capture_render_state = True
    env.physics._steps[:] = [39, 0, 39, 0]
    _, _, done, parts = env.step(scripted_actions(env, 0))
    np.testing.assert_array_equal(done, [True, False, True, False])
    assert (env.age[done] == -1).all()
    assert (env.age[~done] >= 0).all()
    assert not env.last_action[done].any()
    assert (parts["damage"] == 0).all()
    assert not np.shares_memory(env.physics.render_qpos, env.physics._qpos_sync)
    assert not env.physics._steps[done].any()
    assert (env.physics._steps[~done] == 1).all()


def test_config_factory_restores_the_control_space_and_health_rules():
    env = make_env(dict(environment="skills", damage_cap=7, stun_steps=13,
                        round_steps=120, spawn_distance=.64), nworld=2)
    assert env.action_size == 6
    assert env._damage_cap == 7 and env._stun_steps == 13
    assert env._round_steps == 120 and env._spawn_distance == .64
    # The compact guard uses the robot's supported negative elbow range; it
    # reaches above the sternum without bending the wrists to their stops.
    for side in ("red/", "blue/"):
        for hand in ("left", "right"):
            model = env.physics._model
            joint = model.joint(f"{side}{hand}_elbow_joint")
            angle = env.physics._qpos0[joint.qposadr][0]
            assert joint.range[0] <= angle <= joint.range[1]
            frame = env.physics._host_frame(side)
            assert frame[f"{hand}_palm"][2] - frame["torso"][2] > .20
    with pytest.raises(ValueError, match="unknown environment"):
        make_env({"environment": "missing"}, nworld=2)
