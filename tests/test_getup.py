import jax
import jax.numpy as jp
import numpy as np

from boxer.curriculum_control import STAGES, TRAINABLE_STAGES
from boxer.curriculum_env import FLOOR_BODY_GEOMS, GETUP_LYING, UnassistedG1, stage_config


def test_getup_is_trainable_but_outside_the_locomotion_sequence():
    assert "getup" in TRAINABLE_STAGES and "getup" not in STAGES
    config = stage_config("getup")
    assert not config.push_config.enable
    assert config.lin_vel_x == [0.0, 0.0] and config.ang_vel_yaw == [0.0, 0.0]


def test_floor_colliders_touch_only_the_floor_and_leave_the_robot_unchanged():
    env, standing = UnassistedG1("getup"), UnassistedG1("balance")
    m = env.mj_model
    assert m.nbody == standing.mj_model.nbody and m.nu == standing.mj_model.nu
    np.testing.assert_allclose(m.body_mass, standing.mj_model.body_mass)
    floor = m.geom("floor").id
    for _, name, *_ in FLOOR_BODY_GEOMS:
        geom = m.geom(name)
        assert geom.contype == 0 and geom.conaffinity == 0
        partners = {int(b) if int(a) == geom.id else int(a)
                    for a, b in zip(m.pair_geom1, m.pair_geom2) if geom.id in (int(a), int(b))}
        assert partners == {floor}


def test_fallen_starts_rest_on_the_floor_without_ending_the_episode():
    env = UnassistedG1("getup")
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(3), 32))
    lying = np.asarray(state.metrics["getup_start_lying"]) > 0
    assert 0 < lying.sum() < 32 and abs(lying.mean() - GETUP_LYING) < .3
    assert np.all(np.asarray(state.info["command"]) == 0)
    step = jax.jit(jax.vmap(env.step))
    for _ in range(50):
        state = step(state, jp.zeros((32, env.action_size)))
    z = np.asarray(state.data.qpos[:, 2])
    assert np.all(np.isfinite(z)) and z.min() > .05      # on the floor, not through it
    assert np.all(z[lying] < .4)
    assert not np.asarray(state.done).any()


def test_getup_reward_prefers_standing_in_guard_to_lying():
    env = UnassistedG1("getup")
    standing = UnassistedG1("balance").reset(jax.random.PRNGKey(0)).data
    standing = standing.replace(qpos=standing.qpos.at[7:].set(env._default_pose), qvel=jp.zeros_like(standing.qvel))
    lying = standing.replace(qpos=standing.qpos.at[2].set(.12).at[3:7].set(jp.array([.7071, .7071, 0., 0.])))
    action = jp.zeros(env.action_size)
    up, terms = env.getup_reward(standing, action)
    down, down_terms = env.getup_reward(lying, action)
    assert float(terms["standing"]) == 1. and float(down_terms["standing"]) == 0.
    assert float(down_terms["height"]) == 0. and float(terms["height"]) == 1.
    assert float(up) > float(down) + 2.5 * env.dt  # height and standing; qpos alone does not move the head


def test_getup_summary_counts_first_time_up_per_start_kind():
    from boxer.getup_eval import summarize
    heights = np.array([[.1, .8, .5], [.8, .8, .5], [.8, .3, .5]])
    upright = np.ones_like(heights)
    result = summarize(heights, upright, np.array([2, 0, 1]), dt=.5)
    assert {k: result["lying"][k] for k in ("worlds", "up_at_end", "ever_up", "seconds_to_up", "up_share_of_time")} == dict(
        worlds=1, up_at_end=1., ever_up=1., seconds_to_up=.5, up_share_of_time=2 / 3)
    assert result["lying"]["peak_height"] == .8 and result["lying"]["end_height"] == .8
    assert result["standing"]["up_at_end"] == 0. and result["standing"]["ever_up"] == 1.
    assert result["falling"]["ever_up"] == 0. and result["falling"]["seconds_to_up"] is None


def test_assist_force_lifts_the_torso_and_is_off_by_default():
    heights = {}
    for assist in (0., 1.5):
        config = stage_config("getup")
        config.getup_assist = assist
        env = UnassistedG1("getup", config=config)
        state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(5), 8))
        step = jax.jit(jax.vmap(env.step))
        for _ in range(40):
            state = step(state, jp.zeros((8, env.action_size)))
        heights[assist] = float(np.asarray(state.data.xpos[:, env._torso_body, 2]).mean())
    assert heights[1.5] > heights[0.] + .2
