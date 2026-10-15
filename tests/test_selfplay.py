"""The self-play stack: one shared policy, twice as many learners as worlds."""

import jax
import mujoco
import numpy as np
import pytest
from jax import numpy as jp

from boxer import policy as policy_module
from boxer import selfplay
from boxer.env import NUM_AGENTS


def _keys(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


@pytest.fixture(scope="session")
def wrapped(env, obs_dim):
    return selfplay.wrap(env, episode_length=100, observation_dim=obs_dim)


def test_learners_outnumber_physics_worlds(wrapped, env, obs_dim):
    learners = 8
    state = jax.jit(wrapped.reset)(_keys(learners))
    assert state.obs.shape == (learners, obs_dim)
    assert state.reward.shape == (learners,)
    assert state.done.shape == (learners,)
    assert state.info["steps"].shape == (learners,)
    assert state.info["episode_metrics"]["sum_reward"].shape == (learners,)
    assert state.pipeline_state.qpos.shape[0] == learners // NUM_AGENTS


def test_step_keeps_shapes(wrapped, env, obs_dim):
    learners = 8
    state = jax.jit(wrapped.reset)(_keys(learners, 1))
    step = jax.jit(wrapped.step)
    action = jp.zeros((learners, env.action_size))
    for _ in range(3):
        state = step(state, action)
    assert state.obs.shape == (learners, obs_dim)
    steps = np.asarray(state.info["steps"])
    assert np.all(steps == 3)
    # Both corners of one world share a single round counter.
    per_world = steps.reshape(-1, NUM_AGENTS)
    assert np.all(per_world[:, 0] == per_world[:, 1])


def test_fold_unfold_roundtrip(env, obs_dim):
    """The two reshapes must be exact inverses or state drifts across steps."""
    inner = selfplay.AutoResetWrapper(
        selfplay.RoundEpisode(selfplay.VmapWrapper(env), 10)
    )
    state = jax.jit(inner.reset)(_keys(4, 2))
    unfolded = selfplay._unfold(state, obs_dim)
    back = selfplay._fold(unfolded, obs_dim)

    for leaf_a, leaf_b in zip(
        jax.tree_util.tree_leaves(back.pipeline_state),
        jax.tree_util.tree_leaves(state.pipeline_state),
    ):
        assert np.allclose(np.asarray(leaf_a), np.asarray(leaf_b))
    assert np.allclose(np.asarray(back.obs), np.asarray(state.obs))
    assert np.allclose(np.asarray(back.reward), np.asarray(state.reward))
    assert np.allclose(np.asarray(back.done), np.asarray(state.done))
    assert np.allclose(np.asarray(back.info["steps"]), np.asarray(state.info["steps"]))
    assert np.allclose(
        np.asarray(back.info["first_obs"]), np.asarray(state.info["first_obs"])
    )
    for name in state.metrics:
        assert np.allclose(np.asarray(back.metrics[name]), np.asarray(state.metrics[name]))


def test_round_times_out_and_restarts(env, obs_dim):
    short = selfplay.wrap(env, episode_length=6, observation_dim=obs_dim)
    learners = 4
    state = jax.jit(short.reset)(_keys(learners, 3))
    step = jax.jit(short.step)
    action = jp.zeros((learners, env.action_size))

    seen = []
    for _ in range(8):
        state = step(state, action)
        seen.append((
            np.asarray(state.done).copy(),
            np.asarray(state.info["steps"]).copy(),
            np.asarray(state.info["truncation"]).copy(),
        ))

    done, steps, truncation = seen[5]
    assert np.all(done == 1.0)
    assert np.all(steps == 6)
    assert np.all(truncation == 1.0)  # timed out, not knocked out
    done, steps, _ = seen[6]
    assert np.all(steps == 1)
    assert np.all(done == 0.0)


def test_episode_return_accumulates_per_learner(wrapped, env, obs_dim):
    learners = 4
    state = jax.jit(wrapped.reset)(_keys(learners, 4))
    step = jax.jit(wrapped.step)
    action = jp.zeros((learners, env.action_size))
    returns = []
    for _ in range(5):
        state = step(state, action)
        returns.append(np.asarray(state.info["episode_metrics"]["sum_reward"]).copy())
    # Holding the guard pose pays a positive reward every step, so the running
    # return must grow monotonically while the round is alive.
    deltas = np.diff(np.stack(returns), axis=0)
    assert np.all(deltas > 0)


def _qpos_address(env, name: str) -> int:
    model = env.sys.mj_model
    return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


@pytest.fixture(scope="session")
def frozen(level):
    """One learner per world, blue driven by the jabber script."""
    opponent = policy_module.jabber(
        level.action_size,
        pitch_adr=level.actuator_index("red/left_shoulder_pitch_joint"),
        elbow_adr=level.actuator_index("red/left_elbow_joint"),
    )
    return selfplay.wrap_frozen(level, opponent, episode_length=500, learner_side=0)


def test_frozen_mode_has_one_learner_per_physics_world(frozen, level):
    worlds = 6
    state = jax.jit(frozen.reset)(_keys(worlds, 7))
    assert state.obs.shape == (worlds, level.observation_size)
    assert state.reward.shape == (worlds,)
    assert state.done.shape == (worlds,)
    # Not folded: no doubling of the batch, and the physics state is per world.
    assert state.pipeline_state.qpos.shape[0] == worlds


def test_frozen_opponent_actually_moves(frozen, level):
    """The wrapper must feed the opponent's actions into its corner.

    Measured as the largest excursion over the round rather than the last frame: a
    jab is a clipped sine, so the arm sits back at the guard for half of every cycle,
    and a single endpoint sample can land exactly in that rest and report a motionless
    opponent that was swinging a moment ago.
    """
    learner_red = _qpos_address(level, "red/left_shoulder_pitch_joint")
    opponent_blue = _qpos_address(level, "blue/left_shoulder_pitch_joint")
    stance = np.asarray(level._stance_qpos)

    state = jax.jit(frozen.reset)(_keys(4, 8))
    step = jax.jit(frozen.step)
    action = jp.zeros((4, level.action_size))
    moved = held = 0.0
    for _ in range(30):
        state = step(state, action)
        qpos = np.asarray(state.pipeline_state.qpos)
        moved = max(moved, float(np.abs(qpos[:, opponent_blue] - stance[opponent_blue]).max()))
        held = max(held, float(np.abs(qpos[:, learner_red] - stance[learner_red]).max()))

    assert moved > 0.05, "the scripted opponent should have swung its arm"
    assert held < moved, "the learner's own corner must not be driven"


def test_frozen_opponent_receives_its_own_observation_vector(level):
    """An opponent that reads its observation must get a (worlds, D) slice.

    The wrapper projects the learner's corner out, and Brax hands that projected
    state straight back into the next step -- so the second step onward used to
    slice an already-projected array and hand the opponent a (worlds,) vector.
    Scripted opponents never noticed because they only read ``obs.shape[0]``; a
    trained checkpoint subtracts its observation normaliser and died.
    """
    seen = []

    def opponent(obs, time, key):
        seen.append(tuple(obs.shape))
        return 0.05 * jp.clip(obs[:, :level.action_size], -1.0, 1.0)

    worlds = 4
    env = selfplay.wrap_frozen(level, opponent, episode_length=500, learner_side=0)
    state = jax.jit(env.reset)(_keys(worlds, 12))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jp.zeros((worlds, level.action_size)))

    assert seen, "the opponent was never asked to act"
    assert seen[0] == (worlds, level.observation_size)
    assert np.asarray(state.obs).shape == (worlds, level.observation_size)


def test_frozen_mode_keeps_per_side_reward_bookkeeping(frozen):
    worlds = 4
    state = jax.jit(frozen.reset)(_keys(worlds, 9))
    step = jax.jit(frozen.step)
    for _ in range(5):
        state = step(state, jp.zeros((worlds, frozen.action_size)))
    episode = state.info["episode_metrics"]
    assert episode["sum_reward_red"].shape == (worlds,)
    assert episode["sum_reward_blue"].shape == (worlds,)
    # Red is the learner and gets its own column; blue keeps the other one.
    assert not np.allclose(np.asarray(episode["sum_reward_red"]),
                           np.asarray(episode["sum_reward_blue"]))
