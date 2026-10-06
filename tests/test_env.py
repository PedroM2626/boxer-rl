"""The duel env: shapes, jittability, and whether hits are real."""

import jax
import numpy as np
import pytest
from jax import numpy as jp

from boxer import scene
from boxer.env import NUM_AGENTS, BoxingEnv


def _rollout(env, state, action, length):
    def body(state, _):
        state = env.step(state, action)
        return state, state.reward
    return jax.lax.scan(body, state, None, length=length)


def test_reset_shapes(env):
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (NUM_AGENTS, env.observation_size)
    assert state.reward.shape == (NUM_AGENTS,)
    assert state.done.shape == ()
    assert np.all(np.isfinite(np.asarray(state.obs)))


def test_observation_size_matches_the_real_observation():
    """Guards the hand-maintained block count in ``BoxingEnv.observation_size``."""
    built = BoxingEnv()
    state = built.reset(jax.random.PRNGKey(0))
    assert state.obs.shape[-1] == built.observation_size


def test_step_runs_under_jit_and_stays_finite(env):
    action = jp.zeros((NUM_AGENTS, env.action_size))
    state, rewards = _rollout(env, env.reset(jax.random.PRNGKey(1)), action, 20)
    rewards = np.asarray(rewards)
    assert rewards.shape == (20, NUM_AGENTS)
    assert np.all(np.isfinite(rewards))
    assert np.all(np.isfinite(np.asarray(state.obs)))


def test_symmetric_agents_see_symmetric_worlds(level):
    """Both corners start in the same situation, so a shared policy is fair.

    This is what proves the observation really is egocentric: the two boxers are
    mirror images in world space, so any world-frame leak would show up here as
    a difference between the two observation vectors.
    """
    state = level.reset(jax.random.PRNGKey(2))
    obs = np.asarray(state.obs)
    reward = np.asarray(
        level.step(state, jp.zeros((NUM_AGENTS, level.action_size))).reward
    )
    assert obs[0] == pytest.approx(obs[1], abs=1e-4)
    assert reward[0] == pytest.approx(reward[1], abs=1e-4)


def test_no_phantom_hits_at_range(env):
    """A contact slot exists for every declared pair; only touching ones count."""
    state = env.reset(jax.random.PRNGKey(3))
    state = env.step(state, jp.zeros((NUM_AGENTS, env.action_size)))
    metrics = {k: float(np.asarray(v)) for k, v in state.metrics.items()}
    assert metrics["distance"] > 1.0, "the boxers must start out of reach"
    assert metrics["hits_landed"] == 0.0
    assert metrics["damage"] == 0.0
    assert metrics["head_shots"] == 0.0


def test_a_static_collision_is_not_a_punch(level):
    """Separates "touching" from "hitting", which is what the arena measures.

    Blue is dropped onto red's raised fists with both boxers holding still. The
    contact is real -- hand and forearm geoms interpenetrate -- but no fist is
    moving, so it must register as a contact and not as landed punches. Without
    this distinction a policy could score by falling onto its opponent, and the
    measured drift (1.3 m apart to 0.42 m while both stand passively) makes that
    an easy accident rather than a rare trick.
    """
    state = level.reset(jax.random.PRNGKey(4))
    pipeline = state.pipeline_state
    blue_free = level._sides[scene.BLUE]["free_adr"]
    state = state.replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[blue_free].add(-0.75))
    )

    state = level.step(state, jp.zeros((NUM_AGENTS, level.action_size)))
    metrics = {k: float(np.asarray(v)) for k, v in state.metrics.items()}
    assert metrics["contacts"] > 0.0, "the forced overlap should register contact"
    assert metrics["hits_landed"] == 0.0, "a stationary touch is not a punch"
    assert metrics["damage"] == 0.0


def test_a_moving_fist_that_connects_scores(level):
    """The other half of the gate: commitment turns contact into a punch."""
    from boxer import policy as policy_module

    jab = policy_module.jabber(
        level.action_size,
        pitch_adr=level.actuator_index("red/left_shoulder_pitch_joint"),
        roll_adr=level.actuator_index("red/left_shoulder_roll_joint"),
    )
    state = level.reset(jax.random.PRNGKey(6))
    pipeline = state.pipeline_state
    blue_free = level._sides[scene.BLUE]["free_adr"]
    state = state.replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[blue_free].add(-0.75))
    )

    scored = False
    for _ in range(40):
        action = jp.stack([
            jp.zeros(level.action_size),
            jab(state.obs[1][None], jp.array([float(state.pipeline_state.time)]),
                jax.random.PRNGKey(0))[0],
        ])
        state = level.step(state, action)
        if float(np.asarray(state.metrics["hits_landed"])) > 0.0:
            scored = True
            break
    assert scored, "a swinging fist that reaches the opponent should land a punch"


def test_contact_slot_mapping_is_static(level):
    """Guards the optimisation that folds per-geom roles into per-slot constants.

    ``BoxingEnv._slot_tables`` resolves each contact slot's role once and then
    evaluates hits with elementwise products on every physics substep. That is
    only correct while MJX's slot -> geom-pair mapping is independent of the
    state; if a MuJoCo change ever reorders the buffer, this fails instead of
    silently attributing punches to the wrong boxer.
    """
    state = level.reset(jax.random.PRNGKey(9))
    before = np.asarray(state.pipeline_state.contact.geom)
    action = jax.random.uniform(
        jax.random.PRNGKey(10), (NUM_AGENTS, level.action_size), minval=-1.0, maxval=1.0
    )
    state = jax.jit(level.step)(state, action)
    after = np.asarray(state.pipeline_state.contact.geom)
    assert before.shape == after.shape
    assert np.array_equal(before, after)


def test_a_tipped_boxer_ends_the_round(env):
    state = env.reset(jax.random.PRNGKey(5))
    pipeline = state.pipeline_state
    free = env._sides[scene.RED]["free_adr"]
    on_its_side = jp.array([0.7071068, 0.7071068, 0.0, 0.0])  # 90 degrees about x
    state = state.replace(
        pipeline_state=pipeline.replace(
            qpos=pipeline.qpos.at[free + 3 : free + 7].set(on_its_side)
        )
    )
    state = env.step(state, jp.zeros((NUM_AGENTS, env.action_size)))
    assert float(np.asarray(state.done)) == 1.0
    assert float(np.asarray(state.metrics["down_red"])) == 1.0
    # The boxer still standing is paid for the knockout.
    reward = np.asarray(state.reward)
    assert reward[1] > reward[0]


def test_leaving_the_ring_ends_the_round(env):
    state = env.reset(jax.random.PRNGKey(6))
    pipeline = state.pipeline_state
    free = env._sides[scene.BLUE]["free_adr"]
    state = state.replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[free].set(9.0))
    )
    state = env.step(state, jp.zeros((NUM_AGENTS, env.action_size)))
    assert float(np.asarray(state.done)) == 1.0
    assert float(np.asarray(state.metrics["ring_out"])) == 1.0


def test_guard_pose_is_the_zero_action(env):
    """Action 0 must reproduce the upstream guard, not a slack-limb ragdoll."""
    state = env.reset(jax.random.PRNGKey(7))
    state = env.step(state, jp.zeros((NUM_AGENTS, env.action_size)))
    assert float(np.asarray(state.metrics["guard"])) > 0.5
    assert float(np.asarray(state.metrics["upright"])) > 0.9
