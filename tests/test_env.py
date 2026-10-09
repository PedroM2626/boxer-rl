"""The duel env: shapes, jittability, and whether hits are real."""

import json
import jax
import numpy as np
import pytest
from jax import numpy as jp

from boxer import scene
from boxer import env
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

    def body(state, _):
        clock = jp.atleast_1d(state.pipeline_state.time)[:1]
        action = jp.stack([
            jp.zeros(level.action_size),
            jab(state.obs[1][None], clock, jax.random.PRNGKey(0))[0],
        ])
        state = level.step(state, action)
        return state, state.metrics["hits_landed"]

    # One compile for the whole round. Stepping eagerly re-traced the physics scan
    # on every control step, and the XLA CPU backend aborted in the fourth minute
    # of compiling them (Fatal Python error: Aborted, in backend_compile_and_load).
    _, hits = jax.lax.scan(body, state, None, length=40)
    assert float(hits.max()) > 0.0, "a swinging fist that reaches the opponent should land a punch"


def test_a_fast_fist_travelling_away_from_the_opponent_is_not_a_punch(level):
    """The gate's third condition, added after watching a trained policy.

    Rendered rollout: the boxer topples forward with its arm over the opponent's
    chest. It is upright enough, and its fist is fast relative to its own torso,
    so both older conditions pass and the collapse scores as a punch. Direction
    is the only thing that separates throwing from falling forward, so the gate
    has to require the fist to be closing on a target.
    """
    state = level.reset(jax.random.PRNGKey(3))
    me = level._frame(state.pipeline_state, scene.RED)
    foe = level._frame(state.pipeline_state, scene.BLUE)
    assert float(level._committed(me, foe)) == 0.0, "nothing is moving at reset"

    toward = foe["torso"] - me["torso"]
    toward = toward / jp.linalg.norm(toward)

    retract = dict(me, left_palm_vel=me["torso_vel"] - 5.0 * toward)
    assert float(level._committed(retract, foe)) == 0.0, (
        "a fist moving at 5 m/s away from the opponent is not a punch"
    )

    jab = dict(me, left_palm_vel=me["torso_vel"] + 5.0 * toward)
    assert float(level._committed(jab, foe)) > 0.5, (
        "the same speed aimed at the opponent should read as a punch"
    )


def test_fighting_distance_only_pays_an_upright_boxer(level):
    """The range term used to pay a boxer for collapsing into its opponent.

    It was multiplied by ``alive``, so the reward for standing at punching range
    was still collected on the way down and diving paid better than boxing. The
    torso quaternion *sensor* is rewritten here, not the joint state, so head
    height, distance and "alive" stay identical between the two evaluations and
    posture is the only thing that can move the number.
    """
    def reward_at(roll: float):
        state = level.reset(jax.random.PRNGKey(5))
        pipeline = state.pipeline_state
        adr = level._sensor_adr["red/torso_quat"]
        half = roll / 2.0
        data = pipeline.sensordata.at[adr:adr + 4].set(
            jp.array([jp.cos(half), 0.0, jp.sin(half), 0.0])
        )
        pipeline = pipeline.replace(sensordata=data)
        reward, parts = level._reward(
            pipeline, scene.RED, scene.BLUE, jp.zeros(level.action_size),
            level._contact_damage(pipeline), 1.0,
        )
        return float(reward), float(parts["upright"]), float(parts["distance"])

    standing, upright_standing, gap = reward_at(0.0)
    tipped, upright_tipped, gap_after = reward_at(jp.pi / 2.0)
    assert upright_standing > 0.99 and upright_tipped < 0.01
    assert gap == pytest.approx(gap_after), "posture must not move the boxers"
    assert standing - tipped > 0.5, (
        f"tipping over at fighting distance must cost real reward, got "
        f"{standing:.2f} upright against {tipped:.2f} sideways"
    )


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


def test_a_run_directory_rebuilds_its_own_environment(level, tmp_path):
    """A checkpoint is only meaningful in the env it learned, and that has to travel."""
    (tmp_path / env.CONFIG_FILE).write_text(json.dumps(level.config()))
    rebuilt = env.build_from_run(tmp_path)
    assert rebuilt.config() == level.config()


def test_evaluation_defaults_to_the_scale_every_run_trained_at(tmp_path):
    """The bug this pins: the arena used to build BoxingEnv() and get 1.0.

    Actions here are relative joint targets, so evaluating a 0.25-scale policy at
    1.0 multiplies both its mean command and its sampled noise by four. That made a
    trained agent fall in half a second and read as a training failure.
    """
    resolved = env.resolve_env(("passive", "jabber"), 5, action_scale=0.25)
    assert resolved.config()["action_scale"] == 0.25


def test_a_recorded_config_beats_the_fallback_scale(level, tmp_path):
    (tmp_path / env.CONFIG_FILE).write_text(json.dumps({**level.config(),
                                                        "action_scale": 0.1}))
    resolved = env.resolve_env((str(tmp_path / "checkpoint"), "passive"),
                               5, action_scale=0.25)
    assert resolved.config()["action_scale"] == 0.1


def _close_quarters_env():
    return BoxingEnv(spawn_distance=0.85, spawn_jitter=0.0, yaw_jitter=0.0,
                     pose_noise=0.0, action_scale=0.25)


def test_sustained_contact_stops_paying_after_the_first_touch():
    """A fist resting on the opponent is one punch, not one per substep.

    The 204M arm measured 185 "punches" a round doing this: hands wrapped around
    the other boxer's head, scoring ~0.7 times per control step because hits summed
    the contact buffer over all five substeps. Contact frames are still reported, so
    the difference between touching and hitting is measurable either way.
    """
    from boxer import policy as policy_module

    env = _close_quarters_env()
    jab = policy_module.jabber(
        env.action_size,
        pitch_adr=env.actuator_index("red/left_shoulder_pitch_joint"),
        roll_adr=env.actuator_index("red/left_shoulder_roll_joint"),
    )
    state = env.reset(jax.random.PRNGKey(11))
    step = jax.jit(env.step)
    hits = contacts = 0.0
    for _ in range(40):
        action = jp.stack([
            jab(state.obs[0][None], jp.array([float(state.pipeline_state.time)]),
                jax.random.PRNGKey(0))[0],
            jp.zeros(env.action_size),
        ])
        state = step(state, action)
        m = {k: float(np.asarray(v)) for k, v in state.metrics.items()}
        hits += m["hits_red"]
        contacts += m["contacts"]
    assert contacts > 5.0, "the boxers should be in contact at 0.85 m apart"
    assert hits <= 8.0, f"contact frames must not be counted as punches: {hits} vs {contacts}"


def test_embrace_costs_more_than_fighting_distance():
    """The env had no notion of a clinch, so holding was free and then optimal.

    Two environments rather than one moved qpos: the boxer's position, torso
    quaternion and palm velocities all come out of sensors, and editing the joint
    state without running kinematics leaves the sensors describing the old pose.
    """
    def reward_at(separation: float) -> float:
        env = BoxingEnv(spawn_distance=separation, spawn_jitter=0.0, yaw_jitter=0.0,
                        pose_noise=0.0, action_scale=0.25)
        state = env.reset(jax.random.PRNGKey(12))
        reward, parts = env._reward(
            state.pipeline_state, scene.RED, scene.BLUE, jp.zeros(env.action_size),
            env._contact_damage(state.pipeline_state), 0.0,
        )
        assert abs(float(parts["distance"]) - separation) < 0.05, (
            "the two setups must differ in separation, not in something else"
        )
        return float(reward)

    embraced, fighting = reward_at(0.45), reward_at(0.75)
    assert fighting > embraced, (
        f"being inside embrace ({embraced:.2f}) should not pay better than being "
        f"at punching distance ({fighting:.2f})"
    )
