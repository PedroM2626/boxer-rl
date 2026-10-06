"""Checkpoint resolution and the scripted opponents."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer import policy


def test_resolve_checkpoint_picks_the_highest_step(tmp_path):
    for step in ("0000000100", "0000000300", "0000000200"):
        (tmp_path / step).mkdir()
    assert policy.resolve_checkpoint(tmp_path).name == "0000000300"


def test_resolve_checkpoint_accepts_a_step_directory(tmp_path):
    step = tmp_path / "0000000300"
    step.mkdir()
    (step / "ppo_network_config.json").write_text("{}")
    assert policy.resolve_checkpoint(step).name == "0000000300"


def test_resolve_checkpoint_rejects_an_empty_root(tmp_path):
    with pytest.raises(SystemExit):
        policy.resolve_checkpoint(tmp_path)


def test_passive_is_exactly_zero_action(level):
    action = policy.passive(level.action_size)(
        jp.zeros((4, level.observation_size)), jp.arange(4) * 0.02, jax.random.PRNGKey(0)
    )
    assert action.shape == (4, level.action_size)
    assert float(jp.abs(action).max()) == 0.0


def test_jabber_moves_only_the_lead_arm_joints(level):
    pitch = level.actuator_index("red/left_shoulder_pitch_joint")
    roll = level.actuator_index("red/left_shoulder_roll_joint")
    jab = policy.jabber(level.action_size, pitch, roll)

    # Start the clock off zero: the jab is a sine, so t=0 is a legitimate
    # no-motion sample and would mask a policy that never moved at all.
    times = 0.1 + jp.arange(8) * 0.2
    action = jab(jp.zeros((8, level.observation_size)), times, jax.random.PRNGKey(0))

    moved = np.asarray(jp.abs(action) > 0)
    assert moved[:, pitch].all(), "the lead shoulder should be oscillating"
    assert moved[:, roll].all()
    assert int(moved.sum(axis=1)[0]) == 2, "exactly two joints should be driven"
    assert float(jp.abs(action).max()) <= 1.0, "actions must stay normalised"


def test_actuator_index_rejects_unknown_names(level):
    with pytest.raises(KeyError):
        level.actuator_index("red/definitely_not_a_joint")


def test_make_opponent_resolves_indices_from_the_env_not_from_literals(level):
    """Guards against re-introducing a hard-coded actuator index."""
    jab = policy.make_opponent("jabber", level)
    pitch = level.actuator_index("red/left_shoulder_pitch_joint")
    action = jab(jp.zeros((2, level.observation_size)), jp.array([0.0, 0.1]), jax.random.PRNGKey(0))
    assert float(jp.abs(action[:, pitch]).max()) > 0.0

    passive = policy.make_opponent("passive", level)
    assert float(jp.abs(passive(jp.zeros((2, level.observation_size)), jp.zeros(2),
                                 jax.random.PRNGKey(0))).max()) == 0.0
