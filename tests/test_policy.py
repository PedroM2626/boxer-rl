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


def test_jabber_extends_the_lead_arm_instead_of_waving_it(level):
    pitch = level.actuator_index("red/left_shoulder_pitch_joint")
    elbow = level.actuator_index("red/left_elbow_joint")
    jab = policy.jabber(level.action_size, pitch, elbow)

    # Start the clock off zero: the extension is a clipped sine, so t=0 is a
    # legitimate no-motion sample and would mask a policy that never moved at all.
    times = 0.1 + jp.arange(8) * 0.2
    values = np.asarray(jab(jp.zeros((8, level.observation_size)), times,
                            jax.random.PRNGKey(0)))

    assert int((np.abs(values) > 0).sum(axis=1).max()) == 2, \
        "exactly two joints should be driven"
    # The direction is the whole point. The fixture used to be a sine on shoulder
    # pitch alone, which rotates the arm about the shoulder and lifts the fist through
    # an arc -- a greeting, and the reason a scripted "jab" here kept arriving on the
    # opponent's gloves. Below the stance value means straightening.
    assert (values[:, pitch] <= 0).all() and (values[:, elbow] <= 0).all()
    assert values[:, elbow].min() < -0.3, "the cycle has to actually straighten the arm"
    assert (values[:, elbow] == 0).any(), "and it has to come back to the guard"
    assert float(np.abs(values).max()) <= 1.0, "actions must stay normalised"


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


def test_a_scripted_name_still_works_as_a_path(level):
    """argparse turns a positional into a Path, and that is what rollout passes in.

    Comparing it to the text "passive" silently sent the scripted corner to the
    checkpoint reader, so asking for two scripted boxers died on a missing directory.
    """
    from pathlib import Path

    scripted = policy.make_opponent(Path("passive"), level)
    action = scripted(jp.zeros((2, level.observation_size)), jp.zeros(2), jax.random.PRNGKey(0))
    assert float(jp.abs(action).max()) == 0.0
