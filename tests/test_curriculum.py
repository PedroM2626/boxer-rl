"""Physical control and promotion checks for the unassisted curriculum."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.curriculum_env import UnassistedG1, stage_config
from boxer.curriculum_eval import passes_gate
from boxer.curriculum_train import load_policy_params, save_policy


def trial(**changes):
    value = dict(survival_rate=1.0, linear_tracking_error=.05,
                 yaw_tracking_error=.05, required_distance=1.8,
                 directed_distance=2.5, arrival_rate=1.0)
    return dict(value, **changes)


def test_gate_rejects_falls_stationary_walking_and_failed_directions():
    assert not passes_gate("balance", [])
    assert passes_gate("balance", [trial()])
    assert not passes_gate("balance", [trial(survival_rate=.9)])
    assert not passes_gate("walk", [trial(directed_distance=.1)])
    assert not passes_gate("walk", [trial(), trial(yaw_tracking_error=.4)])
    assert not passes_gate("approach", [trial(arrival_rate=.5)])
    assert not passes_gate("walk", [trial(linear_tracking_error=float("nan"))])


def test_checkpoint_checksum_detects_corruption(tmp_path):
    params = (np.array([1., 2.]), {"weights": np.eye(2)}, {})
    save_policy(tmp_path, params, dict(stage_steps=123, stage="balance"))
    restored, metadata = load_policy_params(tmp_path)
    assert not metadata["optimizer_saved"]
    np.testing.assert_array_equal(restored[0], params[0])
    (tmp_path / metadata["file"]).write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        load_policy_params(tmp_path)


def test_real_physics_has_free_root_and_leg_action_authority():
    config = stage_config("balance")
    env = UnassistedG1(config=config)
    model = env.mj_model
    assert model.nq == 36 and model.nu == 29 and model.neq == 0
    assert not np.any(model.body_gravcomp)
    assert len(env.leg_action_indices) == 12
    assert not config.push_config.enable
    state = jax.jit(env.reset)(jax.random.PRNGKey(7))
    step = jax.jit(env.step)
    zero = step(state, jp.zeros(29))
    action = jp.zeros(29).at[env.leg_action_indices[0]].set(.4)
    moved = step(state, action)
    jax.block_until_ready(moved)
    expected = np.asarray(env._default_pose) + np.asarray(action) * config.action_scale
    np.testing.assert_allclose(moved.data.ctrl, expected, atol=1e-6)
    assert np.max(np.abs(np.asarray(moved.data.qpos - zero.data.qpos))) > 1e-5
    np.testing.assert_array_equal(moved.data.xfrc_applied, 0.)
    np.testing.assert_array_equal(moved.data.qfrc_applied, 0.)
    # A robot placed too low is a fall, even if its linear speed is zero.
    fallen = state.data.replace(qpos=state.data.qpos.at[2].set(.4))
    assert bool(env._get_termination(fallen))
