"""The diagnostic may compute actions, never reposition or support a body."""
import jax.numpy as jp
import numpy as np

from boxer.striking_env import StrikingEnv
from tools.striking_directed_feasibility import directed_action
from test_paired_foundation import foundation


def test_directed_correction_is_bounded_and_does_not_modify_physics(foundation):
    def motor(obs,rng):return jp.zeros((2,29)),{}
    env=StrikingEnv(foundation,None,motor=motor)
    env.reset(seed=7)
    # A diagnostic setup for the Jacobian helper only, not a runtime reset.
    env.pair.data.qpos[env.pair.maps[1]['root']]=env.pair.data.qpos[env.pair.maps[0]['root']]+.6
    import mujoco
    mujoco.mj_forward(env.pair.model,env.pair.data)
    qpos=env.pair.data.qpos.copy();qvel=env.pair.data.qvel.copy();ctrl=env.pair.data.ctrl.copy()
    action,used=directed_action(env,np.zeros((2,17)),0,'head')
    assert used and np.isfinite(action).all() and np.max(np.abs(action))<=1.
    np.testing.assert_array_equal(action[1],np.zeros(17))
    np.testing.assert_array_equal(env.pair.data.qpos,qpos)
    np.testing.assert_array_equal(env.pair.data.qvel,qvel)
    np.testing.assert_array_equal(env.pair.data.ctrl,ctrl)
    assert not np.any(env.pair.data.xfrc_applied) and not np.any(env.pair.data.qfrc_applied)
    assert env.pair.model.neq==0 and not np.any(env.pair.model.body_gravcomp)


def test_teacher_phase_is_missing_from_legacy_actor_observation(foundation):
    """Controlled history intervention, not an eligible physics trial."""
    def motor(obs,rng):return jp.zeros((2,29)),{}
    env=StrikingEnv(foundation,None,motor=motor);env.reset(seed=7)
    env.pair.data.qpos[env.pair.maps[1]['root']]=env.pair.data.qpos[env.pair.maps[0]['root']]+.6
    import mujoco
    mujoco.mj_forward(env.pair.model,env.pair.data)
    before=env.observation().copy()
    reaching,used=directed_action(env,np.zeros((2,17)),0,'head')
    assert used and np.any(reaching)
    env.pair.scorecard.last_score[0,0]=env.pair.elapsed
    recovering,used=directed_action(env,np.zeros((2,17)),0,'head')
    assert not used
    np.testing.assert_array_equal(before,env.observation())
    assert not np.array_equal(reaching,recovering)
