"""Residual neutrality, physical leg authority, and episodic PPO targets."""
import jax
import jax.numpy as jp
import numpy as np
import pytest
from pathlib import Path
import sys

from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, advantages, main
from test_paired_foundation import foundation, pair


def test_neutral_head_and_arm_residuals_preserve_motor_leg_actions(foundation,pair):
    baseline=np.tile(np.linspace(-.1,.1,29),(2,1)).astype(np.float32)
    def motor(obs,rng):
        return jp.array(baseline),{}
    env=StrikingEnv(foundation,None,motor=motor,pair=pair)
    env.reset(seed=23)
    observation,reward,done,outcome=env.step(np.zeros((2,17)))
    np.testing.assert_allclose(outcome['applied_actions'],baseline,atol=1e-7)
    assert observation.shape==(2,146) and np.isfinite(reward).all()
    env.reset(seed=23)
    action=np.zeros((2,17));action[:,3:]=1.
    _,_,_,outcome=env.step(action)
    legs=[i for i in range(29) if i not in env.arm_indices]
    np.testing.assert_allclose(outcome['applied_actions'][:,legs],baseline[:,legs])
    np.testing.assert_allclose(outcome['applied_actions'][:,env.arm_indices],
                               baseline[:,env.arm_indices]+.32,atol=1e-7)


def test_new_actor_starts_at_exact_neutral_mean():
    actor=Actor();obs=jp.ones((2,146))
    params=actor.init(jax.random.PRNGKey(2),obs)
    mean,log_std=actor.apply(params,obs)
    np.testing.assert_array_equal(mean,np.zeros((2,17)))
    assert np.isfinite(log_std).all()


def test_first_fall_and_clock_cut_gae_between_rounds():
    rewards=np.array([[1.],[1.]],np.float32)
    dones=np.array([[1.],[0.]],np.float32)
    adv,returns=advantages(rewards,dones,np.zeros_like(rewards),np.array([100.]))
    assert returns[0,0]==1.
    assert returns[1,0]==100.
    assert np.isfinite(adv).all()


@pytest.mark.skipif(sys.platform=='win32',reason='Linux supervisor uses flock')
def test_second_boxing_job_is_rejected_before_loading_physics(tmp_path,monkeypatch):
    import fcntl
    monkeypatch.chdir(tmp_path)
    Path('runs').mkdir()
    with Path('runs/unassisted_boxing.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            main(['--output','runs/other','--ceiling','64','--worlds','1','--unroll','32'])
    assert not Path('runs/other/config.json').exists()
