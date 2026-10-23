"""Whole-body authority, true action history and explicit warm-state migration."""
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_body import WholeBodyStrikingEnv, expand_learning_state
from boxer.striking_train import Actor, Critic, normalize
from boxer.striking_env import StrikingEnv
from test_paired_foundation import foundation


def test_body_residual_uses_only_joint_targets_and_records_actual_history(foundation):
    def motor(obs, key):
        return jp.zeros((2,29)), {}
    env = WholeBodyStrikingEnv(foundation, None, motor=motor)
    obs = env.reset(7)
    assert obs.shape == (2,161)
    assert np.array_equal(obs[:,-15:],np.zeros((2,15)))
    action = np.zeros((2,32)); action[:,17:] = 1
    obs, _, _, outcome = env.step(action)
    np.testing.assert_allclose(outcome['applied_actions'][:,env.body_indices],.04)
    np.testing.assert_array_equal(outcome['applied_actions'][:,env.arm_indices],0.)
    np.testing.assert_array_equal(env.pair.last_action,outcome['applied_actions'])
    np.testing.assert_array_equal(obs[:,-15:],1.)
    assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)
    assert env.pair.model.neq == 0 and not np.any(env.pair.model.body_gravcomp)
    env.reset(7)
    np.testing.assert_array_equal(env.body_residual,0.)


def test_zero_authority_control_has_no_added_body_torque(foundation):
    def motor(obs, key):
        return jp.zeros((2,29)), {}
    env=WholeBodyStrikingEnv(foundation,None,motor=motor,body_residual_radians=0.)
    action=np.zeros((2,32));action[:,17:]=1
    _,_,_,outcome=env.step(action)
    np.testing.assert_array_equal(outcome['applied_actions'],0.)
    with pytest.raises(ValueError):env.step(np.zeros((2,17)))


def test_zero_added_actions_preserve_base_trajectory_and_first_input_block(foundation):
    def motor(obs,key):
        return jp.ones((2,29)) * .02, {}
    old=StrikingEnv(foundation,None,motor=motor)
    new=WholeBodyStrikingEnv(foundation,None,motor=motor)
    np.testing.assert_array_equal(old.reset(23),new.reset(23)[:,:146])
    action=np.random.default_rng(2).uniform(-.1,.1,(2,17))
    for _ in range(5):
        a,ra,da,oa=old.step(action)
        b,rb,db,ob=new.step(np.c_[action,np.zeros((2,15))])
        np.testing.assert_array_equal(a,b[:,:146])
        np.testing.assert_array_equal(old.pair.data.qpos,new.pair.data.qpos)
        np.testing.assert_array_equal(ra,rb)
        assert da==db


def test_migration_preserves_old_outputs_values_adam_and_counters():
    root=Path('checkpoints/striking-pilot/step-1048576')
    if not root.exists():pytest.skip('archived source needed for migration validation')
    old=pickle.loads((root/'step-1048576.pkl').read_bytes())
    new=expand_learning_state(old)
    obs=np.random.default_rng(4).normal(size=(4,146)).astype(np.float32)
    expanded=np.concatenate([obs,np.zeros((4,15),np.float32)],axis=1)
    a0,s0=Actor().apply(old['actor'],normalize(obs,old['norm']))
    a1,s1=Actor(action_size=32).apply(new['actor'],normalize(expanded,new['norm']))
    np.testing.assert_allclose(a0,a1[:,:17],atol=2e-6)
    np.testing.assert_array_equal(a1[:,17:],0.)
    np.testing.assert_array_equal(s0,s1[:17])
    np.testing.assert_allclose(s1[17:],-1.9)
    np.testing.assert_allclose(Critic().apply(old['critic'],normalize(obs,old['norm'])),
        Critic().apply(new['critic'],normalize(expanded,new['norm'])),atol=2e-6)
    for original,padded in zip(jax.tree_util.tree_leaves(old['optimizer']),jax.tree_util.tree_leaves(new['optimizer'])):
        slices=tuple(slice(0,n) for n in np.shape(original))
        np.testing.assert_array_equal(original,np.asarray(padded)[slices])
    for key in ('consumed_steps','optimized_steps','foundation_sha256'):
        assert new[key]==old[key]
    np.testing.assert_array_equal(new['key'],old['key'])
    assert new['norm']['count']==old['norm']['count']
