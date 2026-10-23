"""Contact feedback timing, unchanged physics, and exact BC warm-state retention."""
import copy

import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest

from boxer.striking_contact import (ContactAwareStrikingEnv, demonstration_flags,
    expand_supervised_state, recover_sampler)
from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, Critic, normalize, require_ppo_checkpoint
from tools.striking_contact_compare import SOURCE, SOURCE_SHA, load_checked, load_demonstrations
from tools.striking_demonstration_init import behavior_loss
from test_paired_foundation import foundation


def test_contact_features_are_physical_history_with_reset_and_unchanged_dynamics(foundation):
    def motor(obs,key):return jp.ones((2,29))*.02,{}
    old=StrikingEnv(foundation,None,motor=motor)
    new=ContactAwareStrikingEnv(foundation,None,motor=motor)
    np.testing.assert_array_equal(old.reset(23),new.reset(23)[:,:146])
    np.testing.assert_array_equal(new.observation()[:,146:],0.)
    action=np.random.default_rng(2).uniform(-.1,.1,(2,17)).astype(np.float32)
    for _ in range(5):
        a,ra,da,_=old.step(action);b,rb,db,_=new.step(action)
        np.testing.assert_array_equal(a,b[:,:146])
        np.testing.assert_array_equal(old.pair.data.qpos,new.pair.data.qpos)
        np.testing.assert_array_equal(old.pair.data.qvel,new.pair.data.qvel)
        np.testing.assert_array_equal(ra,rb);assert da==db
    before=new.pair.data.qpos.copy()
    new.pair.scorecard.last_score[1,0]=new.pair.elapsed
    np.testing.assert_array_equal(new.observation()[:,146:],[[0,0],[1,0]])
    np.testing.assert_array_equal(new.pair.data.qpos,before)
    np.testing.assert_array_equal(old.observation(),new.observation()[:,:146])
    assert new.pair.model.neq==0 and not np.any(new.pair.model.body_gravcomp)
    assert not np.any(new.pair.data.qfrc_applied) and not np.any(new.pair.data.xfrc_applied)
    np.testing.assert_array_equal(new.reset(23)[:,146:],0.)


def test_offline_history_uses_control_end_not_substep_event_time():
    trial=dict(learner_corner=1,impact_events=[
        dict(elapsed=1.94,hands=[dict(corner=1,hand=0,time=1.926)]),
        dict(elapsed=2.,hands=[dict(corner=0,hand=1),dict(corner=1,hand=1)])])
    flags=demonstration_flags(trial)
    np.testing.assert_array_equal(flags[:97],0.)
    np.testing.assert_array_equal(flags[97:100],[[1,0]]*3)
    np.testing.assert_array_equal(flags[100:],1.)


def test_actual_bc_migration_retains_prefix_predictions_moments_and_lineage():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=expand_supervised_state(saved)
    obs=np.random.default_rng(4).normal(size=(8,146)).astype(np.float32)
    padded=np.c_[obs,np.ones((8,2),np.float32)]
    for module,key in ((Actor(),'actor'),(Critic(),'critic')):
        a=module.apply(saved[key],normalize(obs,saved['norm']))
        b=module.apply(new[key],normalize(padded,new['norm']))
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(x,y,atol=2e-6)
    for key in ('actor','critic','optimizer'):
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(new[key])):
            slices=tuple(slice(0,n) for n in np.shape(a))
            np.testing.assert_array_equal(a,np.asarray(b)[slices])
            if np.shape(a)!=np.shape(b):np.testing.assert_array_equal(np.asarray(b)[146:],0.)
    assert int(new['optimizer'][1][0].count)==2000
    for key in saved.keys()-{'actor','critic','optimizer','norm','actor_inputs','training_kind'}:
        np.testing.assert_equal(new[key],saved[key])
    for k,tail in (('mean',0.),('var',1.)):
        np.testing.assert_array_equal(new['norm'][k][:146],saved['norm'][k])
        np.testing.assert_array_equal(new['norm'][k][146:],tail)
    assert new['norm']['count']==saved['norm']['count']
    with pytest.raises(ValueError):require_ppo_checkpoint(new)
    with pytest.raises(ValueError):expand_supervised_state(new)


def test_sampler_recovery_dataset_flags_and_zero_control_update():
    rng=np.random.default_rng(313)
    for _ in range(2000):rng.integers(0,7500,256)
    recovered=recover_sampler()
    assert recovered.bit_generator.state==rng.bit_generator.state
    clone=np.random.default_rng();clone.bit_generator.state=copy.deepcopy(recovered.bit_generator.state)
    ix=recovered.integers(0,7500,256)
    np.testing.assert_array_equal(ix,clone.integers(0,7500,256))
    observations,labels,flags,files=load_demonstrations()
    assert observations.shape==(7500,146) and len(files)==15
    assert flags.shape==(7500,2) and flags.sum()>0
    assert np.isin(flags,[0.,1.]).all()
    assert all(np.all(np.diff(f,axis=0)>=0) for f in flags.reshape(15,500,2))
    assert not flags.reshape(15,500,2)[:,0].any()
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=expand_supervised_state(saved)
    actor=Actor();tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
    x=jp.array(normalize(np.c_[observations[ix],np.zeros((256,2),np.float32)],new['norm']))
    gradient=jax.grad(lambda p:behavior_loss(actor,p,x,jp.array(labels[ix]),jp.ones(256)))(new['actor'])
    updates,opt=tx.update(gradient,new['optimizer'],new['actor'])
    params=optax.apply_updates(new['actor'],updates)
    for tree in (params,opt[1][0].mu,opt[1][0].nu):
        np.testing.assert_array_equal(np.asarray(tree['params']['Dense_0']['kernel'])[146:],0.)
    assert int(opt[1][0].count)==2001
