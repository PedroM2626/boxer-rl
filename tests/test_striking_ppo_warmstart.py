"""Real optimizer migration, single clipping, interface and no-reset safeguards."""
import copy
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest

from boxer.striking_ppo_warmstart import (SplitAdam,anchor_loss,contact_config,
    migrate_supervised_state,validate_split_checkpoint)
from boxer.striking_train import (Actor,Critic,normalize,require_ppo_checkpoint,validate_resume_config,
    main,make_ppo_update,log_probability)
from tools.striking_contact_compare import load_checked,ORIGINAL,ORIGINAL_SHA

SOURCE=Path('checkpoints/striking-validation/motor-anchoring-bc4000/zero_anchor')
SHA='c7d6f1d5458f4767e6d7bedd98350ed8f79231de1030783fe9feb4b779d81e19'


def sources():
    saved,_=load_checked(SOURCE,SHA);original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
    return saved,original


def test_actual_migration_preserves_state_predictions_moments_and_round_trip():
    saved,original=sources();new=migrate_supervised_state(saved,original,SHA,ORIGINAL_SHA)
    require_ppo_checkpoint(new)
    for key in saved.keys()-{'training_kind','optimizer','ppo_optimizer_compatible','learners_per_world','opponent_mode'}:
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(new[key])):
            np.testing.assert_equal(a,b)
    assert int(new['optimizer']['actor_adam'][0].count)==5000
    assert int(new['optimizer']['critic_adam'][0].count)==28672
    for a,b in zip(jax.tree_util.tree_leaves(saved['optimizer'][1]),
            jax.tree_util.tree_leaves(new['optimizer']['actor_adam'])):np.testing.assert_array_equal(a,b)
    for name in ('mu','nu'):
        old=getattr(original['optimizer'][1][0],name)[1]
        padded=getattr(new['optimizer']['critic_adam'][0],name)
        for a,b in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(padded)):
            slices=tuple(slice(0,n) for n in np.shape(a));np.testing.assert_array_equal(a,np.asarray(b)[slices])
            if np.shape(a)!=np.shape(b):np.testing.assert_array_equal(np.asarray(b)[146:],0.)
    obs=np.random.default_rng(4).normal(size=(5,148)).astype(np.float32)
    restored=pickle.loads(pickle.dumps(new));validate_split_checkpoint(restored)
    for model,key in ((Actor(),'actor'),(Critic(),'critic')):
        for a,b in zip(jax.tree_util.tree_leaves(model.apply(saved[key],normalize(obs,saved['norm']))),
                jax.tree_util.tree_leaves(model.apply(restored[key],normalize(obs,restored['norm'])))):
            np.testing.assert_array_equal(a,b)
    assert saved['training_kind']=='supervised_motor_anchor' and saved['ppo_optimizer_compatible'] is False


def test_split_step_matches_joint_adam_and_clips_once():
    params=({'w':jp.array([1.,2.])},{'w':jp.array([3.])})
    joint=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
    split=SplitAdam();jstate=joint.init(params);sstate=split.init(params)
    # Unequal groups and changing directions make clipping twice observable in moments.
    for grads in (({'w':jp.array([30.,40.])},{'w':jp.array([120.])}),
                  ({'w':jp.array([-15.,4.])},{'w':jp.array([2.])})):
        a,jstate=joint.update(grads,jstate,params);b,sstate=split.update(grads,sstate,params)
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(x,y,rtol=2e-7,atol=1e-11)
        for i,name in enumerate(('actor_adam','critic_adam')):
            assert int(sstate[name][0].count)==int(jstate[1][0].count)
            for attr in ('mu','nu'):
                for x,y in zip(jax.tree_util.tree_leaves(getattr(jstate[1][0],attr)[i]),
                        jax.tree_util.tree_leaves(getattr(sstate[name][0],attr))):np.testing.assert_allclose(x,y,rtol=2e-7,atol=1e-10)


def test_unequal_counts_advance_independently_and_auxiliary_has_no_critic_gradient():
    saved,original=sources();new=migrate_supervised_state(saved,original,SHA,ORIGINAL_SHA)
    obs=jp.ones((3,148))*.1;labels=jp.zeros((3,17))
    def loss(ap,vp):return anchor_loss(Actor(),ap,obs,labels)
    grads=jax.grad(loss,argnums=(0,1))(new['actor'],new['critic'])
    for x in jax.tree_util.tree_leaves(grads[1]):np.testing.assert_array_equal(x,0.)
    _,optimizer=SplitAdam().update(grads,new['optimizer'],(new['actor'],new['critic']))
    assert int(optimizer['actor_adam'][0].count)==5001
    assert int(optimizer['critic_adam'][0].count)==28673


def test_fake_rename_missing_format_counts_and_profile_changes_fail_closed():
    import json
    saved,original=sources()
    with pytest.raises(ValueError):require_ppo_checkpoint(dict(saved,training_kind='ppo'))
    renamed=dict(saved,training_kind='ppo',ppo_optimizer_compatible=True)
    with pytest.raises(ValueError):require_ppo_checkpoint(renamed)
    new=migrate_supervised_state(saved,original,SHA,ORIGINAL_SHA)
    for changes in (dict(optimizer_kind='unknown'),dict(migration_version=0),dict(learners_per_world=1)):
        with pytest.raises(ValueError):require_ppo_checkpoint(dict(new,**changes))
    bad=copy.deepcopy(new);bad['optimizer']['actor_adam']=(bad['optimizer']['actor_adam'][0]._replace(count=jp.array(0)),bad['optimizer']['actor_adam'][1])
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    cfg=contact_config(json.loads((SOURCE/'config.json').read_text()),.02)
    validate_resume_config(cfg,cfg)
    with pytest.raises(ValueError):validate_resume_config(cfg,dict(cfg,anchor_loss_weight=0.))
    with pytest.raises(ValueError):main(['--contact-ppo'])


def test_legacy_update_remains_bit_exact_against_original_joint_ppo():
    actor=Actor();critic=Critic();key=jax.random.PRNGKey(4)
    ap=actor.init(key,jp.zeros((1,146)));vp=critic.init(key,jp.zeros((1,146)))
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001));optimizer=tx.init((ap,vp))
    rng=np.random.default_rng(9)
    batch={name:jp.array(rng.normal(size=shape).astype(np.float32)) for name,shape in
        [('obs',(16,146)),('raw',(16,17)),('logp',(16,)),('adv',(16,)),('returns',(16,))]}
    batch['raw']*=.1
    mean,log_std=actor.apply(ap,batch['obs'])
    batch['logp']=log_probability(batch['raw'],mean,log_std)+jp.linspace(-.3,.3,16)
    # Frozen reference from the original native trainer, prior to optional auxiliary support.
    @jax.jit
    def original(ap,vp,optimizer,batch):
        def loss(ap,vp):
            mean,log_std=actor.apply(ap,batch['obs'])
            ratio=jp.exp(log_probability(batch['raw'],mean,log_std)-batch['logp'])
            policy=-jp.mean(jp.minimum(ratio*batch['adv'],jp.clip(ratio,.8,1.2)*batch['adv']))
            value_loss=jp.mean((critic.apply(vp,batch['obs'])-batch['returns'])**2)
            entropy=jp.sum(log_std+.5*np.log(2*np.pi*np.e))
            return policy+.25*value_loss-.001*entropy
        loss_value,grads=jax.value_and_grad(loss,argnums=(0,1))(ap,vp)
        updates,optimizer=tx.update(grads,optimizer,(ap,vp))
        ap,vp=optax.apply_updates((ap,vp),updates)
        return ap,vp,optimizer,loss_value
    a=original(ap,vp,optimizer,batch);b=make_ppo_update(actor,critic,tx)(ap,vp,optimizer,batch)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)
