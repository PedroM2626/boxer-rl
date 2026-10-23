import copy
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_retention import (load_retention_data,prepare_retention_state,
    retention_config,validate_retention_state,retention_batch,retention_loss)
from boxer.striking_ppo_warmstart import SplitAdam
from boxer.striking_train import Actor,Critic,make_ppo_update,log_probability,main,validate_resume_config
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/ppo-anchor-bc5000/ppo_only')
SHA='e3e01b71de64e3d26a431ba87d484a45d349621df5461c274fb7e394b935ef4e'


def test_profile_transition_preserves_every_existing_state_and_rejects_mismatch():
    saved,_=load_checked(SOURCE,SHA);_,provenance=load_retention_data()
    prepared=prepare_retention_state(saved,SHA,1.,provenance)
    for key in saved:
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a,b)
    validate_retention_state(prepared,provenance)
    bad=copy.deepcopy(prepared);bad['retention_provenance']['demo_rows']=1
    with pytest.raises(ValueError):validate_retention_state(bad,provenance)
    with pytest.raises(ValueError):prepare_retention_state(dict(saved,optimized_steps=1),SHA,1.,provenance)
    cfg=retention_config(json.loads((SOURCE/'config.json').read_text()),1.)
    with pytest.raises(ValueError):validate_resume_config(cfg,dict(cfg,retention_loss_weight=0.))
    with pytest.raises(ValueError):main(['--demonstration-ppo'])
    with pytest.raises(ValueError):main(['--retention-loss-weight','1'])


def test_retention_batches_match_and_zero_coefficient_preserves_ppo_update():
    saved,_=load_checked(SOURCE,SHA);data,_=load_retention_data()
    first=np.random.default_rng();first.bit_generator.state=copy.deepcopy(saved['aux_sampler_state'])
    second=np.random.default_rng();second.bit_generator.state=copy.deepcopy(saved['aux_sampler_state'])
    extra=retention_batch(data,first,saved['norm']);other=retention_batch(data,second,saved['norm'])
    for k in extra:np.testing.assert_array_equal(extra[k],other[k])
    assert first.bit_generator.state==second.bit_generator.state
    assert extra['retention_obs'].shape==(256,148)
    np.testing.assert_array_equal(extra['retention_labels'][192:],0.)
    actor=Actor();critic=Critic();ap=saved['actor'];vp=saved['critic'];tx=SplitAdam()
    x=jp.zeros((16,148));mean,std=actor.apply(ap,x);raw=mean+.1
    batch=dict(obs=x,raw=raw,logp=log_probability(raw,mean,std),adv=jp.linspace(-1,1,16),returns=jp.ones(16),**extra)
    regular=make_ppo_update(actor,critic,tx)(ap,vp,saved['optimizer'],batch)
    zero=make_ppo_update(actor,critic,tx,retention_weight=0.)(ap,vp,saved['optimizer'],batch)
    for a,b in zip(jax.tree_util.tree_leaves(regular),jax.tree_util.tree_leaves(zero)):np.testing.assert_array_equal(a,b)
    updated=make_ppo_update(actor,critic,tx,retention_weight=1.)(ap,vp,saved['optimizer'],batch)
    assert int(updated[2]['actor_adam'][0].count)==7049
    assert int(updated[2]['critic_adam'][0].count)==30721
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(updated))
    gradients=jax.grad(lambda a,v:retention_loss(actor,a,extra['retention_obs'],extra['retention_labels'],extra['retention_weights']),argnums=(0,1))(ap,vp)
    assert any(np.any(x) for x in jax.tree_util.tree_leaves(gradients[0]))
    assert all(not np.any(x) for x in jax.tree_util.tree_leaves(gradients[1]))
