import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest

from boxer.striking_train import Actor,Critic,log_probability,make_ppo_update,main
from boxer.striking_update_audit import UpdateAudit


def test_passive_observer_does_not_change_update_params_moments_or_loss(tmp_path):
    actor=Actor(); critic=Critic(); key=jax.random.PRNGKey(0)
    obs=jp.ones((8,148))*.01
    ap=actor.init(key,obs); vp=critic.init(key,obs)
    tx=optax.adam(.0001); optimizer=tx.init((ap,vp))
    mean,std=actor.apply(ap,obs);raw=jp.ones((8,17))*.1
    batch=dict(obs=obs,raw=raw,logp=log_probability(raw,mean,std),adv=jp.linspace(-1,1,8),returns=jp.ones(8))
    update=make_ppo_update(actor,critic,tx)
    expected=update(ap,vp,optimizer,batch)
    observer=UpdateAudit(actor,log_probability,tmp_path/'audit')
    observer.begin(ap,batch,np.array([0,1,1,0,0,1,1,0]))
    actual=update(ap,vp,optimizer,batch);observer.observe(actual[0],0,0)
    observer.after_normalizer(actual[0],obs)
    for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
        np.testing.assert_array_equal(a,b)
    assert observer.records[0]['profiles']['all']['exact_gaussian_kl_mean']<1e-10
    assert observer.records[-1]['profiles']['neutral']['rows']==4
    assert observer.records[-1]['profiles']['active']['rows']==4
    assert observer.records[-1]['phase']=='after_normalizer'
    with pytest.raises(ValueError):main(['--update-audit-directory',str(tmp_path/'bad')])
