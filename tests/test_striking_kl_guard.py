"""Actor stops retain Adam momentum/counts while the critic and accounting continue."""
import copy,json
from pathlib import Path
import jax,jax.numpy as jp,numpy as np,optax,pytest
from boxer.striking_ppo_warmstart import SplitAdam
from boxer.striking_train import Actor,Critic,make_ppo_update,log_probability,require_ppo_checkpoint,main
from boxer.striking_kl_guard import SOURCE_SHA,TARGET,prepare_guard_state,guard_config,mean_rollout_kl
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/sampling-1933312/control/gate-16384')


def same_tree(a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)


def test_disabled_actor_does_not_drift_under_nonzero_adam_momentum():
    tx=SplitAdam();params=({'w':jp.array([1.,2.])},{'w':jp.array([3.])});state=tx.init(params)
    grads=({'w':jp.array([.3,.4])},{'w':jp.array([.2])})
    updates,state=tx.update(grads,state,params);params=optax.apply_updates(params,updates)
    assert np.any(np.asarray(state['actor_adam'][0].mu['w'])!=0)
    updates,after=tx.update(grads,state,params,update_actor=False)
    same_tree(state['actor_adam'],after['actor_adam']);same_tree(params[0],optax.apply_updates(params,updates)[0])
    assert int(after['critic_adam'][0].count)==int(state['critic_adam'][0].count)+1
    assert np.any(np.asarray(updates[1]['w'])!=0)


def test_critic_only_full_loss_update_preserves_real_actor_and_its_optimizer():
    saved,_=load_checked(SOURCE,SOURCE_SHA);actor=Actor();critic=Critic();obs=jp.ones((4,148))*.01
    mean,std=actor.apply(saved['actor'],obs);raw=mean+.1
    batch=dict(obs=obs,raw=raw,logp=log_probability(raw,mean,std),adv=jp.linspace(-1.,1.,4),returns=jp.ones(4))
    update=make_ppo_update(actor,critic,SplitAdam(),actor_enabled=False)
    ap,vp,state,loss=update(saved['actor'],saved['critic'],saved['optimizer'],batch)
    same_tree(ap,saved['actor']);same_tree(state['actor_adam'],saved['optimizer']['actor_adam'])
    assert int(state['critic_adam'][0].count)==36609 and np.isfinite(loss)
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(vp),jax.tree_util.tree_leaves(saved['critic'])))


def test_transition_and_variable_actor_ledger_keep_all_legacy_prefixes():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=prepare_guard_state(saved,SOURCE_SHA);require_ppo_checkpoint(new)
    for key in saved:same_tree(saved[key],new[key])
    assert guard_config(json.loads((SOURCE/'config.json').read_text()))['version']==11
    for key in ('kl_guard_actor_steps_by_rollout','kl_guard_target','kl_guard_transition'):
        bad=copy.deepcopy(new);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    advanced=copy.deepcopy(new)
    advanced.update(consumed_steps=1950720,optimized_steps=1950720,world_control_steps=1328128,kl_guard_actor_steps_by_rollout=[4])
    advanced['mixed_credits_by_mode_and_corner'][0][0]+=512;advanced['mixed_credits_by_mode_and_corner'][1][0]+=512
    for name,count in (('actor_adam',12940),('critic_adam',36624)):
        a=advanced['optimizer'][name];advanced['optimizer'][name]=(a[0]._replace(count=jp.array(count)),a[1])
    require_ppo_checkpoint(advanced)
    for ledger in ([],[0],[17],[5],[4,4]):
        bad=copy.deepcopy(advanced);bad['kl_guard_actor_steps_by_rollout']=ledger
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    bad=copy.deepcopy(advanced);bad['kl_guard_ppo']=False
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_guard_state(new,SOURCE_SHA)
    with pytest.raises(ValueError):main(['--kl-guard-ppo'])


def test_full_rollout_gaussian_kl_is_finite_and_detects_mean_shift():
    old=(np.zeros((8,17)),np.zeros(17))
    assert mean_rollout_kl(old,old)==0
    assert mean_rollout_kl(old,(np.full((8,17),.1),np.zeros(17)))>TARGET
    with pytest.raises(FloatingPointError):mean_rollout_kl(old,(np.full((8,17),np.nan),np.zeros(17)))
