"""Pre-contact selection and actual saved-state continuation contracts."""
import copy

import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest

from boxer.striking_anchor import load_anchor_candidates,mixed_indices,require_anchor_continuation
from boxer.striking_train import Actor,normalize,require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA
from tools.review_striking_anchor_compare import candidate_reasons
from tools.striking_demonstration_init import behavior_loss


def test_candidate_filter_rejects_incomplete_close_contact_and_invalid_prefixes():
    row=dict(survived=True,first_observed_contacts=[dict(geoms=['red/left_foot','floor'])])
    trace=dict(action=np.zeros((50,2,17),np.float32),root_distance_before=np.ones(50)*2,
        root_distance_after=np.ones(50)*1.5)
    assert candidate_reasons(row,trace)==[]
    assert 'incomplete_or_fallen_prefix' in candidate_reasons(dict(row,survived=False),trace)
    trace['root_distance_after'][0]=.9
    assert 'root_separation_at_or_below_0.9m' in candidate_reasons(row,trace)
    trace['root_distance_after'][0]=1.5
    row['first_observed_contacts']=[dict(geoms=['red/left_hand_collision','blue/torso_collision'])]
    assert 'positive_force_inter_body_contact' in candidate_reasons(row,trace)
    trace['action'][0,0,0]=np.nan
    assert 'nonfinite_state' in candidate_reasons(row,trace)


def test_actual_anchors_are_deduplicated_fresh_and_optimizer_resume_is_explicit():
    data,manifest=load_anchor_candidates()
    assert data['observation'].shape==(1600,148) and manifest['unique_role_trajectories']==32
    assert manifest['rows_by_role']=={'0':800,'1':800}
    assert not np.any(data['action']) and not np.any(data['observation'][:,146:])
    assert not set(data['seed'])&{7,23,31,47}
    saved,_=load_checked(SOURCE,SOURCE_SHA);require_anchor_continuation(saved)
    for changed in (dict(saved,supervised_updates=500),dict(saved,training_kind='ppo')):
        with pytest.raises(ValueError):require_anchor_continuation(changed)
    changed=dict(saved);changed.pop('sampler_state')
    with pytest.raises(ValueError):require_anchor_continuation(changed)
    rngs=[np.random.default_rng() for _ in range(2)]
    for rng in rngs:rng.bit_generator.state=copy.deepcopy(saved['sampler_state'])
    for a,b in zip(mixed_indices(rngs[0],7500,1600),mixed_indices(rngs[1],7500,1600)):
        np.testing.assert_array_equal(a,b)
    assert rngs[0].bit_generator.state==rngs[1].bit_generator.state


def test_preservation_labels_have_zero_initial_anchor_loss_and_bc_adam_continues():
    data,_=load_anchor_candidates();saved,_=load_checked(SOURCE,SOURCE_SHA)
    x=jp.array(normalize(data['observation'][:64],saved['norm']));actor=Actor()
    labels=jp.tanh(actor.apply(saved['actor'],x)[0]);weight=jp.ones(64)
    assert float(behavior_loss(actor,saved['actor'],x,labels,weight))<1e-12
    assert float(behavior_loss(actor,saved['actor'],x,jp.zeros_like(labels),weight))>1e-4
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
    gradient=jax.grad(lambda p:behavior_loss(actor,p,x,jp.zeros_like(labels),weight))(saved['actor'])
    changes,optimizer=tx.update(gradient,saved['optimizer'],saved['actor'])
    params=optax.apply_updates(saved['actor'],changes)
    assert int(optimizer[1][0].count)==4001
    np.testing.assert_array_equal(params['params']['log_std'],saved['actor']['params']['log_std'])
    snapshot=dict(saved,actor=params,optimizer=optimizer,training_kind='supervised_motor_anchor')
    for key in ('critic','norm','key'):
        for a,b in zip(jax.tree_util.tree_leaves(snapshot[key]),jax.tree_util.tree_leaves(saved[key])):
            np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):require_ppo_checkpoint(snapshot)
