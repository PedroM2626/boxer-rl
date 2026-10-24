"""Archived policy routing, independent normalization and durable warm-state credits."""
import copy
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_opponent_pool import (SOURCE, SOURCE_SHA, SPECS, OFFSET,
    prepare_pool_state, pool_config, pool_predictor, draw_opponent, restore_sampler,
    validate_pool_ledger, ContactPoolMixedStrikingEnv)
from boxer.striking_train import Actor, normalize, require_ppo_checkpoint, validate_resume_config, main
from tools.striking_contact_compare import load_checked
from test_paired_foundation import foundation


@pytest.fixture(scope='module')
def prepared():
    source,_ = load_checked(Path(SOURCE),SOURCE_SHA)
    snapshots=[]
    for path,sha,_ in SPECS:
        saved,_=load_checked(Path(path),sha)
        snapshots.append(dict(source_sha256=sha,actor=saved['actor'],norm=saved['norm']))
    return source,prepare_pool_state(source,SOURCE_SHA,snapshots)


def test_lossless_transition_and_explicit_resume(prepared):
    source,changed=prepared
    for key in source.keys()-{'opponent_mode'}:
        for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(changed[key])):
            np.testing.assert_equal(a,b)
    require_ppo_checkpoint(pickle.loads(pickle.dumps(changed)))
    cfg=json.loads((Path(SOURCE)/'config.json').read_text());new=pool_config(cfg)
    assert new['version']==13 and new['gae_lambda']==.99 and new['gamma']==.999
    validate_resume_config(new,new)
    with pytest.raises(ValueError):validate_resume_config(cfg,new)
    with pytest.raises(ValueError):prepare_pool_state(changed,SOURCE_SHA,changed['opponent_pool'])
    with pytest.raises(ValueError):prepare_pool_state(source,'0'*64,changed['opponent_pool'])
    with pytest.raises(ValueError):main(['--pool-opponent-ppo'])
    changed_copy=copy.deepcopy(changed)
    changed_copy['opponent_pool'][0]['norm']['mean']=np.ones(148)
    np.testing.assert_array_equal(source['norm']['mean'],changed['norm']['mean'])
    with pytest.raises(ValueError):require_ppo_checkpoint(changed_copy)


@pytest.mark.parametrize('key', ['pool_opponent_ppo','opponent_pool_transition','pool_sampler_state',
    'pool_assignments','opponent_pool','pool_credits_by_opponent_and_corner'])
def test_missing_pool_provenance_rejected(prepared,key):
    saved=copy.deepcopy(prepared[1]);del saved[key]
    with pytest.raises((ValueError,TypeError)):require_ppo_checkpoint(saved)


def test_routed_actions_use_each_opponents_own_normalizer(prepared):
    snapshots=prepared[1]['opponent_pool'];actor=Actor()
    obs=np.random.default_rng(19).normal(size=(8,148)).astype(np.float32)
    assignments=np.array([-1,0,1,-1,-1,2,0,-1])
    actual=np.asarray(pool_predictor(actor,snapshots)(jp.asarray(obs),jp.asarray(assignments)))
    for i,index in enumerate(np.maximum(assignments,0)):
        snapshot=snapshots[index]
        expected=np.asarray(jp.tanh(actor.apply(snapshot['actor'],jp.asarray(normalize(obs[i],snapshot['norm'])))[0]))
        np.testing.assert_allclose(actual[i],expected,atol=2e-6,rtol=2e-6)
    wrong=np.asarray(jp.tanh(actor.apply(snapshots[0]['actor'],jp.asarray(normalize(obs[1],snapshots[2]['norm'])))[0]))
    assert np.max(np.abs(actual[1]-wrong))>1e-4


def test_sampler_and_attempted_corner_credits_survive_resume(prepared):
    saved=prepared[1];rng=restore_sampler(saved['pool_sampler_state'])
    assert draw_opponent(rng,False)==-1
    checkpoint=copy.deepcopy(rng.bit_generator.state)
    draws=[draw_opponent(rng,True) for _ in range(100)]
    restored=restore_sampler(checkpoint)
    assert draws==[draw_opponent(restored,True) for _ in range(100)] and set(draws)=={0,1,2}
    credits=np.zeros((3,2),dtype=np.int64)
    mixed=np.asarray(saved['mixed_credits_by_mode_and_corner']).copy()
    for i,index in enumerate(draws):credits[index,i%2]+=1;mixed[1,i%2]+=1
    np.testing.assert_array_equal(validate_pool_ledger(credits,mixed),credits)
    for bad in (credits+.1,credits[:2],credits-1):
        with pytest.raises(ValueError):validate_pool_ledger(bad,mixed)
    mixed[1]+=np.array([1,-1])
    with pytest.raises(ValueError):validate_pool_ledger(credits,mixed)


@pytest.mark.parametrize('seed,active',[(22,False),(23,True)])
def test_pool_wrapper_preserves_unassisted_action_routing(foundation,seed,active):
    from boxer.striking_contact import ContactAwareStrikingEnv
    motor=lambda obs,key:(jp.zeros((2,29)),{})
    env=ContactPoolMixedStrikingEnv(foundation,None,motor=motor,active_opponent=active)
    ref=ContactAwareStrikingEnv(foundation,None,motor=motor)
    corner=seed%2
    np.testing.assert_array_equal(env.reset(seed),ref.reset(seed)[corner:corner+1])
    learner=np.linspace(-.1,.1,17)[None];opponent=np.linspace(.05,-.05,17)
    paired=np.zeros((2,17));paired[corner]=learner[0]
    if active:paired[1-corner]=opponent
    for _ in range(3):
        obs,reward,done,_=env.step(learner,opponent if active else None)
        expected,rewards,stop,_=ref.step(paired)
        np.testing.assert_array_equal(obs,expected[corner:corner+1])
        np.testing.assert_array_equal(reward,rewards[corner:corner+1])
        np.testing.assert_array_equal(env.env.pair.data.qpos,ref.pair.data.qpos)
        assert done==stop
    assert env.env.pair.model.neq==0 and not np.any(env.env.pair.model.body_gravcomp)
    assert not np.any(env.env.pair.data.qfrc_applied) and not np.any(env.env.pair.data.xfrc_applied)
