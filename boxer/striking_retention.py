"""Explicit demonstration-retention profile over an unchanged PPO warm start."""
import copy
import hashlib

import jax
import jax.numpy as jp
import numpy as np

from boxer.striking_anchor import load_anchor_candidates
from boxer.striking_ppo_warmstart import contact_config,validate_split_checkpoint
from boxer.striking_train import Actor,normalize


def retention_loss(actor,params,observations,labels,row_weights):
    prediction=jp.tanh(actor.apply(params,observations)[0])
    channels=jp.array([2.]*3+[4.]*7+[1.]*7)
    return jp.mean(jp.mean((prediction-labels)**2*channels,axis=-1)*row_weights)


def load_retention_data():
    from tools.striking_contact_compare import load_demonstrations,load_checked,ORIGINAL,ORIGINAL_SHA
    anchors,manifest=load_anchor_candidates()
    obs,labels,flags,files=load_demonstrations()
    original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
    actor=Actor()
    predicted=np.asarray(jp.tanh(actor.apply(original['actor'],jp.array(normalize(obs,original['norm'])))[0]))
    weights=np.where(np.max(np.abs(labels-predicted),axis=1)>.05,5.,1.).astype(np.float32)
    data=dict(demo_obs=np.c_[obs,flags],demo_labels=labels,demo_weights=weights,
        anchor_obs=anchors['observation'],anchor_labels=anchors['action'])
    provenance=dict(anchor_sha256=manifest['sha256'],demonstration_files=files,
        row_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        weight_source_sha256=ORIGINAL_SHA,demo_rows=7500,anchor_rows=1600,
        demo_batch_rows=192,anchor_batch_rows=64)
    return data,provenance


def prepare_retention_state(saved,source_sha,weight,provenance):
    validate_split_checkpoint(saved)
    if (weight not in (0.,1.) or saved['consumed_steps']!=saved['optimized_steps'] or
            saved['optimized_steps']!=1572864 or saved.get('anchor_loss_weight')!=0. or
            int(saved['optimizer']['actor_adam'][0].count)!=7048 or
            int(saved['optimizer']['critic_adam'][0].count)!=30720):
        raise ValueError('latest completed PPO-only source with preserved independent counts required')
    result=copy.deepcopy(saved)
    result.update(demonstration_ppo=True,retention_loss_weight=weight,
        retention_sampler_state=copy.deepcopy(saved['aux_sampler_state']),
        retention_provenance=copy.deepcopy(provenance),
        retention_transition=dict(source_sha256=source_sha,consumed_steps=1572864,
            world_control_steps=saved['world_control_steps'],actor_count=7048,critic_count=30720,
            optimizer_reset=False,physics_or_optimizer_steps_at_transition=0))
    return result


def retention_config(source,weight):
    if weight not in (0.,1.):raise ValueError('only reviewed retention weights allowed')
    return dict(contact_config(source,0.),version=3,demonstration_ppo=True,retention_loss_weight=weight)


def validate_retention_state(saved,provenance):
    validate_split_checkpoint(saved)
    if (saved.get('demonstration_ppo') is not True or saved.get('retention_loss_weight') not in (0.,1.) or
            saved.get('anchor_loss_weight')!=0. or saved.get('retention_provenance')!=provenance or
            saved.get('retention_transition',{}).get('consumed_steps')!=1572864):
        raise ValueError('explicit reviewed demonstration profile/provenance required')
    transition=saved['retention_transition']
    if (transition.get('actor_count')!=7048 or transition.get('critic_count')!=30720 or
            transition.get('world_control_steps')!=983040 or transition.get('optimizer_reset') is not False or
            len(transition.get('source_sha256',''))!=64 or
            int(saved['optimizer']['actor_adam'][0].count)<7048 or
            int(saved['optimizer']['critic_adam'][0].count)<30720):
        raise ValueError('retention continuation lost optimizer or world-control prefix')
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved['retention_sampler_state'])


def retention_batch(data,rng,norm):
    di=rng.integers(0,7500,192);ai=rng.integers(0,1600,64)
    return dict(retention_obs=jp.array(normalize(np.r_[data['demo_obs'][di],data['anchor_obs'][ai]],norm)),
        retention_labels=jp.array(np.r_[data['demo_labels'][di],data['anchor_labels'][ai]]),
        retention_weights=jp.array(np.r_[data['demo_weights'][di],np.ones(64,np.float32)]))
