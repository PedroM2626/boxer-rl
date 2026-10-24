"""Frozen-policy matched gradient/critic assay after a pass followed by regression."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, contract, utc
from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT, fingerprint
from boxer.striking_mixed_partner import ContactMixedStrikingEnv, MIXED_ACTIVE_WORLDS
from boxer.striking_retention import load_retention_data, retention_batch, retention_loss
from boxer.striking_train import Actor, Critic, normalize, log_probability, advantages, require_ppo_checkpoint
from tools.striking_contact_compare import load_checked

JOBS = (
    ('source', 'gae-1966080/long_gae/gate-16384', '95dbdff01def598cdc907e93f1dc7bceb2f1d6b58bc22f488a1936afa86553f9'),
    ('passing', 'gae-continue-1982464/gate-8192', 'fa8080110edee9b73f2272f4a8d6f94bebc640a0fd56b04f2e66c8ce44c12548'),
    ('final', 'gae-continue-1982464/gate-16384', '3052e0f69fcadc823be14069ed462ee3057ae75a8028cb2d8aa5bf3615aa48b5'))


def vector(tree):
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in jax.tree_util.tree_leaves(tree)])


def policy_loss(actor, params, batch):
    mean, std = actor.apply(params, batch['obs'])
    ratio = jp.exp(log_probability(batch['raw'], mean, std)-batch['logp'])
    return -jp.mean(jp.minimum(ratio*batch['adv'], jp.clip(ratio,.8,1.2)*batch['adv']))


def cosine(a, b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for filename in (args.output/'.lock',Path('runs/unassisted_boxing.lock')):
            handle=stack.enter_context(filename.open('w'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'gradients.json'
        if path.exists():raise ValueError('Gradient assay exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/gae-continue-1982464/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==2065664 and review['formal_world_controls_total']==1792372
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            maximum_world_controls=12288,diagnostic_world_controls=0,new_optimizer_updates=0,new_ppo_training_credits=0,
            new_supervised_updates=0,new_diagnostic_retention_rows=0,diagnostic_ppo_retained=2065664,formal_world_controls_before_job=1792372,
            formal_world_controls_ceiling=1804660,main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            assistance=False,policy_adoption='none',promotion=False,worlds=8,blocks=4,unroll=128,
            reset_seeds=list(range(28000003,28000011)),common_policy_noise_seed=314159,common_retention_seed=9091,
            frozen_normalizers=True,variants=[])
        atomic_json(path,state)
        try:
            foundation=UnassistedG1('approach',impl='jax')
            motor,metadata=load_policy_params(Path('checkpoints/curriculum/approach'))
            assert contract(foundation)==metadata['contract']
            data,provenance=load_retention_data();actor=Actor();critic=Critic()
            predict=jax.jit(lambda p,x:actor.apply(p,x));value=jax.jit(lambda p,x:critic.apply(p,x))
            pg=jax.jit(jax.value_and_grad(lambda p,b:policy_loss(actor,p,b)))
            eg=jax.jit(jax.value_and_grad(lambda p,x:-.001*jp.sum(actor.apply(p,x)[1]+.5*np.log(2*np.pi*np.e))))
            rg=jax.jit(jax.value_and_grad(lambda p,b:retention_loss(actor,p,b['retention_obs'],b['retention_labels'],b['retention_weights'])))
            vg=jax.jit(jax.value_and_grad(lambda p,b:.25*jp.mean((critic.apply(p,b['obs'])-b['returns'])**2)))
            modes=np.tile([int(i in MIXED_ACTIVE_WORLDS) for i in range(8)],128)
            compiled_motor=None
            for name,relative,sha in JOBS:
                folder=Path('checkpoints/striking-validation')/relative;saved,_=load_checked(folder,sha);require_ppo_checkpoint(saved)
                assert saved['foundation_sha256']==metadata['sha256'] and saved['retention_provenance']==provenance
                assert saved.get('long_gae_ppo') and saved['training_gae_lambda']==.99 and fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
                config=json.loads((folder/'config.json').read_text());assert config['version']==12 and config['gamma']==.999 and config['gae_lambda']==.99
                envs=[ContactMixedStrikingEnv(foundation,motor,active_opponent=i in MIXED_ACTIVE_WORLDS,
                    engagement_distance=config['engagement_distance'],engagement_reward_distance=config['engagement_reward_distance']) for i in range(8)]
                if compiled_motor is None:compiled_motor=envs[0].motor
                for env in envs:env.motor=compiled_motor
                episode=np.zeros(8,dtype=int)
                obs=np.stack([env.reset(seed,swap=bool(i%2)) for i,(env,seed) in enumerate(zip(envs,state['reset_seeds']))]).reshape(8,148)
                key=jax.random.PRNGKey(314159);retention_rng=np.random.default_rng(9091)
                variant=dict(variant=name,sha256=sha,blocks=[],scored_by_corner=[0.,0.],fall_rounds=0,completed_rounds=0)
                state['variants'].append(variant)
                for block in range(4):
                    state.update(active_variant=name,active_block=block,phase='collecting',updated_utc=utc());atomic_json(path,state)
                    storage={k:[] for k in ('obs','raw','logp','reward','done','value')}
                    for _ in range(128):
                        normalized=normalize(obs,saved['norm']);mean,std=predict(saved['actor'],jp.array(normalized))
                        key,sub=jax.random.split(key);raw=mean+jp.exp(std)*jax.random.normal(sub,mean.shape)
                        actions=np.asarray(jp.tanh(raw)).reshape(8,1,17)
                        frozen=saved['frozen_opponent'];other_obs=np.stack([env.opponent_observation for env in envs])
                        other=np.asarray(jp.tanh(predict(frozen['actor'],jp.array(normalize(other_obs,frozen['norm'])))[0]))
                        rewards=[];dones=[];following=[]
                        for i,env in enumerate(envs):
                            state['diagnostic_world_controls']+=1;assert state['diagnostic_world_controls']<=12288;atomic_json(path,state)
                            nxt,reward,done,outcome=env.step(actions[i],other[i] if env.active_opponent else None)
                            rewards.extend(reward);dones.append(done)
                            variant['scored_by_corner']=(np.array(variant['scored_by_corner'])+outcome['scored']).tolist()
                            if done:
                                variant['completed_rounds']+=1;variant['fall_rounds']+=int(outcome['done']);episode[i]+=1
                                nxt=env.reset(state['reset_seeds'][i]+1009*int(episode[i]),swap=bool(episode[i]%2))
                            following.append(nxt)
                        row=dict(obs=obs.copy(),raw=np.asarray(raw),logp=np.asarray(log_probability(raw,mean,std)),reward=np.asarray(rewards),
                            done=np.asarray(dones,dtype=np.float32),value=np.asarray(value(saved['critic'],jp.array(normalized))))
                        for k,v in row.items():storage[k].append(v)
                        obs=np.asarray(following).reshape(8,148)
                    rollout={k:np.asarray(v) for k,v in storage.items()}
                    bootstrap=np.asarray(value(saved['critic'],jp.array(normalize(obs,saved['norm']))))
                    adv,returns=advantages(rollout['reward'],rollout['done'],rollout['value'],bootstrap,gamma=.999,lam=.99)
                    batch=dict(obs=jp.array(normalize(rollout['obs'].reshape(1024,148),saved['norm'])),raw=jp.array(rollout['raw'].reshape(1024,17)),
                        logp=jp.array(rollout['logp'].reshape(1024)),adv=jp.array(((adv-adv.mean())/(adv.std()+1e-8)).reshape(1024)),returns=jp.array(returns.reshape(1024)))
                    replay=retention_batch(data,retention_rng,saved['norm'])
                    state['new_diagnostic_retention_rows']+=256
                    assert all(np.isfinite(x).all() for x in (*rollout.values(),*batch.values(),*replay.values(),bootstrap))
                    state.update(phase='measuring_gradients',updated_utc=utc());atomic_json(path,state)
                    batch_path=args.output/f'{name}-block-{block}.npz'
                    np.savez_compressed(batch_path,**{k:np.asarray(v) for k,v in batch.items()},**{k:np.asarray(v) for k,v in replay.items()},
                        **{'rollout_'+k:v for k,v in rollout.items()},bootstrap=bootstrap,mode_labels=modes)
                    measurements={};gradients={}
                    for label,fn,params,arg in (('ppo',pg,saved['actor'],batch),('entropy',eg,saved['actor'],batch['obs']),
                            ('retention',rg,saved['actor'],replay),('critic',vg,saved['critic'],batch)):
                        loss,g=fn(params,arg);v=vector(g)
                        assert np.isfinite(v).all() and np.isfinite(loss)
                        gradients[label]=v;measurements[label]=dict(loss=float(loss),gradient_l2=float(np.linalg.norm(v)))
                    for mode in (0,1):
                        subset={k:v[modes==mode] for k,v in batch.items()};loss,g=pg(saved['actor'],subset);v=vector(g)
                        measurements['neutral' if mode==0 else 'active']=dict(loss=float(loss),gradient_l2=float(np.linalg.norm(v)),
                            retention_cosine=cosine(v,gradients['retention']),mean_reward=float(rollout['reward'].reshape(-1)[modes==mode].mean()),
                            value_mse=float(np.mean((returns.reshape(-1)[modes==mode]-rollout['value'].reshape(-1)[modes==mode])**2)))
                    actor_sum=gradients['ppo']+gradients['entropy']+gradients['retention']
                    joint=float(np.sqrt(np.dot(actor_sum,actor_sum)+np.dot(gradients['critic'],gradients['critic'])))
                    variant['blocks'].append(dict(block=block,batch_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),measurements=measurements,
                        retention_ppo_cosine=cosine(gradients['retention'],gradients['ppo']),joint_gradient_l2=joint,
                        hypothetical_global_clip_scale=min(1.,1./max(joint,1e-12)),mean_return=float(returns.mean()),
                        return_std=float(returns.std()),value_mse=float(np.mean((returns-rollout['value'])**2))))
                    load_checked(folder,sha);atomic_json(path,state)
            assert state['diagnostic_world_controls']==12288
            state.update(status='completed',phase='awaiting_review',updated_utc=utc(),
                limitations='Frozen normalizers and policies across four sequential blocks, unlike updating PPO. Each policy follows its own sampled trajectory under common exogenous seeds. Gradients are full-batch pre-update measurements, not the sixteen actual minibatch updates. Retention draws use a common independent copied-data sampler; saved sampler/RNG/Adam leaves are unchanged. Gradient scale/alignment and critic return error are descriptive, not a causal remedy or a survival gate.')
            atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
