"""Read-only short-rollout credit and retention gradient assay; no Adam updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,contract,utc
from boxer.striking_contact import ContactAwareStrikingEnv
from boxer.striking_ppo_warmstart import anchor_loss,validate_split_checkpoint
from boxer.striking_train import Actor,Critic,normalize,log_probability,advantages
from tools.striking_contact_compare import load_checked,load_demonstrations,ORIGINAL,ORIGINAL_SHA
from tools.striking_demonstration_init import behavior_loss
from boxer.striking_anchor import load_anchor_candidates


def vector(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)])


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'gradients.json'
        if path.exists():raise ValueError('Existing gradient assay; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/postppo-sampling-bc5000/analysis.json').read_text())
        assert review['formal_world_controls_total']==546358
        foundation=UnassistedG1('approach',impl='jax')
        motor,metadata=load_policy_params(Path('checkpoints/curriculum/approach'))
        assert contract(foundation)==metadata['contract']
        data,anchor_manifest=load_anchor_candidates();dobs,labels,flags,files=load_demonstrations()
        original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
        actor=Actor();critic=Critic()
        predict=jax.jit(lambda p,x:actor.apply(p,x))
        value=jax.jit(lambda p,x:critic.apply(p,x))
        baseline=np.asarray(jp.tanh(predict(original['actor'],jp.array(normalize(dobs,original['norm'])))[0]))
        weights=np.where(np.max(np.abs(labels-baseline),axis=1)>.05,5.,1.).astype(np.float32)
        rng=np.random.default_rng(9091);di=rng.integers(0,7500,192);ai=rng.integers(0,1600,64)
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            maximum_world_controls=1536,diagnostic_world_controls=0,new_optimizer_updates=0,
            new_ppo_training_credits=0,diagnostic_ppo_retained=1450240,
            formal_world_controls_before_job=546358,assistance=False,promotion_eligible=False,
            policy_adoption='none',seeds=list(range(6000003,6000007)),unroll=128,worlds=4,
            anchor_sha256=anchor_manifest['sha256'],demonstration_files=files,variants=[])
        atomic_json(path,state)
        try:
            for name,info in review['variants'].items():
                folder=Path('checkpoints/striking-validation/ppo-anchor-bc5000')/('migration' if name=='source' else name)
                saved,manifest=load_checked(folder,info['sha256']);validate_split_checkpoint(saved)
                assert saved['foundation_sha256']==metadata['sha256']
                config=json.loads((folder/'config.json').read_text())
                envs=[ContactAwareStrikingEnv(foundation,motor,engagement_distance=config['engagement_distance'],
                    engagement_reward_distance=config['engagement_reward_distance']) for _ in range(4)]
                for env in envs[1:]:env.motor=envs[0].motor
                observations=np.stack([env.reset(seed,swap=bool(i%2)) for i,(env,seed) in enumerate(zip(envs,state['seeds']))]).reshape(8,148)
                key=saved['key'];episodes=np.zeros(4,dtype=int);storage={k:[] for k in ('obs','raw','logp','reward','done','value')}
                events=np.zeros(2);falls=0
                state.update(active_variant=name,phase='collecting',updated_utc=utc());atomic_json(path,state)
                for t in range(128):
                    normalized=normalize(observations,saved['norm']);mean,std=predict(saved['actor'],jp.array(normalized))
                    key,sub=jax.random.split(key);raw=mean+jp.exp(std)*jax.random.normal(sub,mean.shape)
                    logp=np.asarray(log_probability(raw,mean,std));actions=np.asarray(jp.tanh(raw)).reshape(4,2,17)
                    rewards=[];dones=[];following=[]
                    for i,env in enumerate(envs):
                        state['diagnostic_world_controls']+=1;atomic_json(path,state)
                        nxt,reward,done,outcome=env.step(actions[i]);events+=outcome['scored']
                        rewards.extend(reward);dones.extend([done]*2)
                        if done:
                            falls+=int(outcome['done']);episodes[i]+=1
                            nxt=env.reset(state['seeds'][i]+1009*int(episodes[i]),swap=bool(i%2))
                        following.append(nxt)
                    for k,v in dict(obs=observations.copy(),raw=np.asarray(raw),logp=logp,reward=np.asarray(rewards),
                            done=np.asarray(dones,dtype=np.float32),value=np.asarray(value(saved['critic'],jp.array(normalized)))).items():storage[k].append(v)
                    observations=np.asarray(following).reshape(8,148)
                rollout={k:np.asarray(v) for k,v in storage.items()}
                bootstrap=np.asarray(value(saved['critic'],jp.array(normalize(observations,saved['norm']))))
                adv,returns=advantages(rollout['reward'],rollout['done'],rollout['value'],bootstrap)
                batch=dict(obs=normalize(rollout['obs'].reshape(1024,148),saved['norm']),raw=rollout['raw'].reshape(1024,17),
                    logp=rollout['logp'].reshape(1024),adv=((adv-adv.mean())/(adv.std()+1e-8)).reshape(1024),returns=returns.reshape(1024))
                np.savez_compressed(args.output/(name+'-batch.npz'),**batch,**{'rollout_'+k:v for k,v in rollout.items()},
                    bootstrap=bootstrap,demo_indices=di,anchor_indices=ai)
                def policy_loss(p):
                    m,s=actor.apply(p,jp.array(batch['obs']));ratio=jp.exp(log_probability(batch['raw'],m,s)-batch['logp'])
                    return -jp.mean(jp.minimum(ratio*batch['adv'],jp.clip(ratio,.8,1.2)*batch['adv']))
                losses={'ppo_policy':policy_loss,
                    'entropy':lambda p:-.001*jp.sum(actor.apply(p,jp.array(batch['obs']))[1]+.5*np.log(2*np.pi*np.e)),
                    'approach_anchor':lambda p:anchor_loss(actor,p,jp.array(normalize(data['observation'][ai],saved['norm'])),jp.array(data['action'][ai])),
                    'strike_demo':lambda p:behavior_loss(actor,p,jp.array(normalize(np.c_[dobs,flags][di],saved['norm'])),jp.array(labels[di]),jp.array(weights[di]))}
                gradients={};measurements={}
                for label,fn in losses.items():
                    loss,g=jax.value_and_grad(fn)(saved['actor']);g=vector(g)
                    assert np.isfinite(g).all() and np.isfinite(float(loss))
                    gradients[label]=g;measurements[label]=dict(loss=float(loss),gradient_l2=float(np.linalg.norm(g)))
                cosine=lambda a,b:float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
                for label in ('approach_anchor','strike_demo'):
                    measurements[label]['cosine_with_ppo_policy']=cosine(gradients[label],gradients['ppo_policy'])
                batch_payload=(args.output/(name+'-batch.npz')).read_bytes()
                state['variants'].append(dict(variant=name,sha256=manifest['sha256'],batch_sha256=hashlib.sha256(batch_payload).hexdigest(),
                    new_world_controls=512,new_training_updates=0,measurements=measurements,events_by_corner=events.tolist(),
                    fall_rounds=falls,mean_return=float(returns.mean()),return_std=float(returns.std()),
                    initial_value_mean=float(rollout['value'].mean()),value_mse=float(np.mean((returns-rollout['value'])**2))))
                # Verify this assay never rewrites any saved policy.
                load_checked(folder,info['sha256']);atomic_json(path,state)
            assert state['diagnostic_world_controls']==1536
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
