"""Bounded native-physics self-play PPO over the certified frozen motor policy.

JAX stays on CPU. Both corners learn one shared opponent-aware residual policy.
Foundation steps, boxing learner steps and physical world steps remain distinct.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import shutil
import time

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, load_policy_params, utc
from boxer.curriculum_control import promotion_ready
from boxer.striking_env import StrikingEnv


class Actor(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x=nn.tanh(nn.Dense(128)(obs))
        x=nn.tanh(nn.Dense(128)(x))
        mean=nn.Dense(17,kernel_init=nn.initializers.zeros)(x)
        log_std=self.param('log_std',nn.initializers.constant(-1.9),(17,))
        return mean,jp.clip(log_std,-3.,-.5)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x=nn.tanh(nn.Dense(128)(obs))
        x=nn.tanh(nn.Dense(128)(x))
        return nn.Dense(1)(x)[...,0]


def log_probability(raw, mean, log_std):
    return -.5*jp.sum(((raw-mean)/jp.exp(log_std))**2 + 2*log_std + np.log(2*np.pi),axis=-1)


def advantages(rewards, dones, values, bootstrap, gamma=.99, lam=.95):
    """Time-major GAE; both first falls and the round clock cut bootstrapping."""
    result=np.zeros_like(rewards)
    carry=np.zeros_like(bootstrap)
    following=bootstrap
    for t in reversed(range(len(rewards))):
        continuation=1.-dones[t]
        delta=rewards[t]+gamma*following*continuation-values[t]
        carry=delta+gamma*lam*continuation*carry
        result[t]=carry
        following=values[t]
    return result,result+values


def normalize(obs, norm):
    return np.clip((obs-norm['mean'])/np.sqrt(norm['var']+.01),-10.,10.).astype(np.float32)


def update_norm(norm, obs):
    n=len(obs); delta=obs.mean(axis=0)-norm['mean'];total=norm['count']+n
    variance=(norm['var']*norm['count']+obs.var(axis=0)*n
              +delta**2*norm['count']*n/total)/total
    return dict(mean=norm['mean']+delta*n/total,var=variance,count=total)


def main(argv=None):
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/striking-pilot'))
    parser.add_argument('--foundation',type=Path,default=Path('checkpoints/curriculum/approach'))
    parser.add_argument('--ceiling',type=int,default=32768)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--unroll',type=int,default=128)
    parser.add_argument('--seed',type=int,default=53)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--publish-dir',type=Path,
                        help='optional repository mirror; durable per-step ledger stays in output')
    args=parser.parse_args(argv)
    if args.ceiling <= 0 or args.worlds <= 0 or args.unroll < 2:
        raise ValueError('positive ceiling/worlds and unroll >=2 required')
    if jax.devices()[0].platform!='cpu':
        raise ValueError('native boxing pilot requires JAX_PLATFORMS=cpu')
    args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (args.output/'.lock',Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(path.open('w'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            return train(args)
        except BaseException as error:
            state_path=args.output/'state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else {}
            atomic_json(state_path,dict(state,status='failed',error=str(error),updated_utc=utc()))
            if args.publish_dir:
                args.publish_dir.mkdir(parents=True,exist_ok=True)
                shutil.copy2(state_path,args.publish_dir/'state.json')
            raise


def train(args):
    foundation=UnassistedG1('approach',impl='jax')
    for stage in ('balance','walk_easy','walk','approach'):
        source=json.loads((args.foundation.parent/stage/'latest.json').read_text())
        history=json.loads((Path('docs/results/curriculum')/(stage+'-gate_history.json')).read_text())
        if (not promotion_ready(history,stage)
                or history[-1]['checkpoint_sha256']!=source['sha256']
                or history[-1]['stage_steps']!=source['stage_steps']):
            raise ValueError('all four complete current foundation certifications required')
    motor_params,metadata=load_policy_params(args.foundation)
    if contract(foundation)!=metadata['contract']:
        raise ValueError('frozen motor contract mismatch')
    config=dict(version=1,foundation_sha256=metadata['sha256'],foundation_main_steps=metadata['total_steps'],
        worlds=args.worlds,unroll=args.unroll,seed=args.seed,actor_inputs=146,actions=17,
        arm_residual_radians=.8,command_offsets=[.15,.15,.25],assistance=False,
        dt=.02,physics_dt=.002,round_steps=500,learning_rate=.0001,epochs=4,
        minibatches=4,gamma=.99,gae_lambda=.95,scorecard_version=1)
    config_path=args.output/'config.json'
    checkpoint_path=args.output/'checkpoint/latest.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text())!=config:
            raise ValueError('existing pilot requires --resume with the exact same configuration')
    atomic_json(config_path,config)
    actor,critic=Actor(),Critic()
    key=jax.random.PRNGKey(args.seed)
    ap=actor.init(key,jp.zeros((1,146)));vp=critic.init(jax.random.fold_in(key,1),jp.zeros((1,146)))
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
    optimizer=tx.init((ap,vp))
    norm=dict(mean=np.zeros(146),var=np.ones(146),count=1e-4)
    consumed=optimized=0
    if args.resume and checkpoint_path.exists():
        manifest=json.loads(checkpoint_path.read_text())
        payload=(checkpoint_path.parent/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:
            raise ValueError('boxing checkpoint checksum mismatch')
        saved=pickle.loads(payload)
        if saved['foundation_sha256']!=metadata['sha256']:
            raise ValueError('boxing motor provenance mismatch')
        ap,vp,optimizer,norm,key=(saved[k] for k in ('actor','critic','optimizer','norm','key'))
        optimized=saved['optimized_steps']
        progress_path=args.output/'progress.json'
        consumed=max(saved['consumed_steps'],json.loads(progress_path.read_text())['consumed_steps'])
    elif args.resume:
        # A failure before the first PPO update still consumed simulation.
        # Keep that ledger when recreating the original neutral head.
        consumed=json.loads((args.output/'progress.json').read_text())['consumed_steps']
    rows=args.worlds*2
    block=rows*args.unroll
    if block%4:
        raise ValueError('rollout must divide into four minibatches')
    stopping=[False]
    def stop(signum,frame):
        stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    @jax.jit
    def act(parameters, obs, rng):
        mean,log_std=actor.apply(parameters,obs)
        raw=mean+jp.exp(log_std)*jax.random.normal(rng,mean.shape)
        return jp.tanh(raw),raw,log_probability(raw,mean,log_std)

    @jax.jit
    def value(parameters,obs):
        return critic.apply(parameters,obs)

    @jax.jit
    def update(ap,vp,optimizer,batch):
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

    envs=[StrikingEnv(foundation,motor_params) for _ in range(args.worlds)]
    # Share the already compiled frozen network across worlds.
    for env in envs[1:]:env.motor=envs[0].motor
    episode=np.full(args.worlds,consumed//rows,dtype=np.int64)
    obs=np.stack([env.reset(args.seed+1009*i+int(episode[i]),swap=bool(i%2)) for i,env in enumerate(envs)]).reshape(rows,146)
    state=dict(status='running',pid=os.getpid(),started_utc=utc(),ceiling=args.ceiling,
        foundation_steps=metadata['total_steps'],foundation_sha256=metadata['sha256'],
        assistance=False,optimizer_saved=True,physics_state_saved=False,
        discarded_collected_steps=consumed-optimized)
    started=time.monotonic()
    session_start=consumed
    completed_rounds=falls=0
    scored=np.zeros(2)
    def record(status='running',**extra):
        progress=dict(consumed_steps=consumed,optimized_steps=optimized,
            world_control_steps=consumed//2,combined_project_consumed_steps=metadata['total_steps']+consumed,
            updated_utc=utc(),rounds=completed_rounds,fall_rounds=falls,scored_by_corner=scored.tolist(),
            actor_steps_per_second=(consumed-session_start)/max(time.monotonic()-started,1e-6),**extra)
        atomic_json(args.output/'progress.json',progress)
        atomic_json(args.output/'state.json',dict(state,status=status,**progress))
    def publish():
        if args.publish_dir is None:
            return
        if args.publish_dir.resolve()==args.output.resolve():
            raise ValueError('publish directory must differ from the durable output')
        args.publish_dir.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','state.json','progress.json'):
            atomic_bytes(args.publish_dir/name,(args.output/name).read_bytes())
        if checkpoint_path.exists():
            manifest=json.loads(checkpoint_path.read_text())
            destination=args.publish_dir/'checkpoint'
            atomic_bytes(destination/manifest['file'],(checkpoint_path.parent/manifest['file']).read_bytes())
            atomic_json(destination/'latest.json',manifest)
    record()
    publish()
    while consumed+block <= args.ceiling and not stopping[0]:
        storage={k:[] for k in ('obs','raw','logp','reward','done','value')}
        for t in range(args.unroll):
            normalized=normalize(obs,norm)
            key,sub=jax.random.split(key)
            action,raw,logp=act(ap,jp.array(normalized),sub)
            predicted=np.asarray(value(vp,jp.array(normalized)))
            actions=np.asarray(action).reshape(args.worlds,2,17)
            rewards=[];dones=[];following=[]
            for i,env in enumerate(envs):
                consumed+=2
                # Count attempted physics even if a later world/update fails.
                record(phase='collecting')
                next_obs,reward,done,outcome=env.step(actions[i])
                rewards.extend(reward);dones.extend([done,done])
                scored+=outcome['scored']
                if done:
                    completed_rounds+=1;falls+=int(outcome['done'])
                    episode[i]+=1
                    next_obs=env.reset(args.seed+1009*i+int(episode[i]),swap=bool(episode[i]%2))
                following.append(next_obs)
            for k,v in dict(obs=obs.copy(),raw=np.asarray(raw),logp=np.asarray(logp),
                reward=np.asarray(rewards),done=np.asarray(dones,dtype=np.float32),value=predicted).items():
                storage[k].append(v)
            obs=np.asarray(following).reshape(rows,146)
        rollout={k:np.asarray(v) for k,v in storage.items()}
        bootstrap=np.asarray(value(vp,jp.array(normalize(obs,norm))))
        adv,returns=advantages(rollout['reward'],rollout['done'],rollout['value'],bootstrap)
        flat_obs=rollout['obs'].reshape(block,146)
        batch=dict(obs=normalize(flat_obs,norm),raw=rollout['raw'].reshape(block,17),
            logp=rollout['logp'].reshape(block),adv=((adv-adv.mean())/(adv.std()+1e-8)).reshape(block),
            returns=returns.reshape(block))
        losses=[]
        for epoch in range(4):
            key,sub=jax.random.split(key)
            order=np.asarray(jax.random.permutation(sub,block))
            for indices in np.split(order,4):
                ap,vp,optimizer,loss=update(ap,vp,optimizer,{k:jp.array(v[indices]) for k,v in batch.items()})
                losses.append(float(loss))
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((ap,vp,optimizer))):
            raise FloatingPointError('non-finite boxing update; prior checkpoint retained')
        norm=update_norm(norm,flat_obs)
        optimized+=block
        saved=dict(actor=ap,critic=vp,optimizer=optimizer,norm=norm,key=key,
            optimized_steps=optimized,consumed_steps=consumed,foundation_sha256=metadata['sha256'])
        payload=pickle.dumps(jax.device_get(saved),protocol=pickle.HIGHEST_PROTOCOL)
        name=f'step-{consumed}.pkl'
        atomic_bytes(checkpoint_path.parent/name,payload)
        atomic_json(checkpoint_path,dict(file=name,sha256=hashlib.sha256(payload).hexdigest(),
            consumed_steps=consumed,optimized_steps=optimized,foundation_sha256=metadata['sha256']))
        record(phase='updated',mean_loss=float(np.mean(losses)))
        publish()
        print(json.dumps(json.loads((args.output/'progress.json').read_text())),flush=True)
    record(status='stopped' if stopping[0] else 'budget_reached',phase='stopped')
    publish()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
