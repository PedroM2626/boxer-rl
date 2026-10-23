"""Bounded offline actor initialization from exactly replayed free-body actions.

This does not resume PPO, modify the motor, or deploy the geometric teacher at
evaluation. BC optimizer state is separate and explicitly incompatible with PPO.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import Actor, normalize


def behavior_loss(actor, params, obs, labels, row_weights):
    prediction=jp.tanh(actor.apply(params,obs)[0])
    # Four times the weight for the demonstrated left arm, twice for commands.
    channels=jp.array([2.]*3+[4.]*7+[1.]*7)
    return jp.mean(jp.mean((prediction-labels)**2*channels,axis=-1)*row_weights)


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'initialization.json'
        if path.exists():raise ValueError('initialization exists; inspect before recovery')
        if jax.devices()[0].platform!='cpu':raise ValueError('CPU initialization required')
        source=Path('checkpoints/striking-validation/partner-duration-1441792')
        manifest=json.loads((source/'latest.json').read_text());payload=(source/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('source checksum mismatch')
        saved=pickle.loads(payload)
        dataset=Path('docs/results/striking-pilot/directed-holdout-1441792')
        replay=json.loads((dataset/'replay.json').read_text())
        trials=json.loads((dataset/'feasibility.json').read_text())
        if (len(replay['trials'])!=15 or replay['source_sha256']!=manifest['sha256'] or
                trials['source_sha256']!=manifest['sha256']):raise ValueError('verified demonstration source mismatch')
        observations=[];actions=[];files={}
        for trial in replay['trials']:
            file=dataset/trial['trajectory'];b=file.read_bytes()
            if (hashlib.sha256(b).hexdigest()!=trial['sha256'] or not trial['observations_match_exactly'] or
                    not trial['events_match_exactly'] or not trial['survived']):raise ValueError('demonstration integrity mismatch')
            arrays=np.load(file);obs=arrays['observation'];act=arrays['action']
            if obs.shape!=(500,146) or act.shape!=(500,17) or not np.isfinite(obs).all() or not np.isfinite(act).all() or np.max(np.abs(act))>1.:
                raise ValueError('invalid complete demonstration arrays')
            observations.append(obs);actions.append(act);files[file.name]=trial['sha256']
        observations=np.concatenate(observations);labels=np.concatenate(actions)
        actor=Actor();params=saved['actor'];obs=normalize(observations,saved['norm'])
        inference=jax.jit(lambda p,x:jp.tanh(actor.apply(p,x)[0]))
        baseline=np.asarray(inference(params,jp.array(obs)))
        weights=np.where(np.max(np.abs(labels-baseline),axis=1)>.05,5.,1.).astype(np.float32)
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001));optimizer=tx.init(params)
        rng=np.random.default_rng(313)
        @jax.jit
        def update(params,optimizer,x,y,w):
            loss,gradient=jax.value_and_grad(lambda p:behavior_loss(actor,p,x,y,w))(params)
            changes,optimizer=tx.update(gradient,optimizer,params)
            return optax.apply_updates(params,changes),optimizer,loss
        state=dict(status='running',phase='offline_training',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=manifest['sha256'],
            source_lineage_steps=manifest['consumed_steps'],demonstration_files=files,demonstration_rows=len(obs),
            maximum_supervised_updates=2000,batch_size=256,supervised_updates=0,data_presentations=0,
            new_ppo_transitions=0,diagnostic_ppo_retained=1188096,formal_world_controls_before_job=346091,
            diagnostic_evaluation_world_controls=0,maximum_evaluation_world_controls=24000,
            teacher_used_at_evaluation=False,policy_adoption='none',checks=[])
        atomic_json(path,state)
        try:
            for step in range(1,2001):
                ix=rng.integers(0,len(obs),256)
                params,optimizer,loss=update(params,optimizer,jp.array(obs[ix]),jp.array(labels[ix]),jp.array(weights[ix]))
                if not np.isfinite(float(loss)):raise FloatingPointError('nonfinite supervised loss')
                state.update(supervised_updates=step,data_presentations=step*256,loss=float(loss),updated_utc=utc())
                if step%100==0:atomic_json(path,state)
                if step not in (500,2000):continue
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((params,optimizer))):
                    raise FloatingPointError('nonfinite supervised learning state')
                folder=args.output/f'update-{step}';folder.mkdir(exist_ok=False)
                snapshot=dict(saved,actor=jax.device_get(params),optimizer=jax.device_get(optimizer),
                    training_kind='supervised_demonstration',supervised_updates=step,
                    ppo_optimizer_compatible=False,source_ppo_sha256=manifest['sha256'])
                b=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL);name=f'bc-update-{step}.pkl'
                atomic_bytes(folder/name,b)
                atomic_json(folder/'latest.json',dict(manifest,file=name,sha256=hashlib.sha256(b).hexdigest(),
                    training_kind='supervised_demonstration',supervised_updates=step,ppo_optimizer_compatible=False))
                atomic_json(folder/'config.json',dict(json.loads((source/'config.json').read_text()),
                    training_kind='supervised_demonstration',supervised_updates=step))
                state.update(phase='physical_evaluation',active_checkpoint=str(folder),updated_utc=utc());atomic_json(path,state)
                checks={}
                for corner in (0,1,None):
                    label='selfplay' if corner is None else 'neutral-'+str(corner)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(folder),
                        '--output',str(folder/(label+'.json'))]
                    if corner is not None:command+=['--neutral-partner',str(corner)]
                    with (folder/(label+'.log')).open('w') as out,(folder/(label+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    report=json.loads((folder/(label+'.json')).read_text());rows=report['trials']
                    if len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in (7,23,31,47) for w in (False,True)} or not all(r['finite'] for r in rows):
                        raise ValueError('incomplete independent learned-controller evaluation')
                    state['diagnostic_evaluation_world_controls']+=report['simulated_world_transitions']
                    checks[label]=dict(pair_survivors=sum(r['survived'] for r in rows),
                        own_surviving_scoring_seeds=sorted({r['seed'] for r in rows if r['survived'] and (sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])>0}),
                        own_surviving_events=sum((sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner]) for r in rows if r['survived']))
                    atomic_json(path,state)
                state['checks'].append(dict(supervised_updates=step,checkpoint_sha256=hashlib.sha256(b).hexdigest(),evaluations=checks))
                state.update(phase='offline_training',updated_utc=utc());atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
