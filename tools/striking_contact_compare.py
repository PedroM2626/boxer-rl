"""Matched, bounded contact-history continuation of the latest supervised actor."""
import argparse
import copy
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
from boxer.striking_contact import demonstration_flags, expand_supervised_state, recover_sampler
from boxer.striking_train import Actor, normalize
from tools.striking_demonstration_init import behavior_loss

SOURCE = Path('checkpoints/striking-validation/demonstration-init-1441792/update-2000')
SOURCE_SHA = 'fc48cecf46647ba70fb372a657e9b8a889d37ff7b4dfbdddeb2453872a42540f'
ORIGINAL = Path('checkpoints/striking-validation/partner-duration-1441792')
ORIGINAL_SHA = '87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8'


def load_checked(folder, expected):
    manifest=json.loads((folder/'latest.json').read_text())
    payload=(folder/manifest['file']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected or manifest['sha256']!=expected:
        raise ValueError('reviewed source checksum mismatch')
    return pickle.loads(payload),manifest


def load_demonstrations():
    dataset=Path('docs/results/striking-pilot/directed-holdout-1441792')
    replay=json.loads((dataset/'replay.json').read_text())
    report=json.loads((dataset/'feasibility.json').read_text())
    if len(replay['trials'])!=15 or replay['source_sha256']!=ORIGINAL_SHA or report['source_sha256']!=ORIGINAL_SHA:
        raise ValueError('verified data source mismatch')
    by_file={t['diagnostic_trajectory']:t for t in report['trials'] if t.get('diagnostic_trajectory')}
    observations=[];labels=[];flags=[];files={}
    for trial in replay['trials']:
        file=dataset/trial['trajectory'];payload=file.read_bytes()
        if (hashlib.sha256(payload).hexdigest()!=trial['sha256'] or not trial['observations_match_exactly'] or
                not trial['events_match_exactly'] or not trial['survived']):
            raise ValueError('demonstration integrity mismatch')
        with np.load(file) as arrays:
            obs=arrays['observation'];action=arrays['action']
        if (obs.shape!=(500,146) or action.shape!=(500,17) or obs.dtype!=np.float32 or action.dtype!=np.float32 or
                not np.isfinite(obs).all() or not np.isfinite(action).all() or np.max(np.abs(action))>1.):
            raise ValueError('invalid complete demonstration')
        observations.append(obs);labels.append(action)
        flags.append(demonstration_flags(by_file[file.name]));files[file.name]=trial['sha256']
    return np.concatenate(observations),np.concatenate(labels),np.concatenate(flags),files


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('comparison exists; inspect before recovery')
        if jax.devices()[0].platform!='cpu':raise ValueError('CPU diagnostic required')
        saved,manifest=load_checked(SOURCE,SOURCE_SHA)
        original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
        if saved['supervised_updates']!=2000 or int(saved['optimizer'][1][0].count)!=2000:
            raise ValueError('sampler prefix and BC Adam count disagree')
        expanded=expand_supervised_state(saved)
        observations,labels,flags,files=load_demonstrations()
        actor=Actor()
        inference=jax.jit(lambda p,x:jp.tanh(actor.apply(p,x)[0]))
        baseline=np.asarray(inference(original['actor'],jp.array(normalize(observations,original['norm']))))
        weights=np.where(np.max(np.abs(labels-baseline),axis=1)>.05,5.,1.).astype(np.float32)
        rng=recover_sampler();sampler_start=copy.deepcopy(rng.bit_generator.state)
        expanded['sampler_state']=copy.deepcopy(sampler_start)
        migrated=pickle.dumps(expanded,protocol=pickle.HIGHEST_PROTOCOL)
        atomic_bytes(args.output/'expanded-source.pkl',migrated)
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
        @jax.jit
        def update(params,optimizer,x,y,w):
            loss,gradient=jax.value_and_grad(lambda p:behavior_loss(actor,p,x,y,w))(params)
            changes,optimizer=tx.update(gradient,optimizer,params)
            return optax.apply_updates(params,changes),optimizer,loss
        state=dict(status='running',phase='offline_training',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,original_weight_source_sha256=ORIGINAL_SHA,
            expanded_source_sha256=hashlib.sha256(migrated).hexdigest(),demonstration_files=files,
            demonstration_rows=len(observations),sampler_recovery=dict(seed=313,prefix_batches=2000,
                batch_size=256,rows=7500,state=sampler_start),
            maximum_new_supervised_updates=4000,supervised_updates_before_job=2000,
            new_supervised_updates=0,new_data_presentations=0,new_ppo_transitions=0,
            diagnostic_ppo_retained=1188096,formal_world_controls_before_job=366084,
            diagnostic_evaluation_world_controls=0,maximum_evaluation_world_controls=24000,
            policy_adoption='none',teacher_used_at_evaluation=False,arms=[])
        atomic_json(path,state)
        try:
            for arm in ('zero_history','contact_history'):
                params=copy.deepcopy(expanded['actor']);optimizer=copy.deepcopy(expanded['optimizer'])
                rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(sampler_start)
                x=normalize(np.c_[observations,flags if arm=='contact_history' else np.zeros_like(flags)],expanded['norm'])
                state.update(phase='offline_training',active_arm=arm);atomic_json(path,state)
                for step in range(1,2001):
                    ix=rng.integers(0,len(x),256)
                    params,optimizer,loss=update(params,optimizer,jp.array(x[ix]),jp.array(labels[ix]),jp.array(weights[ix]))
                    if not np.isfinite(float(loss)):raise FloatingPointError('nonfinite supervised loss')
                    state.update(active_arm_updates=step,new_supervised_updates=step+len(state['arms'])*2000,
                        new_data_presentations=(step+len(state['arms'])*2000)*256,loss=float(loss),updated_utc=utc())
                    if step%100==0:atomic_json(path,state)
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((params,optimizer))):
                    raise FloatingPointError('nonfinite supervised state')
                if arm=='zero_history':
                    for tree in (params,optimizer[1][0].mu,optimizer[1][0].nu):
                        np.testing.assert_array_equal(np.asarray(tree['params']['Dense_0']['kernel'])[146:],0.)
                folder=args.output/arm;folder.mkdir(exist_ok=False)
                snapshot=dict(expanded,actor=jax.device_get(params),optimizer=jax.device_get(optimizer),
                    supervised_updates=4000,source_bc_sha256=SOURCE_SHA,contact_training_arm=arm,
                    sampler_state=copy.deepcopy(rng.bit_generator.state))
                payload=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL)
                atomic_bytes(folder/'bc-update-4000.pkl',payload)
                metadata=dict(manifest,file='bc-update-4000.pkl',sha256=hashlib.sha256(payload).hexdigest(),
                    actor_inputs=148,actions=17,contact_history=True,training_kind='supervised_contact_state',
                    supervised_updates=4000,contact_training_arm=arm,source_bc_sha256=SOURCE_SHA)
                atomic_json(folder/'latest.json',metadata)
                atomic_json(folder/'config.json',dict(json.loads((SOURCE/'config.json').read_text()),
                    actor_inputs=148,actions=17,contact_history=True,training_kind='supervised_contact_state',
                    supervised_updates=4000,contact_training_arm=arm))
                state.update(phase='physical_evaluation',active_checkpoint=str(folder));atomic_json(path,state)
                checks={}
                for corner in (0,1,None):
                    label='selfplay' if corner is None else 'neutral-'+str(corner)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(folder),
                        '--output',str(folder/(label+'.json')),'--progress',str(folder/(label+'-progress.json'))]
                    if corner is not None:command+=['--neutral-partner',str(corner)]
                    with (folder/(label+'.log')).open('w') as out,(folder/(label+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    report=json.loads((folder/(label+'.json')).read_text());rows=report['trials']
                    if (len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in (7,23,31,47) for w in (False,True)} or
                            not all(r['finite'] for r in rows) or report['striking_sha256']!=metadata['sha256']):
                        raise ValueError('incomplete independent physical evaluation')
                    state['diagnostic_evaluation_world_controls']+=report['simulated_world_transitions']
                    checks[label]=dict(pair_survivors=sum(r['survived'] for r in rows),
                        own_surviving_scoring_seeds=sorted({r['seed'] for r in rows if r['survived'] and
                            (sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])>0}),
                        own_surviving_events=sum((sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])
                            for r in rows if r['survived']))
                    atomic_json(path,state)
                state['arms'].append(dict(arm=arm,checkpoint_sha256=metadata['sha256'],evaluations=checks,
                    sampler_final=snapshot['sampler_state']))
                atomic_json(path,state)
            if state['arms'][0]['sampler_final']!=state['arms'][1]['sampler_final']:
                raise ValueError('matched sampler streams diverged')
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
