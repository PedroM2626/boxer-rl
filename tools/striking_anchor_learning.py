"""Matched bounded BC anchoring, preserving the latest actor/Adam/sampler lineage."""
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

from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_anchor import load_anchor_candidates,mixed_indices,require_anchor_continuation
from boxer.striking_train import Actor,normalize
from tools.striking_contact_compare import load_checked,load_demonstrations,ORIGINAL,ORIGINAL_SHA
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA
from tools.striking_demonstration_init import behavior_loss

FRESH_SEEDS=list(range(4000003,4000007))


def evaluate(checkpoint,destination,seeds,state,path):
    destination.mkdir(parents=True,exist_ok=False)
    summaries={}
    for corner in (0,1,None):
        name='selfplay' if corner is None else 'neutral-'+str(corner)
        state.update(active_evaluation=str(destination/name),updated_utc=utc());atomic_json(path,state)
        command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
            '--seeds',*[str(s) for s in seeds],'--output',str(destination/(name+'.json')),
            '--progress',str(destination/(name+'-progress.json'))]
        if corner is not None:command+=['--neutral-partner',str(corner)]
        with (destination/(name+'.log')).open('w') as out,(destination/(name+'-errors.log')).open('w') as err:
            subprocess.run(command,stdout=out,stderr=err,check=True)
        report=json.loads((destination/(name+'.json')).read_text());rows=report['trials']
        state['diagnostic_world_controls']+=report['simulated_world_transitions'];atomic_json(path,state)
        manifest=json.loads((checkpoint/'latest.json').read_text())
        if (len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in seeds for w in (False,True)} or
                report['striking_sha256']!=manifest['sha256'] or not all(r['finite'] for r in rows)):
            raise ValueError('incomplete independent anchoring evaluation')
        summaries[name]=dict(pair_survivors=sum(r['survived'] for r in rows),
            own_surviving_events=sum((sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])
                for r in rows if r['survived']),
            own_surviving_scoring_seeds=sorted({r['seed'] for r in rows if r['survived'] and
                (sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])>0}))
    return summaries


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'learning.json'
        if path.exists():raise ValueError('anchoring experiment exists; inspect before recovery')
        if jax.devices()[0].platform!='cpu':raise ValueError('CPU comparison required')
        saved,manifest=load_checked(SOURCE,SOURCE_SHA);require_anchor_continuation(saved)
        data,dataset=load_anchor_candidates()
        review=json.loads(Path('docs/results/striking-pilot/precontact-anchors-bc4000/analysis.json').read_text())
        if (review['formal_world_controls_total']!=403150 or not review['matched_initial_observations_exact'] or
                review['native_numerical_or_capacity_warnings']):raise ValueError('reviewed anchor comparison required')
        if dataset['source_sha256']!=SOURCE_SHA or dataset['unique_role_trajectories']!=32:
            raise ValueError('reviewed anchor coverage mismatch')
        original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
        obs,labels,flags,files=load_demonstrations();actor=Actor()
        inference=jax.jit(lambda p,x:jp.tanh(actor.apply(p,x)[0]))
        baseline=np.asarray(inference(original['actor'],jp.array(normalize(obs,original['norm']))))
        weights=np.where(np.max(np.abs(labels-baseline),axis=1)>.05,5.,1.).astype(np.float32)
        demos=normalize(np.c_[obs,flags],saved['norm'])
        anchors=normalize(data['observation'],saved['norm'])
        preserve=np.asarray(inference(saved['actor'],jp.array(anchors)))
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
        @jax.jit
        def update(params,optimizer,x,y,w):
            loss,gradient=jax.value_and_grad(lambda p:behavior_loss(actor,p,x,y,w))(params)
            changes,optimizer=tx.update(gradient,optimizer,params)
            return optax.apply_updates(params,changes),optimizer,loss
        state=dict(status='running',phase='fresh_source_baseline',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,anchor_dataset_sha256=dataset['sha256'],
            demonstration_files=files,maximum_new_supervised_updates=2000,new_supervised_updates=0,
            new_data_presentations=0,cumulative_supervised_updates_before_job=6000,
            source_lineage_supervised_updates=4000,maximum_world_controls=60000,diagnostic_world_controls=0,
            formal_world_controls_before_job=403150,new_ppo_transitions=0,diagnostic_ppo_retained=1188096,
            policy_adoption='none',promotion=False,fresh_seeds=FRESH_SEEDS,
            sampler_start=copy.deepcopy(saved['sampler_state']),arms=[])
        atomic_json(path,state)
        try:
            state['source_fresh']=evaluate(SOURCE,args.output/'source-fresh',FRESH_SEEDS,state,path);atomic_json(path,state)
            for arm in ('preserve_anchor','zero_anchor'):
                params=copy.deepcopy(saved['actor']);optimizer=copy.deepcopy(saved['optimizer'])
                rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved['sampler_state'])
                anchor_labels=preserve if arm=='preserve_anchor' else data['action']
                state.update(phase='offline_training',active_arm=arm);atomic_json(path,state)
                for step in range(1,1001):
                    di,ai=mixed_indices(rng,len(demos),len(anchors))
                    x=np.r_[demos[di],anchors[ai]];y=np.r_[labels[di],anchor_labels[ai]]
                    w=np.r_[weights[di],np.ones(64,np.float32)]
                    params,optimizer,loss=update(params,optimizer,jp.array(x),jp.array(y),jp.array(w))
                    if not np.isfinite(float(loss)):raise FloatingPointError('nonfinite anchored loss')
                    state.update(active_arm_updates=step,new_supervised_updates=step+len(state['arms'])*1000,
                        new_data_presentations=(step+len(state['arms'])*1000)*256,loss=float(loss),updated_utc=utc())
                    if step%100==0:atomic_json(path,state)
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((params,optimizer))):
                    raise FloatingPointError('nonfinite anchored learning state')
                folder=args.output/arm;folder.mkdir(exist_ok=False)
                snapshot=dict(saved,actor=jax.device_get(params),optimizer=jax.device_get(optimizer),
                    training_kind='supervised_motor_anchor',supervised_updates=5000,
                    anchor_training_arm=arm,source_bc_sha256=SOURCE_SHA,anchor_dataset_sha256=dataset['sha256'],
                    sampler_state=copy.deepcopy(rng.bit_generator.state))
                payload=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL)
                atomic_bytes(folder/'bc-update-5000.pkl',payload)
                metadata=dict(manifest,file='bc-update-5000.pkl',sha256=hashlib.sha256(payload).hexdigest(),
                    training_kind='supervised_motor_anchor',supervised_updates=5000,anchor_training_arm=arm,
                    source_bc_sha256=SOURCE_SHA,anchor_dataset_sha256=dataset['sha256'])
                atomic_json(folder/'latest.json',metadata)
                atomic_json(folder/'config.json',dict(json.loads((SOURCE/'config.json').read_text()),
                    training_kind='supervised_motor_anchor',supervised_updates=5000,anchor_training_arm=arm))
                state.update(phase='physical_evaluation',active_checkpoint=str(folder));atomic_json(path,state)
                screen=evaluate(folder,folder/'screen',[7,23,31,47],state,path)
                fresh=evaluate(folder,folder/'fresh',FRESH_SEEDS,state,path)
                state['arms'].append(dict(arm=arm,checkpoint_sha256=metadata['sha256'],screen=screen,fresh=fresh,
                    sampler_final=snapshot['sampler_state']))
                atomic_json(path,state)
            if state['arms'][0]['sampler_final']!=state['arms'][1]['sampler_final']:
                raise ValueError('matched anchoring sampler streams diverged')
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
