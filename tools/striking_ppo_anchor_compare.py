"""Guarded matched active PPO continuation from the latest verified BC actor."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_ppo_warmstart import migrate_supervised_state,contact_config,validate_split_checkpoint
from tools.striking_contact_compare import load_checked,ORIGINAL,ORIGINAL_SHA

SOURCE=Path('checkpoints/striking-validation/motor-anchoring-bc4000/zero_anchor')
SOURCE_SHA='c7d6f1d5458f4767e6d7bedd98350ed8f79231de1030783fe9feb4b779d81e19'
FRESH_SEEDS=list(range(5000003,5000007))
STEPS=131072


def evaluate(checkpoint,destination,seeds,state,path):
    destination.mkdir(parents=True,exist_ok=False);summaries={}
    manifest=json.loads((checkpoint/'latest.json').read_text())
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
        state['diagnostic_evaluation_world_controls']+=report['simulated_world_transitions'];atomic_json(path,state)
        if (len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in seeds for w in (False,True)} or
                report['striking_sha256']!=manifest['sha256'] or not all(r['finite'] for r in rows)):
            raise ValueError('incomplete full physical evaluation')
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
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('PPO comparison exists; inspect before any recovery')
        saved,manifest=load_checked(SOURCE,SOURCE_SHA);original,_=load_checked(ORIGINAL,ORIGINAL_SHA)
        prerequisite=json.loads(Path('docs/results/striking-pilot/motor-anchoring-bc4000/analysis.json').read_text())
        if prerequisite['formal_world_controls_total']!=458469:raise ValueError('completed reviewed BC comparison required')
        migrated=migrate_supervised_state(saved,original,SOURCE_SHA,ORIGINAL_SHA)
        migrated['anchor_loss_weight']=0.
        payload=pickle.dumps(migrated,protocol=pickle.HIGHEST_PROTOCOL)
        migration_sha=hashlib.sha256(payload).hexdigest();migration=args.output/'migration';migration.mkdir(exist_ok=False)
        metadata=dict(manifest,file='migrated-bc5000.pkl',sha256=migration_sha,training_kind='ppo',
            ppo_optimizer_compatible=True,migration_version=1,optimizer_kind='split_adam_v1',
            learners_per_world=2,opponent_mode='active_selfplay',anchor_loss_weight=0.)
        source_config=json.loads((SOURCE/'config.json').read_text())
        atomic_bytes(migration/metadata['file'],payload);atomic_json(migration/'latest.json',metadata)
        atomic_json(migration/'config.json',contact_config(source_config,0.))
        atomic_json(migration/'integrity.json',dict(migrated['optimizer_migration'],
            migrated_sha256=migration_sha,all_original_parameters_and_counters_retained=True,
            actor_count=5000,critic_count=28672,profile_transition='neutral one-learner to active two-learner',
            no_physical_or_optimizer_steps_at_conversion=True))
        offset=saved['consumed_steps'];world_offset=saved['world_control_steps']
        state=dict(status='running',phase='source_fresh_evaluation',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,critic_optimizer_source_sha256=ORIGINAL_SHA,
            migrated_sha256=migration_sha,source_consumed_offset=offset,source_world_offset=world_offset,
            new_ppo_steps_per_arm=STEPS,maximum_new_ppo_steps=2*STEPS,diagnostic_consumed_steps=0,
            diagnostic_collection_world_controls=0,diagnostic_evaluation_world_controls=0,
            maximum_evaluation_world_controls=60000,formal_evaluation_world_controls_before_job=458469,
            diagnostic_ppo_before_job=1188096,cumulative_supervised_updates_retained=8000,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            policy_adoption='none',promotion=False,assistance=False,fresh_seeds=FRESH_SEEDS,arms=[])
        atomic_json(path,state)
        def account():
            progress=[json.loads(p.read_text()) for p in (args.output/'ppo_only'/'progress.json',args.output/'ppo_anchor'/'progress.json') if p.exists()]
            state['diagnostic_consumed_steps']=sum(max(0,p['consumed_steps']-offset) for p in progress)
            state['diagnostic_collection_world_controls']=sum(max(0,p['world_control_steps']-world_offset) for p in progress)
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions']
                for p in args.output.rglob('*-progress.json'))
        try:
            state['source_fresh']=evaluate(migration,args.output/'source-fresh',FRESH_SEEDS,state,path);atomic_json(path,state)
            for arm,weight in (('ppo_only',0.),('ppo_anchor',.02)):
                folder=args.output/arm;checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True,exist_ok=False)
                snapshot=copy.deepcopy(migrated);snapshot['anchor_loss_weight']=weight
                b=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL);sha=hashlib.sha256(b).hexdigest()
                atomic_bytes(checkpoint/'initial.pkl',b)
                atomic_json(checkpoint/'latest.json',dict(metadata,file='initial.pkl',sha256=sha,anchor_loss_weight=weight))
                atomic_json(folder/'config.json',contact_config(source_config,weight))
                atomic_json(folder/'progress.json',dict(consumed_steps=offset,optimized_steps=saved['optimized_steps'],world_control_steps=world_offset))
                state.update(active_arm=arm,active_training_output=str(folder),phase='training',updated_utc=utc());atomic_json(path,state)
                # Both arms share the same initial-block checkpoint boundary.
                # This integration guard consumes and retains part of each arm's allocation.
                for target,label in ((1024,'initial-block'),(STEPS,'remaining')):
                    command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(folder),'--resume','--contact-ppo',
                        '--anchor-loss-weight',str(weight),'--ceiling',str(offset+target),'--worlds','4','--unroll','128','--seed','53']
                    with (folder/(label+'.log')).open('w') as out,(folder/(label+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    progress=json.loads((folder/'progress.json').read_text());account();atomic_json(path,state)
                    if (progress['consumed_steps']-offset!=target or progress['optimized_steps']!=offset+target or
                            progress['world_control_steps']-world_offset!=target//2):
                        raise ValueError('active two-learner/physical accounting mismatch')
                    final_manifest=json.loads((checkpoint/'latest.json').read_text())
                    current,_=load_checked(checkpoint,final_manifest['sha256']);validate_split_checkpoint(current)
                    updates=target//1024*16
                    if (int(current['optimizer']['actor_adam'][0].count)!=5000+updates or
                            int(current['optimizer']['critic_adam'][0].count)!=28672+updates):
                        raise ValueError('independent optimizer counts disagree with real update count')
                    if target==1024:
                        atomic_json(folder/'initial-block-verified.json',dict(progress,actor_count=5016,critic_count=28688,
                            checkpoint_sha256=final_manifest['sha256'],included_in_arm_allocation=True))
                state.update(phase='evaluation',updated_utc=utc());atomic_json(path,state)
                screen=evaluate(checkpoint,folder/'screen',[7,23,31,47],state,path)
                fresh=evaluate(checkpoint,folder/'fresh',FRESH_SEEDS,state,path)
                state['arms'].append(dict(arm=arm,anchor_loss_weight=weight,initial_sha256=sha,
                    final_checkpoint_sha256=final_manifest['sha256'],new_ppo_steps=STEPS,screen=screen,fresh=fresh))
                atomic_json(path,state)
            account();state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
