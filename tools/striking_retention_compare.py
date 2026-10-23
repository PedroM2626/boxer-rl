"""Bounded matched PPO demonstration-retention continuation, without rollback."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_retention import (load_retention_data,prepare_retention_state,
    retention_config,validate_retention_state)
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE=Path('checkpoints/striking-validation/ppo-anchor-bc5000/ppo_only')
SHA='e3e01b71de64e3d26a431ba87d484a45d349621df5461c274fb7e394b935ef4e'
STEPS=32768
SEEDS=list(range(7000003,7000007))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('Retention comparison exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/gradient-audit-bc5000/analysis.json').read_text())
        assert review['formal_world_controls_total']==547894 and review['new_optimizer_updates']==0
        saved,manifest=load_checked(SOURCE,SHA);_,provenance=load_retention_data()
        canonical=prepare_retention_state(saved,SHA,0.,provenance)
        config=json.loads((SOURCE/'config.json').read_text())
        transition=args.output/'transition';transition.mkdir(exist_ok=False)
        payload=pickle.dumps(canonical,protocol=pickle.HIGHEST_PROTOCOL);sha=hashlib.sha256(payload).hexdigest()
        metadata=dict(manifest,file='transition.pkl',sha256=sha,demonstration_ppo=True,retention_loss_weight=0.)
        atomic_bytes(transition/'transition.pkl',payload);atomic_json(transition/'latest.json',metadata)
        atomic_json(transition/'config.json',retention_config(config,0.))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),phase='source_fresh_evaluation',
            diagnostic_output=str(args.output),source_sha256=SHA,transition_sha256=sha,
            source_consumed_offset=1572864,source_world_offset=983040,
            new_ppo_steps_per_arm=STEPS,maximum_new_ppo_steps=65536,
            diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0,maximum_evaluation_world_controls=60000,
            formal_evaluation_world_controls_before_job=547894,diagnostic_ppo_before_job=1450240,
            cumulative_supervised_updates_retained=8000,new_separate_supervised_updates=0,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            assistance=False,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,
            retention_provenance=provenance,arms=[])
        atomic_json(path,state)
        def account():
            rows=[json.loads(p.read_text()) for p in (args.output/'ppo_only/progress.json',args.output/'ppo_retention/progress.json') if p.exists()]
            state['diagnostic_consumed_steps']=sum(max(0,r['consumed_steps']-1572864) for r in rows)
            state['diagnostic_collection_world_controls']=sum(max(0,r['world_control_steps']-983040) for r in rows)
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            state['source_fresh']=evaluate(transition,args.output/'source-fresh',SEEDS,state,path)
            for arm,weight in (('ppo_only',0.),('ppo_retention',1.)):
                folder=args.output/arm;checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True,exist_ok=False)
                snapshot=copy.deepcopy(canonical);snapshot['retention_loss_weight']=weight
                data=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL);initial_sha=hashlib.sha256(data).hexdigest()
                atomic_bytes(checkpoint/'initial.pkl',data)
                atomic_json(checkpoint/'latest.json',dict(metadata,file='initial.pkl',sha256=initial_sha,retention_loss_weight=weight))
                atomic_json(folder/'config.json',retention_config(config,weight))
                atomic_json(folder/'progress.json',dict(consumed_steps=1572864,optimized_steps=1572864,world_control_steps=983040))
                state.update(active_arm=arm,active_training_output=str(folder),phase='training',updated_utc=utc());atomic_json(path,state)
                for target,label in ((1024,'initial-block'),(STEPS,'remaining')):
                    command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(folder),'--resume','--contact-ppo',
                        '--demonstration-ppo','--retention-loss-weight',str(weight),'--ceiling',str(1572864+target),
                        '--worlds','4','--unroll','128','--seed','53']
                    with (folder/(label+'.log')).open('w') as out,(folder/(label+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    progress=json.loads((folder/'progress.json').read_text());account();atomic_json(path,state)
                    assert progress['consumed_steps']==progress['optimized_steps']==1572864+target
                    assert progress['world_control_steps']==983040+target//2
                    final_manifest=json.loads((checkpoint/'latest.json').read_text())
                    current,_=load_checked(checkpoint,final_manifest['sha256']);validate_retention_state(current,provenance)
                    assert int(current['optimizer']['actor_adam'][0].count)==7048+target//1024*16
                    assert int(current['optimizer']['critic_adam'][0].count)==30720+target//1024*16
                    if target==1024:
                        atomic_json(folder/'initial-block-verified.json',dict(progress,checkpoint_sha256=final_manifest['sha256'],included_in_arm_allocation=True))
                state.update(phase='evaluation',updated_utc=utc());atomic_json(path,state)
                screen=evaluate(checkpoint,folder/'screen',[7,23,31,47],state,path)
                fresh=evaluate(checkpoint,folder/'fresh',SEEDS,state,path)
                state['arms'].append(dict(arm=arm,retention_loss_weight=weight,initial_sha256=initial_sha,
                    final_checkpoint_sha256=final_manifest['sha256'],new_ppo_steps=STEPS,screen=screen,fresh=fresh))
                atomic_json(path,state)
            finals=[load_checked(args.output/a['arm']/'checkpoint',a['final_checkpoint_sha256'])[0] for a in state['arms']]
            assert finals[0]['retention_sampler_state']==finals[1]['retention_sampler_state']
            assert finals[0]['aux_sampler_state']==finals[1]['aux_sampler_state']
            account();state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
