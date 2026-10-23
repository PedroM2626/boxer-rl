"""Bounded curriculum task transition to neutral-partner practice with retention."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_neutral_retention import prepare_neutral_state,neutral_config
from boxer.striking_retention import validate_retention_state
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE=Path('checkpoints/striking-validation/retention-1605632/gate-32768')
SHA='76b83b4c1bd0a66b439c24296366b557fda02a0fecf7d801b67e670472587c45'
SEEDS=list(range(9000003,9000007))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'curriculum.json'
        if path.exists():raise ValueError('Neutral retention job exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/retention-1605632/analysis.json').read_text())
        assert review['formal_world_controls_total']==656030 and not any(c['passes_complete'] for c in review['checks'].values())
        saved,manifest=load_checked(SOURCE,SHA);prepared=prepare_neutral_state(saved,SHA)
        payload=pickle.dumps(prepared,protocol=pickle.HIGHEST_PROTOCOL);sha=hashlib.sha256(payload).hexdigest()
        metadata=dict(manifest,file='neutral-transition.pkl',sha256=sha,neutral_contact_ppo=True,
            learners_per_world=1,opponent_mode='neutral_motor')
        checkpoint=args.output/'checkpoint';checkpoint.mkdir(exist_ok=False)
        atomic_bytes(checkpoint/metadata['file'],payload);atomic_json(checkpoint/'latest.json',metadata)
        atomic_json(args.output/'config.json',neutral_config(json.loads((SOURCE/'config.json').read_text())))
        atomic_json(args.output/'progress.json',dict(consumed_steps=1638400,optimized_steps=1638400,
            world_control_steps=1015808,neutral_credits_by_corner=[0,0]))
        transition=args.output/'transition';shutil.copytree(checkpoint,transition)
        shutil.copy2(args.output/'config.json',transition/'config.json')
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),phase='source_fresh_evaluation',
            diagnostic_output=str(args.output),source_sha256=SHA,transition_sha256=sha,
            source_consumed_offset=1638400,source_world_offset=1015808,maximum_new_ppo_steps=32768,
            maximum_evaluation_world_controls=48000,diagnostic_consumed_steps=0,
            diagnostic_collection_world_controls=0,diagnostic_evaluation_world_controls=0,
            formal_evaluation_world_controls_before_job=656030,diagnostic_ppo_before_job=1548544,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000,new_separate_supervised_updates=0,
            assistance=False,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,gates=[])
        atomic_json(path,state)
        def account():
            p=json.loads((args.output/'progress.json').read_text())
            state['diagnostic_consumed_steps']=p['consumed_steps']-1638400
            state['diagnostic_collection_world_controls']=p['world_control_steps']-1015808
            state['neutral_credits_by_corner']=p.get('neutral_credits_by_corner',[0,0])
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            state['source_fresh']=evaluate(transition,args.output/'source-fresh',SEEDS,state,path)
            for target in (1024,16384,32768):
                state.update(phase='training',active_training_output=str(args.output),updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(args.output),'--resume',
                    '--contact-ppo','--demonstration-ppo','--neutral-contact-ppo','--retention-loss-weight','1',
                    '--ceiling',str(1638400+target),'--worlds','8','--unroll','128','--seed','53']
                with (args.output/f'train-{target}.log').open('w') as out,(args.output/f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                account();assert state['diagnostic_consumed_steps']==state['diagnostic_collection_world_controls']==target
                assert sum(state['neutral_credits_by_corner'])==target and min(state['neutral_credits_by_corner'])>0
                m=json.loads((checkpoint/'latest.json').read_text());current,_=load_checked(checkpoint,m['sha256'])
                validate_retention_state(current,saved['retention_provenance'])
                assert current['consumed_steps']==current['optimized_steps']==1638400+target
                assert int(current['optimizer']['actor_adam'][0].count)==8072+target//1024*16
                assert int(current['optimizer']['critic_adam'][0].count)==31744+target//1024*16
                if target==1024:
                    atomic_json(args.output/'initial-block-verified.json',dict(current_consumed_steps=current['consumed_steps'],
                        new_learner_credits=1024,new_world_controls=1024,neutral_credits_by_corner=state['neutral_credits_by_corner'],
                        actor_count=8088,critic_count=31760,checkpoint_sha256=m['sha256'],included_in_allocation=True))
                    continue
                gate_model=args.output/f'gate-{target}'/'checkpoint';shutil.copytree(checkpoint,gate_model)
                shutil.copy2(args.output/'config.json',gate_model/'config.json')
                state.update(phase='screen_evaluation',updated_utc=utc());atomic_json(path,state)
                screen=evaluate(gate_model,args.output/f'gate-{target}'/'screen',[7,23,31,47],state,path)
                complete=(screen['neutral-0']['pair_survivors']>=7 and screen['neutral-1']['pair_survivors']==8 and
                    screen['selfplay']['pair_survivors']>=7 and all(len(screen[f'neutral-{r}']['own_surviving_scoring_seeds'])>=2 for r in (0,1)))
                state['gates'].append(dict(new_ppo_steps=target,consumed_steps=1638400+target,sha256=m['sha256'],
                    screen=screen,passes_complete=complete));atomic_json(path,state)
                if min(p['pair_survivors'] for p in screen.values())<=3:
                    state['early_stop_reason']='Severe survival regression in a complete screen profile';break
            state['final_fresh']=evaluate(checkpoint,args.output/'final-fresh',SEEDS,state,path)
            account();assert state['diagnostic_evaluation_world_controls']<=48000
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
