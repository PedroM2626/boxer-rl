"""One bounded ordinary warm continuation of the verified striking treatment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_retention import validate_retention_state
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE=Path('checkpoints/striking-validation/retention-1572864/ppo_retention')
SHA='f10d817527e521b0f95253c1d8cfdf2db4b3a03195958e15a7638222743fd38c'
SEEDS=list(range(8000003,8000007))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recover-pretraining-lock',action='store_true',
        help='verified zero-training recovery of the original supervisor/child lock collision only')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # The native child owns output/.lock; the supervisor must use its own mutex.
    with (args.output/'.continuation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'continuation.json'
        existing=json.loads(path.read_text()) if path.exists() else None
        if existing and not args.recover_pretraining_lock:raise ValueError('Continuation exists; inspect before recovery')
        if args.recover_pretraining_lock and (not existing or existing['status']!='failed' or
                existing['diagnostic_consumed_steps']!=0 or existing['diagnostic_collection_world_controls']!=0):
            raise ValueError('pretraining-lock recovery requires verified failed zero-training state')
        review=json.loads(Path('docs/results/striking-pilot/retention-trace-1605632/analysis.json').read_text())
        assert review['formal_world_controls_total']==611668 and review['all_18_original_trials_exact']
        saved,manifest=load_checked(SOURCE,SHA);validate_retention_state(saved,saved['retention_provenance'])
        assert saved['consumed_steps']==saved['optimized_steps']==1605632 and saved['world_control_steps']==999424
        checkpoint=args.output/'checkpoint'
        if existing:
            original_progress=json.loads((args.output/'progress.json').read_text())
            assert original_progress==dict(consumed_steps=1605632,optimized_steps=1605632,world_control_steps=999424)
            load_checked(checkpoint,SHA)
            error_log=args.output/'train-16384-errors.log'
            assert 'BlockingIOError: [Errno 11]' in error_log.read_text()
            assert not (args.output/'state.json').exists()
            assert json.loads((args.output/'config.json').read_text())==json.loads((SOURCE/'config.json').read_text())
            for profile in ('neutral-0','neutral-1','selfplay'):
                progress=json.loads((args.output/'source-fresh'/(profile+'-progress.json')).read_text())
                assert progress['status']=='completed' and progress['striking_sha256']==SHA
            shutil.copy2(error_log,args.output/'pretraining-lock-failure.txt')
        else:
            checkpoint.mkdir(exist_ok=False)
            shutil.copy2(SOURCE/manifest['file'],checkpoint/manifest['file']);shutil.copy2(SOURCE/'latest.json',checkpoint/'latest.json')
            shutil.copy2(SOURCE/'config.json',args.output/'config.json')
            atomic_json(args.output/'progress.json',dict(consumed_steps=1605632,optimized_steps=1605632,world_control_steps=999424))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),phase='source_fresh_evaluation',
            diagnostic_output=str(args.output),source_sha256=SHA,source_consumed_offset=1605632,
            source_world_offset=999424,maximum_new_ppo_steps=32768,maximum_evaluation_world_controls=48000,
            diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,diagnostic_evaluation_world_controls=0,
            formal_evaluation_world_controls_before_job=611668,diagnostic_ppo_before_job=1515776,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000,new_separate_supervised_updates=0,
            assistance=False,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,gates=[])
        if existing:
            state=existing
            state.update(status='running',phase='verified_pretraining_lock_recovery',updated_utc=utc(),
                pretraining_lock_recovery=dict(previous_error=existing['error'],new_physics_steps=0,
                    new_optimizer_updates=0,source_evaluation_retained=True,checkpoint_sha256=SHA))
            state.pop('error',None)
        atomic_json(path,state)
        def account():
            progress=json.loads((args.output/'progress.json').read_text())
            state['diagnostic_consumed_steps']=progress['consumed_steps']-1605632
            state['diagnostic_collection_world_controls']=progress['world_control_steps']-999424
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            if not existing:state['source_fresh']=evaluate(SOURCE,args.output/'source-fresh',SEEDS,state,path)
            for target in (16384,32768):
                state.update(phase='training',active_training_output=str(args.output),updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(args.output),'--resume',
                    '--contact-ppo','--demonstration-ppo','--retention-loss-weight','1','--ceiling',str(1605632+target),
                    '--worlds','4','--unroll','128','--seed','53']
                with (args.output/f'train-{target}.log').open('w') as out,(args.output/f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                account();assert state['diagnostic_consumed_steps']==target and state['diagnostic_collection_world_controls']==target//2
                m=json.loads((checkpoint/'latest.json').read_text());current,_=load_checked(checkpoint,m['sha256'])
                validate_retention_state(current,saved['retention_provenance'])
                assert current['optimized_steps']==current['consumed_steps']==1605632+target
                assert int(current['optimizer']['actor_adam'][0].count)==7560+target//1024*16
                assert int(current['optimizer']['critic_adam'][0].count)==31232+target//1024*16
                # Preserve each distinct complete physical checkpoint for later streak review.
                gate_model=args.output/f'gate-{target}'/'checkpoint';gate_model.mkdir(parents=True,exist_ok=False)
                for filename in (m['file'],'latest.json'):shutil.copy2(checkpoint/filename,gate_model/filename)
                shutil.copy2(args.output/'config.json',gate_model/'config.json')
                state.update(phase='screen_evaluation',updated_utc=utc());atomic_json(path,state)
                screen=evaluate(gate_model,args.output/f'gate-{target}'/'screen',[7,23,31,47],state,path)
                complete=(screen['neutral-0']['pair_survivors']>=7 and screen['neutral-1']['pair_survivors']==8 and
                    screen['selfplay']['pair_survivors']>=7 and
                    all(len(screen[f'neutral-{role}']['own_surviving_scoring_seeds'])>=2 for role in (0,1)))
                state['gates'].append(dict(new_ppo_steps=target,consumed_steps=1605632+target,sha256=m['sha256'],
                    screen=screen,passes_complete=complete))
                account();atomic_json(path,state)
                if min(p['pair_survivors'] for p in screen.values())<=3:
                    state['early_stop_reason']='Severe survival regression: at most three surviving pairs in a full screen profile'
                    break
            state['final_fresh']=evaluate(checkpoint,args.output/'final-fresh',SEEDS,state,path)
            account();assert state['diagnostic_consumed_steps']<=32768 and state['diagnostic_evaluation_world_controls']<=48000
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
