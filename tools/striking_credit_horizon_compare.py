"""Matched bounded credit-horizon comparison, with unchanged native rewards."""
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
from boxer.striking_credit_horizon import prepare_horizon_state,horizon_config,validate_horizon_state
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE=Path('checkpoints/striking-validation/neutral-retention-1638400/gate-32768')
SHA='63c06464fa0ce0ec238d349e4fac559c454a323f235fc8630a5bf599753ce8e9'
SEEDS=list(range(10000003,10000007))
STEPS=32768


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('Horizon comparison exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/reward-credit-1671168/analysis.json').read_text())
        assert review['formal_world_controls_total']==712013 and review['all_26_physical_trials_exact']
        saved,manifest=load_checked(SOURCE,SHA);canonical=prepare_horizon_state(saved,SHA,.99)
        config=json.loads((SOURCE/'config.json').read_text());transition=args.output/'transition';transition.mkdir(exist_ok=False)
        payload=pickle.dumps(canonical,protocol=pickle.HIGHEST_PROTOCOL);sha=hashlib.sha256(payload).hexdigest()
        metadata=dict(manifest,file='transition.pkl',sha256=sha,credit_horizon_ppo=True,training_gamma=.99)
        atomic_bytes(transition/'transition.pkl',payload);atomic_json(transition/'latest.json',metadata)
        atomic_json(transition/'config.json',horizon_config(config,.99))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),phase='source_fresh_evaluation',
            diagnostic_output=str(args.output),source_sha256=SHA,transition_sha256=sha,
            source_consumed_offset=1671168,source_world_offset=1048576,maximum_new_ppo_steps=65536,
            new_ppo_steps_per_arm=STEPS,maximum_evaluation_world_controls=60000,
            diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,diagnostic_evaluation_world_controls=0,
            formal_evaluation_world_controls_before_job=712013,diagnostic_ppo_before_job=1581312,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000,new_separate_supervised_updates=0,
            assistance=False,unchanged_rewards=True,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,arms=[])
        atomic_json(path,state)
        def account():
            rows=[json.loads(p.read_text()) for p in (args.output/'gamma_099/progress.json',args.output/'gamma_0999/progress.json') if p.exists()]
            state['diagnostic_consumed_steps']=sum(p['consumed_steps']-1671168 for p in rows)
            state['diagnostic_collection_world_controls']=sum(p['world_control_steps']-1048576 for p in rows)
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            state['source_fresh']=evaluate(transition,args.output/'source-fresh',SEEDS,state,path)
            for arm,gamma in (('gamma_099',.99),('gamma_0999',.999)):
                folder=args.output/arm;checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True,exist_ok=False)
                snapshot=copy.deepcopy(canonical);snapshot['training_gamma']=gamma
                snapshot['credit_horizon_transition']['new_gamma']=gamma
                data=pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL);initial_sha=hashlib.sha256(data).hexdigest()
                atomic_bytes(checkpoint/'initial.pkl',data)
                atomic_json(checkpoint/'latest.json',dict(metadata,file='initial.pkl',sha256=initial_sha,training_gamma=gamma))
                atomic_json(folder/'config.json',horizon_config(config,gamma))
                atomic_json(folder/'progress.json',dict(consumed_steps=1671168,optimized_steps=1671168,
                    world_control_steps=1048576,neutral_credits_by_corner=saved['neutral_credits_by_corner']))
                state.update(active_arm=arm,active_training_output=str(folder),phase='training',updated_utc=utc());atomic_json(path,state)
                for target,label in ((1024,'initial-block'),(STEPS,'remaining')):
                    command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(folder),'--resume',
                        '--contact-ppo','--demonstration-ppo','--neutral-contact-ppo','--credit-horizon-ppo',
                        '--training-gamma',str(gamma),'--retention-loss-weight','1','--ceiling',str(1671168+target),
                        '--worlds','8','--unroll','128','--seed','53']
                    with (folder/(label+'.log')).open('w') as out,(folder/(label+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    p=json.loads((folder/'progress.json').read_text());account();atomic_json(path,state)
                    assert p['consumed_steps']==p['optimized_steps']==1671168+target and p['world_control_steps']==1048576+target
                    m=json.loads((checkpoint/'latest.json').read_text());current,_=load_checked(checkpoint,m['sha256'])
                    validate_horizon_state(current)
                    assert current['training_gamma']==gamma and int(current['optimizer']['actor_adam'][0].count)==8584+target//1024*16
                    assert int(current['optimizer']['critic_adam'][0].count)==32256+target//1024*16
                    assert sum(current['neutral_credits_by_corner'])==32768+target
                    if target==1024:
                        atomic_json(folder/'initial-block-verified.json',dict(p,checkpoint_sha256=m['sha256'],included_in_arm_allocation=True))
                state.update(phase='evaluation',updated_utc=utc());atomic_json(path,state)
                screen=evaluate(checkpoint,folder/'screen',[7,23,31,47],state,path)
                fresh=evaluate(checkpoint,folder/'fresh',SEEDS,state,path)
                state['arms'].append(dict(arm=arm,gamma=gamma,initial_sha256=initial_sha,
                    final_checkpoint_sha256=m['sha256'],new_ppo_steps=STEPS,screen=screen,fresh=fresh));atomic_json(path,state)
            finals=[load_checked(args.output/a['arm']/'checkpoint',a['final_checkpoint_sha256'])[0] for a in state['arms']]
            assert finals[0]['retention_sampler_state']==finals[1]['retention_sampler_state'] and finals[0]['aux_sampler_state']==finals[1]['aux_sampler_state']
            account();assert state['diagnostic_consumed_steps']==65536 and state['diagnostic_evaluation_world_controls']<=60000
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
