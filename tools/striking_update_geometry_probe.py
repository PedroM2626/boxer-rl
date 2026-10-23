"""One 1024-credit diagnostic update from the latest regressed final, with passive KL measurements."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/mixed-partner-continue-1769472/gate-65536')
SHA='8c99c2dfe21cd6c94a90ed31c487620118df5b0d0fdfe430eabe2471745a31c2'


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'probe.json'
        if path.exists():raise ValueError('Update geometry probe exists; inspect before recovery')
        prior=json.loads(Path('docs/results/striking-pilot/mixed-partner-continue-1769472/analysis.json').read_text())
        assert prior['status']=='reviewed' and prior['diagnostic_ppo_total']==1810688 and prior['formal_world_controls_total']==989355
        drift=json.loads(Path('docs/results/striking-pilot/mixed-partner-continue-1769472/policy-drift.json').read_text())
        assert drift['status']=='completed' and drift['new_world_controls']==drift['new_optimizer_updates']==0
        source,manifest=load_checked(SOURCE,SHA);require_ppo_checkpoint(source)
        assert source['consumed_steps']==source['optimized_steps']==1835008 and source['world_control_steps']==1212416
        cp=args.output/'checkpoint';cp.mkdir(exist_ok=False)
        for filename in (manifest['file'],'latest.json'):shutil.copy2(SOURCE/filename,cp/filename)
        shutil.copy2(SOURCE/'config.json',args.output/'config.json')
        atomic_json(args.output/'progress.json',dict(consumed_steps=1835008,optimized_steps=1835008,
            world_control_steps=1212416,neutral_credits_by_corner=source['neutral_credits_by_corner'],
            mixed_credits_by_mode_and_corner=source['mixed_credits_by_mode_and_corner']))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),phase='one_block_collection_and_update',
            diagnostic_output=str(args.output),source_sha256=SHA,maximum_new_ppo_steps=1024,
            maximum_collection_world_controls=1024,maximum_evaluation_world_controls=0,
            diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,
            diagnostic_ppo_before_job=1810688,formal_evaluation_world_controls_retained=989355,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            standalone_supervised_updates_retained=8000,maximum_new_integrated_replay_presentations=4096,
            passive_instrumentation=True,optimizer_or_objective_changed=False,
            assistance=False,policy_adoption='none',promotion=False)
        atomic_json(path,state)
        def account():
            p=json.loads((args.output/'progress.json').read_text())
            state['diagnostic_consumed_steps']=p['consumed_steps']-1835008
            state['diagnostic_collection_world_controls']=p['world_control_steps']-1212416
        try:
            command=[sys.executable,'-u','-m','boxer.striking_train','--output',str(args.output),'--resume',
                '--contact-ppo','--demonstration-ppo','--neutral-contact-ppo','--credit-horizon-ppo',
                '--mixed-partner-ppo','--training-gamma','.999','--retention-loss-weight','1',
                '--ceiling','1836032','--worlds','8','--unroll','128','--seed','53',
                '--update-audit-directory',str(args.output/'update-audit')]
            with (args.output/'train.log').open('w') as out,(args.output/'train-errors.log').open('w') as err:
                subprocess.run(command,stdout=out,stderr=err,check=True)
            assert not (args.output/'train-errors.log').read_text().strip()
            account();assert state['diagnostic_consumed_steps']==state['diagnostic_collection_world_controls']==1024
            m=json.loads((cp/'latest.json').read_text());final,_=load_checked(cp,m['sha256']);require_ppo_checkpoint(final)
            assert final['consumed_steps']==final['optimized_steps']==1836032 and final['world_control_steps']==1213440
            assert int(final['optimizer']['actor_adam'][0].count)==11160 and int(final['optimizer']['critic_adam'][0].count)==34832
            measurements=json.loads((args.output/'update-audit/minibatch-measurements.json').read_text())
            assert measurements['rollout_rows']==1024 and len(measurements['records'])==18
            assert sum(r['phase']=='optimizer' for r in measurements['records'])==16
            assert measurements['records'][-1]['phase']=='after_normalizer'
            assert measurements['records'][-1]['profiles']['neutral']['rows']==512
            assert measurements['records'][-1]['profiles']['active']['rows']==512
            state.update(status='completed',phase='awaiting_review',updated_utc=utc(),
                diagnostic_ppo_total=1811712,final_checkpoint_sha256=m['sha256'],
                final_measurement=measurements['records'][-1],native_error_logs_empty=True)
            atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
