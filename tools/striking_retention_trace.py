"""Bounded exact replay and passive stability traces of the retention comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_contact_compare import load_checked


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'trace-review.json'
        if path.exists():raise ValueError('Replay/trace audit exists; inspect before recovery')
        reference=Path('docs/results/striking-pilot/retention-1572864')
        review=json.loads((reference/'analysis.json').read_text())
        assert review['formal_world_controls_total']==603577 and review['diagnostic_ppo_total']==1515776
        variants={a['arm']:a['sha256'] for a in review['arms']}
        models=Path('checkpoints/striking-validation/retention-1572864')
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            maximum_world_controls=9000,diagnostic_world_controls=0,formal_world_controls_before_job=603577,
            diagnostic_ppo_retained=1515776,new_optimizer_updates=0,assistance=False,
            policy_adoption='none',promotion_eligible=False,checks=[])
        atomic_json(path,state)
        jobs=[('exchange','ppo_retention',None,[7000003]),
              ('control-neutral-1','ppo_only',1,list(range(7000003,7000007))),
              ('retention-neutral-1','ppo_retention',1,list(range(7000003,7000007)))]
        try:
            for name,variant,role,seeds in jobs:
                checkpoint=models/variant;load_checked(checkpoint,variants[variant])
                folder=args.output/name;folder.mkdir(exist_ok=False)
                state.update(active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                    '--seeds',*[str(s) for s in seeds],'--output',str(folder/'evaluation.json'),
                    '--progress',str(folder/'evaluation-progress.json'),'--trace-directory',str(folder/'traces')]
                if role is not None:command+=['--neutral-partner',str(role)]
                with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                report=json.loads((folder/'evaluation.json').read_text())
                original=json.loads((reference/variant/'fresh'/('selfplay.json' if role is None else 'neutral-1.json')).read_text())
                rows={(t['seed'],t['swap']):t for t in original['trials']}
                assert len(report['trials'])==len(seeds)*2 and report['striking_sha256']==variants[variant]
                for trial in report['trials']:
                    old=rows[trial['seed'],trial['swap']]
                    assert all(trial.get(k)==v for k,v in old.items()),'Passive trace changed physical evaluation'
                    assert trial['finite']
                state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
                assert state['diagnostic_world_controls']<=9000
                state['checks'].append(dict(name=name,variant=variant,sha256=variants[variant],
                    all_original_physical_fields_exact=True,trials=report['trials']))
                atomic_json(path,state)
                if name=='exchange':
                    import numpy as np
                    from boxer.curriculum_env import UnassistedG1
                    from boxer.paired_foundation import PairedFoundation
                    from tools.evaluate_paired_foundation import render
                    trial=next(t for t in report['trials'] if t['swap'])
                    assert trial['survived'] and trial['scored_by_corner']==[3.,2.]
                    with np.load(folder/'traces/trained-seed-7000003-swap-1.npz') as trace:
                        assert trace['qpos'].shape[0]==500 and not trace['fallen'].any()
                        video=Path('fights/striking-pilot/learned-mean-two-way-retention-1605632.mp4')
                        render(PairedFoundation(UnassistedG1('approach',impl='jax')),trace['qpos'][::2],video,
                            caption='Learned mean policy | Selected 10 s replay | Free bodies',
                            frame_times=trace['time_after'][::2],impact_events=trial['impact_events'])
                    state.update(video=str(video),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                        replay_seed=7000003,replay_swap=True,replay_events=trial['impact_events'])
                    atomic_json(path,state)
            state.update(status='completed',updated_utc=utc(),phase='awaiting_analysis');atomic_json(path,state)
        except BaseException as error:
            state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
