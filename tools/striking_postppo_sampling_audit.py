"""Bounded unchanged saved-Gaussian audit after the reviewed PPO regression."""
import argparse
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
        path=args.output/'sampling.json'
        if path.exists():raise ValueError('Audit already exists; inspect before recovery')
        review=Path('docs/results/striking-pilot/ppo-anchor-bc5000')
        analysis=json.loads((review/'analysis.json').read_text())
        comparison=json.loads((review/'completed-comparison.json').read_text())
        assert analysis['diagnostic_ppo_total']==1450240 and analysis['formal_world_controls_total']==512121
        assert all(not c['passes_complete'] for a in analysis['arms'] for c in a['checks'].values())
        models=Path('checkpoints/striking-validation/ppo-anchor-bc5000')
        variants=[('source',models/'migration',comparison['migrated_sha256'])]
        variants += [(a['arm'],models/a['arm'],a['sha256']) for a in analysis['arms']]
        state=dict(status='running',phase='sampled_evaluation',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),maximum_world_controls=36000,diagnostic_world_controls=0,
            formal_world_controls_before_job=512121,diagnostic_ppo_retained=1450240,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            new_optimizer_updates=0,assistance=False,promotion_eligible=False,policy_adoption='none',
            seeds=comparison['fresh_seeds'],sampled_policy=True,arm_noise_scale=1.,arm_noise_hold=1,checks=[])
        atomic_json(path,state)
        try:
            for name,checkpoint,sha in variants:
                load_checked(checkpoint,sha)
                folder=args.output/name;folder.mkdir(exist_ok=False)
                for role in (0,1,None):
                    profile='selfplay' if role is None else f'neutral-{role}'
                    state.update(active_variant=name,active_evaluation=str(folder/profile),updated_utc=utc());atomic_json(path,state)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                        '--sampled','--seeds',*[str(s) for s in state['seeds']],
                        '--output',str(folder/(profile+'.json')),'--progress',str(folder/(profile+'-progress.json'))]
                    if role is not None:command+=['--neutral-partner',str(role)]
                    with (folder/(profile+'.log')).open('w') as out,(folder/(profile+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    r=json.loads((folder/(profile+'.json')).read_text());rows=r['trials']
                    assert len(rows)==8 and all(t['finite'] for t in rows) and r['striking_sha256']==sha
                    assert {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in state['seeds'] for w in (False,True)}
                    state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
                    state['checks'].append(dict(variant=name,profile=profile,sha256=sha,survivors=sum(t['survived'] for t in rows),
                        surviving_own_events=sum((sum(t['scored_by_corner']) if role is None else t['scored_by_corner'][role]) for t in rows if t['survived']),
                        surviving_two_way_episodes=sum(t['survived'] and all(x>0 for x in t['scored_by_corner']) for t in rows)))
                    assert state['diagnostic_world_controls']<=36000
                    atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
