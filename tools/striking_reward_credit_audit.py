"""Bounded exact passive reward tracing of the latest actor and selected exchange."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_contact_compare import load_checked

TERMS=('scored','received','fall','progress','facing','engagement','residual')


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'reward-credit.json'
        if path.exists():raise ValueError('Reward audit exists; inspect before recovery')
        prerequisite=json.loads(Path('docs/results/striking-pilot/neutral-retention-1638400/analysis.json').read_text())
        assert prerequisite['formal_world_controls_total']==699791 and prerequisite['diagnostic_ppo_total']==1581312
        latest=Path('checkpoints/striking-validation/neutral-retention-1638400/gate-32768')
        source=Path('checkpoints/striking-validation/retention-1572864/ppo_retention')
        jobs=[('selected-exchange',source,'f10d817527e521b0f95253c1d8cfdf2db4b3a03195958e15a7638222743fd38c',
            None,[7000003],Path('docs/results/striking-pilot/retention-1572864/ppo_retention/fresh/selfplay.json'))]
        for role in (0,1,None):
            profile='selfplay' if role is None else f'neutral-{role}'
            jobs.append(('latest-'+profile,latest,'63c06464fa0ce0ec238d349e4fac559c454a323f235fc8630a5bf599753ce8e9',
                role,list(range(9000003,9000007)),Path('docs/results/striking-pilot/neutral-retention-1638400/final-fresh')/(profile+'.json')))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            maximum_world_controls=13000,diagnostic_world_controls=0,formal_world_controls_before_job=699791,
            diagnostic_ppo_retained=1581312,new_optimizer_updates=0,assistance=False,policy_adoption='none',
            promotion_eligible=False,unchanged_reward=True,gamma=.99,dt=.02,
            discount_efolding_seconds={str(g):-.02/math.log(g) for g in (.99,.995,.999)},checks=[])
        atomic_json(path,state)
        try:
            for name,checkpoint,sha,role,seeds,reference in jobs:
                load_checked(checkpoint,sha);folder=args.output/name;folder.mkdir(exist_ok=False)
                state.update(active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                    '--seeds',*[str(s) for s in seeds],'--output',str(folder/'evaluation.json'),
                    '--progress',str(folder/'evaluation-progress.json'),'--trace-directory',str(folder/'traces')]
                if role is not None:command+=['--neutral-partner',str(role)]
                with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                report=json.loads((folder/'evaluation.json').read_text());old=json.loads(reference.read_text())
                by_trial={(t['seed'],t['swap']):t for t in old['trials']}
                assert len(report['trials'])==len(seeds)*2 and report['striking_sha256']==sha
                summaries=[]
                for trial in report['trials']:
                    assert all(trial.get(k)==v for k,v in by_trial[trial['seed'],trial['swap']].items())
                    with np.load(trial['trace_file']) as arrays:
                        parts={key:arrays['reward_'+key] for key in TERMS};actual=arrays['reward_base_total'];n=len(actual)
                        assert all(np.isfinite(x).all() for x in (*parts.values(),actual))
                        np.testing.assert_allclose(sum(parts.values()),actual,rtol=1e-6,atol=1e-7)
                        returns={}
                        for gamma in (.99,.995,.999,1.):
                            w=gamma**np.arange(n)[:,None]
                            returns[str(gamma)]=dict(components={k:(w*v).sum(axis=0).tolist() for k,v in parts.items()},
                                actual_base_total=(w*actual).sum(axis=0).tolist())
                        summaries.append(dict(seed=trial['seed'],swap=trial['swap'],survived=trial['survived'],
                            controls=n,scored_by_corner=trial['scored_by_corner'],learner_corner=role,
                            discounted_observed_returns=returns))
                state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
                assert state['diagnostic_world_controls']<=13000
                state['checks'].append(dict(name=name,sha256=sha,all_original_physical_fields_exact=True,
                    unchanged_rewards_reconstruct=True,trials=summaries))
                atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state['diagnostic_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
