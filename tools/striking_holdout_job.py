"""Predeclared read-only paired holdouts; no policy selection or optimization.

Two fixed whole-body comparison checkpoints, two action modes, 32 new seeds
and both side arrangements. Existing eight-episode screening remains failed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'holdout.json'
        if path.exists():raise ValueError('holdout already exists; inspect state before another launch')
        seeds=list(range(1000003,1000035))
        sources={}
        for arm in ('frozen_body','adaptive_body'):
            folder=Path('checkpoints/striking-validation/body-comparison-1048576')/arm
            m=json.loads((folder/'latest.json').read_text())
            if hashlib.sha256((folder/m['file']).read_bytes()).hexdigest()!=m['sha256']:
                raise ValueError('fixed source checkpoint hash mismatch')
            sources[arm]=dict(path=str(folder),sha256=m['sha256'])
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),sources=sources,
            seeds=seeds,modes=['trained','sampled'],episodes_per_job=64,total_episodes=256,
            maximum_world_controls=128000,diagnostic_world_controls=0,
            optimization_transitions=0,main_consumed_retained=1048576,
            diagnostic_ppo_consumed_retained=793344,policy_adoption='none',jobs={})
        atomic_json(path,state)
        try:
            for mode in ('trained','sampled'):
                for arm,source in sources.items():
                    name=arm+'-'+mode
                    state.update(active_job=name,updated_utc=utc());atomic_json(path,state)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',source['path'],
                        '--output',str(args.output/(name+'.json')),'--seeds',*[str(s) for s in seeds]]
                    if mode=='sampled':command.append('--sampled')
                    with (args.output/(name+'.log')).open('w') as stdout,(args.output/(name+'-errors.log')).open('w') as stderr:
                        subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
                    result=json.loads((args.output/(name+'.json')).read_text());rows=result['trials']
                    if (result['striking_sha256']!=source['sha256'] or len(rows)!=64
                            or {(x['seed'],x['swap']) for x in rows}!={(s,w) for s in seeds for w in (False,True)}
                            or not all(x['finite'] for x in rows)):
                        raise ValueError('incomplete or invalid fixed-source holdout')
                    state['jobs'][name]=dict(pair_survivors=sum(x['survived'] for x in rows),
                        impacts=sum(sum(x['scored_by_corner']) for x in rows),
                        complete_episode_impacts=sum(sum(x['scored_by_corner']) for x in rows if x['survived']),
                        surviving_scoring_episodes=sum(x['survived'] and sum(x['scored_by_corner'])>0 for x in rows),
                        scoring_seeds=sorted({x['seed'] for x in rows if sum(x['scored_by_corner'])}),
                        surviving_scoring_seeds=sorted({x['seed'] for x in rows if x['survived'] and sum(x['scored_by_corner'])}),
                        diagnostic_world_controls=result['simulated_world_transitions'])
                    state['diagnostic_world_controls']+=result['simulated_world_transitions']
                    atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state)
            raise


if __name__=='__main__':main()
