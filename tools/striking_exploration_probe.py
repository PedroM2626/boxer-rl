"""Read-only factorial arm-noise coverage probe; no optimizer or promotion."""
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
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'probe.json'
        if path.exists():raise ValueError('probe exists; inspect state before another launch')
        source=Path('checkpoints/striking-validation/partner-duration-1441792')
        manifest=json.loads((source/'latest.json').read_text())
        if hashlib.sha256((source/manifest['file']).read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('source checksum mismatch')
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            source_sha256=manifest['sha256'],optimization_transitions=0,policy_adoption='none',
            promotion_eligible=False,diagnostic_ppo_retained=1188096,
            maximum_world_controls=32000,diagnostic_world_controls=0,jobs={})
        atomic_json(path,state)
        try:
            for hold,scale in ((1,1.),(10,1.),(1,3.),(10,3.)):
                for corner in (0,1):
                    name=f'hold-{hold}-scale-{scale:g}-role-{corner}'
                    state.update(active_job=name,updated_utc=utc());atomic_json(path,state)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(source),
                        '--output',str(args.output/(name+'.json')),'--sampled','--neutral-partner',str(corner),
                        '--arm-noise-hold',str(hold),'--arm-noise-scale',str(scale)]
                    with (args.output/(name+'.log')).open('w') as out,(args.output/(name+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    report=json.loads((args.output/(name+'.json')).read_text());rows=report['trials']
                    if (report['striking_sha256']!=manifest['sha256'] or len(rows)!=8 or
                            {(x['seed'],x['swap']) for x in rows}!={(s,w) for s in (7,23,31,47) for w in (False,True)} or
                            not all(x['finite'] for x in rows) or report['episode_seconds']!=10.):
                        raise ValueError('incomplete fixed-policy coverage evaluation')
                    state['jobs'][name]=dict(pair_survivors=sum(x['survived'] for x in rows),
                        own_events=sum(x['scored_by_corner'][corner] for x in rows),
                        own_surviving_events=sum(x['scored_by_corner'][corner] for x in rows if x['survived']),
                        own_surviving_scoring_seeds=sorted({x['seed'] for x in rows if x['survived'] and x['scored_by_corner'][corner]}),
                        diagnostic_world_controls=report['simulated_world_transitions'])
                    state['diagnostic_world_controls']+=report['simulated_world_transitions']
                    atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
