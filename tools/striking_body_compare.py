"""Matched diagnostic comparison of zero versus 0.1-rad body residual authority.

Both arms receive identical 161-input/32-output warm-state expansion. Only the
authority to apply the final fifteen joint residuals differs. No main adoption.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_body import WholeBodyStrikingEnv, expand_learning_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('checkpoints/striking-pilot/step-1048576'))
    parser.add_argument('--steps',type=int,default=131072)
    parser.add_argument('--authority',type=float,choices=(0.,.1))
    parser.add_argument('--ceiling',type=int)
    args=parser.parse_args()
    if args.authority is not None:
        import boxer.striking_train as training
        class SelectedInterface(WholeBodyStrikingEnv):
            body_residual_radians=args.authority
        training.StrikingEnv=SelectedInterface
        training.main(['--output',str(args.output),'--resume','--ceiling',str(args.ceiling),
                       '--worlds','4','--unroll','128'])
        return
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=args.output/'comparison.json'
        if state_path.exists():raise ValueError('comparison exists; inspect it before any launch')
        if args.steps<=0 or args.steps%1024:raise ValueError('complete PPO block budget required')
        manifest=json.loads((args.source/'latest.json').read_text())
        payload=(args.source/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('source hash mismatch')
        saved=pickle.loads(payload)
        expanded=expand_learning_state(saved)
        expanded_payload=pickle.dumps(expanded,protocol=pickle.HIGHEST_PROTOCOL)
        expanded_sha=hashlib.sha256(expanded_payload).hexdigest()
        offset=manifest['consumed_steps']
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),
            source_sha256=manifest['sha256'],expanded_sha256=expanded_sha,
            main_consumed_retained=offset,diagnostic_steps_per_arm=args.steps,
            diagnostic_steps_ceiling=2*args.steps,diagnostic_consumed_steps=0,
            diagnostic_output=str(args.output),policy_adoption='none',arms={})
        atomic_json(state_path,state)
        try:
            for name,authority in (('frozen_body',0.),('adaptive_body',.1)):
                folder=args.output/name;checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True,exist_ok=False)
                file=f'step-{offset}-expanded.pkl';(checkpoint/file).write_bytes(expanded_payload)
                atomic_json(checkpoint/'latest.json',dict(manifest,file=file,sha256=expanded_sha,
                    actor_inputs=161,actions=32,body_residual_radians=authority))
                config=json.loads((args.source/'config.json').read_text())
                atomic_json(folder/'config.json',dict(config,actor_inputs=161,actions=32,
                                                     body_residual_radians=authority))
                atomic_json(folder/'progress.json',dict(consumed_steps=offset))
                atomic_json(folder/'diagnostic-profile.json',dict(source_sha256=manifest['sha256'],
                    expanded_sha256=expanded_sha,lineage_counter_offset=offset,
                    body_residual_radians=authority,reward='unchanged original production reward',
                    migration='Old parameters/moments retained; added weights/moments zero; added log_std -1.9; normalizer new mean0 var1; Adam count, RNG and consumed counters retained.',
                    assistance=False,main_training=False,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
                state.update(active_arm=name,phase='training',updated_utc=utc());atomic_json(state_path,state)
                with (folder/'training.log').open('w') as stdout,(folder/'errors.log').open('w') as stderr:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--output',str(folder),
                        '--authority',str(authority),'--ceiling',str(offset+args.steps)],
                        stdout=stdout,stderr=stderr,check=True)
                progress=json.loads((folder/'progress.json').read_text());delta=progress['consumed_steps']-offset
                if delta!=args.steps or progress['optimized_steps']!=offset+args.steps:
                    raise ValueError('diagnostic arm did not complete its exact block budget')
                state['diagnostic_consumed_steps']+=delta
                state['arms'][name]=dict(status='evaluating',diagnostic_consumed_steps=delta)
                state.update(phase='evaluation',updated_utc=utc());atomic_json(state_path,state)
                with (folder/'evaluation.log').open('w') as stdout,(folder/'evaluation-errors.log').open('w') as stderr:
                    subprocess.run([sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                        '--output',str(folder/'evaluation.json')],stdout=stdout,stderr=stderr,check=True)
                result=json.loads((folder/'evaluation.json').read_text());trials=result['trials']
                state['arms'][name].update(status='completed',pair_survivors=sum(x['survived'] for x in trials),
                    scored_by_corner=np.sum([x['scored_by_corner'] for x in trials],axis=0).tolist(),
                    complete_episode_impacts=sum(sum(x['scored_by_corner']) for x in trials if x['survived']),
                    scoring_seeds=sorted({x['seed'] for x in trials if sum(x['scored_by_corner'])}),
                    evaluation_world_controls=result['simulated_world_transitions'],
                    checkpoint=json.loads((checkpoint/'latest.json').read_text()))
                atomic_json(state_path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(state_path,state)
        except BaseException as error:
            state['diagnostic_consumed_steps']=sum(max(0,json.loads(p.read_text())['consumed_steps']-offset)
                                                 for p in args.output.glob('*/progress.json'))
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(state_path,state)
            raise


if __name__=='__main__':main()
