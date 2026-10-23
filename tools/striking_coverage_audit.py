"""Bounded passive policy-state and first-fall audit of the latest BC branch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/contact-history-bc2000/contact_history')
SOURCE_SHA='2da4302cb992d5067a86c8886f72bac2cf54f2def1c41f7f87201d5f76c0bf9a'
REFERENCE=Path('docs/results/striking-pilot/contact-history-bc2000/contact_history')


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'coverage.json'
        if path.exists():raise ValueError('audit exists; inspect before recovery')
        load_checked(SOURCE,SOURCE_SHA)
        state=dict(status='running',phase='physical_traces',started_utc=utc(),pid=os.getpid(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,
            maximum_world_controls=12000,diagnostic_world_controls=0,formal_world_controls_before_job=385343,
            new_ppo_transitions=0,new_supervised_updates=0,policy_adoption='none',promotion_eligible=False,
            assistance=False,checks=[])
        atomic_json(path,state)
        try:
            for corner in (0,1,None):
                name='selfplay' if corner is None else 'neutral-'+str(corner)
                state.update(active_evaluation=name,updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),
                    '--output',str(args.output/(name+'.json')),'--progress',str(args.output/(name+'-progress.json')),
                    '--trace-directory',str(args.output/(name+'-traces'))]
                if corner is not None:command+=['--neutral-partner',str(corner)]
                with (args.output/(name+'.log')).open('w') as out,(args.output/(name+'-errors.log')).open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                report=json.loads((args.output/(name+'.json')).read_text())
                state['diagnostic_world_controls']+=report['simulated_world_transitions'];atomic_json(path,state)
                original=json.loads((REFERENCE/(name+'.json')).read_text())
                if report['striking_sha256']!=SOURCE_SHA or report['simulated_world_transitions']!=original['simulated_world_transitions']:
                    raise ValueError('source or physical count mismatch')
                if len(report['trials'])!=8 or len(original['trials'])!=8:
                    raise ValueError('incomplete physical benchmark')
                for row,reference in zip(report['trials'],original['trials']):
                    if any(row[k]!=v for k,v in reference.items()):
                        raise ValueError('passive trace changed original physical trial')
                state['checks'].append(dict(evaluation=name,original_trials_match_exactly=True,
                    world_controls=report['simulated_world_transitions']))
                atomic_json(path,state)
            state.update(status='completed',phase='awaiting_analysis',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
