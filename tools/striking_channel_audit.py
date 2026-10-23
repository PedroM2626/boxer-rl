"""Bounded causal action-channel intervention after traced pre-contact falls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_contact_compare import load_checked
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'channels.json'
        if path.exists():raise ValueError('channel audit exists; inspect before recovery')
        load_checked(SOURCE,SOURCE_SHA)
        prerequisite=json.loads(Path('docs/results/striking-pilot/policy-coverage-contact-bc4000/analysis.json').read_text())
        if prerequisite['formal_world_controls_total']!=395041 or not prerequisite['complete_original_physical_results_match_exactly']:
            raise ValueError('reviewed trace prerequisite mismatch')
        state=dict(status='running',phase='channel_interventions',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,maximum_world_controls=3600,
            diagnostic_world_controls=0,formal_world_controls_before_job=395041,
            new_ppo_transitions=0,new_supervised_updates=0,assistance=False,
            promotion_eligible=False,policy_adoption='none',checks=[])
        atomic_json(path,state)
        try:
            for arm,flags in (('zero_commands',['--zero-command-offsets']),
                    ('zero_arms',['--zero-arm-residuals']),
                    ('zero_both',['--zero-command-offsets','--zero-arm-residuals'])):
                folder=args.output/arm;folder.mkdir(exist_ok=False)
                for corner in (0,None):
                    name='selfplay' if corner is None else 'neutral-0'
                    state.update(active_arm=arm,active_evaluation=name,updated_utc=utc());atomic_json(path,state)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),
                        '--seeds','23','47','--seconds','3','--output',str(folder/(name+'.json')),
                        '--progress',str(folder/(name+'-progress.json')),
                        '--trace-directory',str(folder/(name+'-traces'))]+flags
                    if corner is not None:command+=['--neutral-partner',str(corner)]
                    with (folder/(name+'.log')).open('w') as out,(folder/(name+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    report=json.loads((folder/(name+'.json')).read_text());rows=report['trials']
                    state['diagnostic_world_controls']+=report['simulated_world_transitions'];atomic_json(path,state)
                    if (len(rows)!=4 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in (23,47) for w in (False,True)} or
                            report['striking_sha256']!=SOURCE_SHA or not all(r['finite'] for r in rows)):
                        raise ValueError('incomplete channel intervention')
                    state['checks'].append(dict(arm=arm,evaluation=name,pair_survivors=sum(r['survived'] for r in rows),
                        world_controls=report['simulated_world_transitions']))
                    atomic_json(path,state)
            state.update(status='completed',phase='awaiting_analysis',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
