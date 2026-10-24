"""Bounded same-reset motor-floor comparison; no training or promotable masks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_generalization_audit import SOURCE, SHA
from tools.striking_mixed_failure_trace import summarize_trial


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists(): raise ValueError('Motor-floor audit exists; inspect before recovery')
        reference=Path('docs/results/striking-pilot/generalization-1916928')
        review=json.loads((reference/'analysis.json').read_text())
        assert review['status']=='reviewed' and review['formal_world_controls_total']==1351541
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            source_sha256=SHA,maximum_evaluation_world_controls=28000,diagnostic_evaluation_world_controls=0,
            formal_evaluation_world_controls_before_job=1351541,diagnostic_ppo_retained=1918208,
            new_optimizer_updates=0,new_collection_world_controls=0,main_consumed_retained=1048576,
            main_project_consumed_retained=476041216,policy_adoption='none',promotion=False,checks=[])
        atomic_json(path,state)
        def account():
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        jobs=[('motor-only',None,list(range(21000003,21000019)),True)]
        # Chosen before seeing ablation outcomes: repeatable early/far and early near-contact failures.
        seeds=[21000006,21000009,21000011,21000018]
        jobs += [(name,role,seeds,False) for name,role in (('neutral-0',0),('neutral-1',1),('selfplay',None))]
        try:
            for name,role,seeds,masked in jobs:
                folder=args.output/name; folder.mkdir(exist_ok=False)
                state.update(active_evaluation=str(folder),updated_utc=utc()); atomic_json(path,state)
                cmd=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),
                    '--seeds',*[str(s) for s in seeds],'--output',str(folder/'evaluation.json'),
                    '--progress',str(folder/'evaluation-progress.json'),'--trace-directory',str(folder/'traces')]
                if role is not None: cmd += ['--neutral-partner',str(role)]
                if masked: cmd += ['--zero-command-offsets','--zero-arm-residuals']
                with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:
                    subprocess.run(cmd,stdout=out,stderr=err,check=True)
                assert not (folder/'errors.log').read_text().strip()
                r=json.loads((folder/'evaluation.json').read_text())
                assert r['striking_sha256']==SHA and len(r['trials'])==2*len(seeds) and all(t['finite'] for t in r['trials'])
                if not masked:
                    original=[t for group in range(4) for t in json.loads((reference/f'group-{group}'/(name+'.json')).read_text())['trials']]
                    original={(t['seed'],t['swap']):t for t in original}
                    for t in r['trials']:
                        assert all(t.get(k)==v for k,v in original[t['seed'],t['swap']].items()), 'Passive replay changed physics'
                summaries=[summarize_trial(t) for t in r['trials']]
                state['checks'].append(dict(name=name,masked_diagnostic_only=masked,all_original_physical_fields_exact=not masked,trials=summaries))
                account(); assert state['diagnostic_evaluation_world_controls']<=28000; atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc()); atomic_json(path,state)
        except BaseException as error:
            account(); state.update(status='failed',error=str(error),updated_utc=utc()); atomic_json(path,state); raise


if __name__=='__main__': main()
