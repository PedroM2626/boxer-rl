"""Bounded passive exact replays: legacy contact counters versus punch motion."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU punch audit required')
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

BASE=2015379
LIMIT=4000
CASES=(
    ('video_source','checkpoints/striking-validation/gae-consolidate-1998848/gate-65536',
     'bea8147f6a13dd2d473a558f2e2ac68d3492e3136d2899e2d9daf139364bb870',29000003,
     'docs/results/striking-pilot/gae-generalization-2064384/selected-replay/evaluation.json'),
    ('pool_final','checkpoints/striking-validation/pool-2064384/pool/gate-16384',
     'd93a000802e111a8c2a7c0a2902c9612add2edca57ec8a3d9e77b90ee785def5',32000003,
     'docs/results/striking-pilot/pool-2064384/pool/fresh/selfplay.json'),
    ('control_final','checkpoints/striking-validation/pool-2064384/control/gate-16384',
     '3a34aa85f0e9cf1ecb178c02f5fe945f8dd17f45dc1b323208b6083e2091b501',32000003,
     'docs/results/striking-pilot/pool-2064384/control/fresh/selfplay.json'),
    ('earlier_long_gae','checkpoints/striking-validation/gae-1966080/long_gae/gate-16384',
     '95dbdff01def598cdc907e93f1dc7bceb2f1d6b58bc22f488a1936afa86553f9',26000003,
     'docs/results/striking-pilot/gae-1966080/long_gae/fresh/selfplay.json'))


def distribution(values):
    x=np.asarray(values,dtype=float)
    return dict(count=len(x),minimum=float(x.min()),median=float(np.median(x)),maximum=float(x.max())) if len(x) else dict(count=0)


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in (args.output/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(p.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists():raise ValueError('existing punch audit requires consumed-state review before recovery')
        prior=json.loads(Path('docs/results/striking-pilot/pool-2064384/analysis.json').read_text())
        assert prior['status']=='reviewed' and prior['diagnostic_ppo_total']==2163968 and prior['formal_world_controls_total']==BASE
        prerequisites=verify_prerequisites();assert prerequisites==prior['prerequisites']
        state=dict(status='running',phase='preflight',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            maximum_evaluation_world_controls=LIMIT,formal_world_controls_before_job=BASE,
            formal_world_controls_ceiling=BASE+LIMIT,diagnostic_evaluation_world_controls=0,
            diagnostic_ppo_retained=2163968,new_optimizer_updates=0,new_collection_world_controls=0,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            policy_adoption='none',promotion=False,assistance=False,prerequisites=prerequisites,cases=[])
        atomic_json(path,state)
        def account():
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.glob('*/evaluation-progress.json'))
            assert state['diagnostic_evaluation_world_controls']<=LIMIT
        try:
            for name,checkpoint,sha,seed,reference in CASES:
                saved,_=load_checked(Path(checkpoint),sha);require_ppo_checkpoint(saved)
                baseline=json.loads(Path(reference).read_text());assert baseline['striking_sha256']==sha
                dest=args.output/name;dest.mkdir(exist_ok=False)
                state.update(phase='native_replay',active_case=name,updated_utc=utc());atomic_json(path,state)
                command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',checkpoint,
                    '--seeds',str(seed),'--output',str(dest/'evaluation.json'),
                    '--progress',str(dest/'evaluation-progress.json'),'--punch-trace-directory',str(dest/'native')]
                with (dest/'evaluation.log').open('w') as out,(dest/'evaluation-errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                account();assert not (dest/'evaluation-errors.log').read_text().strip()
                report=json.loads((dest/'evaluation.json').read_text())
                assert report['striking_sha256']==sha and report['source_sha256']==baseline['source_sha256']
                assert not report['assistance'] and not report['sampled_policy'] and len(report['trials'])==2
                measurements=[];trials=[]
                for row in report['trials']:
                    expected=next(r for r in baseline['trials'] if r['seed']==seed and r['swap']==row['swap'])
                    for key,value in expected.items():
                        if key not in ('trace_file','trace_sha256','first_observed_contacts','first_fall_time','terminal_state'):
                            assert row[key]==value, f'Passive replay changed {name}/{row["swap"]}/{key}'
                    trace=Path(row['punch_trace_file']);assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['punch_trace_sha256']
                    with np.load(trace) as arrays:
                        events=describe_events(arrays);assert events==row['punch_measurements']
                        assert len(events)==int(sum(row['scored_by_corner']))
                        assert np.isfinite(np.concatenate([arrays[k].ravel() for k in arrays.files])).all()
                    measurements+=events
                    trials.append(dict(seed=seed,swap=row['swap'],survived=row['survived'],scored_by_corner=row['scored_by_corner'],
                        events=events,all_original_trial_fields_equal=True,trace_sha256=row['punch_trace_sha256']))
                summaries={k:distribution([e[k] for e in measurements]) for k in (
                    'onset_relative_speed','onset_forward_cosine','pre_peak_relative_speed','pre_max_displacement_m',
                    'pre_elbow_excursion_rad','event_window_impulse_ns','post_contact_seconds')}
                state['cases'].append(dict(case=name,sha256=sha,seed=seed,reference=reference,
                    reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                    selected_episode_count=2,legacy_contact_event_count=len(measurements),
                    descriptive_distributions=summaries,
                    events_below_old_default_speed_component=sum(e['onset_relative_speed']<1. for e in measurements),
                    events_below_old_forward_cone_component=sum(e['onset_forward_cosine']<.7 for e in measurements),
                    incomplete_post_windows=sum(not e['post_window_complete'] for e in measurements),trials=trials))
                atomic_json(path,state)
            assert verify_prerequisites()==prerequisites
            account();state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc());atomic_json(path,state)
            archive=Path('docs/results/striking-pilot/punch-quality-2080768')
            for name,*_ in CASES:shutil.copytree(args.output/name,archive/name,dirs_exist_ok=True)
            atomic_json(archive/'audit.json',state)
            print(json.dumps(dict(status=state['status'],new_physical_controls=state['diagnostic_evaluation_world_controls'])))
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
