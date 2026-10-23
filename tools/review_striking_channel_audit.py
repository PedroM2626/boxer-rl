"""Archive all short channel interventions against already traced baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_train import Actor,normalize
from tools.evaluate_striking import ablate_action
from tools.striking_contact_compare import load_checked
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=json.loads((args.output/'channels.json').read_text())
    if state['status']!='completed' or len(state['checks'])!=6:raise ValueError('all six evaluations must complete')
    saved,_=load_checked(SOURCE,SOURCE_SHA)
    predict=jax.jit(lambda x:jp.tanh(Actor().apply(saved['actor'],x)[0]))
    archive=Path('docs/results/striking-pilot/action-channels-contact-bc4000')
    archive.mkdir(parents=True,exist_ok=True)
    original=Path('docs/results/striking-pilot/policy-coverage-contact-bc4000')
    baseline={};rows=[];controls=0
    for mode in ('neutral-0','selfplay'):
        report=json.loads((original/(mode+'.json')).read_text())
        for trial in report['trials']:
            if trial['seed'] not in (23,47):continue
            file=original/(mode+'-traces')/Path(trial['trace_file']).name
            with np.load(file) as trace:
                ix=np.flatnonzero(trace['time_after']<=3.+1e-8)
                last=int(ix[-1]);fell=trace['fallen'][last].tolist()
                baseline[(mode,trial['seed'],trial['swap'])]=dict(
                    three_second_pair_survived=not any(fell),fallen=fell,
                    elapsed=float(trace['time_after'][last]),controls=len(ix))
    for arm in ('zero_commands','zero_arms','zero_both'):
        zero_commands=arm in ('zero_commands','zero_both');zero_arms=arm in ('zero_arms','zero_both')
        for mode in ('neutral-0','selfplay'):
            folder=args.output/arm;report=json.loads((folder/(mode+'.json')).read_text())
            progress=json.loads((folder/(mode+'-progress.json')).read_text())
            if (report['striking_sha256']!=SOURCE_SHA or report['simulated_world_transitions']!=progress['simulated_world_transitions'] or
                    progress['status']!='completed' or len(report['trials'])!=4 or report['episode_seconds']!=3. or
                    report['diagnostic_zero_command_offsets']!=zero_commands or report['diagnostic_zero_arm_residuals']!=zero_arms):
                raise ValueError('intervention provenance mismatch')
            controls+=report['simulated_world_transitions']
            for trial in report['trials']:
                file=Path(trial['trace_file'])
                if hashlib.sha256(file.read_bytes()).hexdigest()!=trial['trace_sha256']:raise ValueError('trace checksum mismatch')
                with np.load(file) as payload:trace={k:payload[k] for k in payload.files}
                if not all(np.isfinite(x).all() for x in trace.values()):raise FloatingPointError('nonfinite trace')
                np.testing.assert_array_equal(trace['next_observation'][:-1],trace['observation'][1:])
                np.testing.assert_array_equal(trace['scored'].sum(axis=0),trial['scored_by_corner'])
                np.testing.assert_array_equal(trace['fallen'][-1],trial['fallen'])
                shape=trace['observation'].shape
                expected=np.asarray(predict(jp.array(normalize(trace['observation'].reshape(-1,148),saved['norm']))))
                expected=ablate_action(expected,zero_commands,zero_arms).reshape(shape[0],2,17).copy()
                if mode=='neutral-0':expected[:,1]=0.
                np.testing.assert_allclose(trace['action'],expected,atol=2e-6)
                if len(trace['action'])>150:raise ValueError('short diagnostic exceeded bound')
                old=baseline[(mode,trial['seed'],trial['swap'])]
                rows.append(dict(arm=arm,evaluation=mode,seed=trial['seed'],swap=trial['swap'],
                    original_three_second=old,three_second_pair_survived=trial['survived'],
                    elapsed=trial['elapsed'],fallen=trial['fallen'],terminal=trial['terminal_state']['corners'],
                    scored_by_corner=trial['scored_by_corner'],
                    prevented_original_early_pair_fall=not old['three_second_pair_survived'] and trial['survived'],
                    introduced_early_pair_fall=old['three_second_pair_survived'] and not trial['survived']))
            for file in folder.glob(mode+'*.log'):
                text=file.read_text().lower()
                if any(w in text for w in ('badqpos','badqvel','badqacc','cnstrfull','contactfull','nefc overflow','nonfinite','traceback')):
                    raise ValueError('native diagnostic failure: '+str(file))
            destination=archive/arm;destination.mkdir(exist_ok=True)
            for file in (folder/(mode+'.json'),folder/(mode+'-progress.json'),folder/(mode+'.log'),folder/(mode+'-errors.log')):
                shutil.copy2(file,destination/file.name)
            traces=destination/(mode+'-traces');traces.mkdir(exist_ok=True)
            for file in (folder/(mode+'-traces')).glob('*.npz'):shutil.copy2(file,traces/file.name)
    if controls!=state['diagnostic_world_controls'] or controls>3600:raise ValueError('channel accounting mismatch')
    aggregate={}
    for arm in ('zero_commands','zero_arms','zero_both'):
        aggregate[arm]={mode:dict(three_second_pair_survivors=sum(r['three_second_pair_survived'] for r in rows
            if r['arm']==arm and r['evaluation']==mode),
            prevented_original_early_pair_falls=sum(r['prevented_original_early_pair_fall'] for r in rows
                if r['arm']==arm and r['evaluation']==mode),
            introduced_early_pair_falls=sum(r['introduced_early_pair_fall'] for r in rows
                if r['arm']==arm and r['evaluation']==mode)) for mode in ('neutral-0','selfplay')}
    result=dict(source_sha256=SOURCE_SHA,original_three_second_pair_survivors={mode:sum(
        v['three_second_pair_survived'] for k,v in baseline.items() if k[0]==mode) for mode in ('neutral-0','selfplay')},
        interventions=aggregate,trials=rows,diagnostic_world_controls=controls,
        formal_world_controls_total=395041+controls,new_ppo_transitions=0,new_supervised_updates=0,
        diagnostic_ppo_retained=1188096,cumulative_supervised_updates=6000,
        main_project_consumed=476041216,policy_adoption='none',promotion=False,
        native_numerical_or_capacity_warnings=False,
        limitation='Three-second channel interventions on selected seeds, not full gates or learned boxing; closed-loop observations also change.',
        decision=('Arm-channel removal prevents both original early failures in each selected context, '
            'but these short interventions are not a boxing policy. Next validate pre-contact motor-preserving '
            'anchor samples on fresh seeds in a separate matched comparison bounded at 4800 world controls, '
            'zero learning. Do not train on benchmark-seed traces or extend imitation/PPO before reviewing that data.'))
    shutil.copy2(args.output/'channels.json',archive/'channels.json')
    atomic_json(archive/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('original_three_second_pair_survivors','interventions','diagnostic_world_controls','formal_world_controls_total')},indent=2))


if __name__=='__main__':main()
