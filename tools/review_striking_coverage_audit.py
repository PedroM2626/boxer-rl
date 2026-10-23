"""Verify passive traces and analyze same-role demonstration coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_coverage import block_distances,leave_trajectory_out_baselines
from boxer.striking_train import Actor,normalize
from tools.striking_contact_compare import load_checked,load_demonstrations
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA,REFERENCE


def summarize(distances,mask,baseline,times):
    return {name:dict(rows=int(mask.sum()),mean=float(values[mask].mean()),
        q95=float(np.quantile(values[mask],.95)),
        fraction_above_demonstration_q99=float(np.mean(values[mask]>baseline[name]['q99']+1e-8)),
        first_time_above_q99=float(times[np.flatnonzero(mask & (values>baseline[name]['q99']+1e-8))[0]])
            if np.any(mask & (values>baseline[name]['q99']+1e-8)) else None)
        for name,values in distances.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=json.loads((args.output/'coverage.json').read_text())
    if state['status']!='completed' or len(state['checks'])!=3:raise ValueError('complete traces required')
    saved,manifest=load_checked(SOURCE,SOURCE_SHA)
    observations,labels,flags,files=load_demonstrations()
    roles=np.repeat([int(name.split('-role-')[1][0]) for name in files],500)
    ids=np.repeat(np.arange(15),500)
    data=normalize(np.c_[observations,flags],saved['norm'])
    baselines=leave_trajectory_out_baselines(data,roles,ids)
    inference=jax.jit(lambda x:jp.tanh(Actor().apply(saved['actor'],x)[0]))
    archive=Path('docs/results/striking-pilot/policy-coverage-contact-bc4000')
    archive.mkdir(parents=True,exist_ok=True)
    rows=[];controls=0
    for corner in (0,1,None):
        name='selfplay' if corner is None else 'neutral-'+str(corner)
        report=json.loads((args.output/(name+'.json')).read_text())
        previous=json.loads((REFERENCE/(name+'.json')).read_text())
        progress=json.loads((args.output/(name+'-progress.json')).read_text())
        if (report['striking_sha256']!=SOURCE_SHA or report['simulated_world_transitions']!=progress['simulated_world_transitions'] or
                progress['status']!='completed' or len(report['trials'])!=8):raise ValueError('trace report provenance mismatch')
        controls+=report['simulated_world_transitions']
        for row,original in zip(report['trials'],previous['trials']):
            if any(row[k]!=v for k,v in original.items()):raise ValueError('original physical result changed')
            file=Path(row['trace_file'])
            if hashlib.sha256(file.read_bytes()).hexdigest()!=row['trace_sha256']:raise ValueError('trace checksum mismatch')
            with np.load(file) as payload:trace={k:payload[k] for k in payload.files}
            if not all(np.isfinite(x).all() for x in trace.values()):raise FloatingPointError('nonfinite trace')
            np.testing.assert_array_equal(trace['next_observation'][:-1],trace['observation'][1:])
            np.testing.assert_array_equal(trace['fallen'][-1],row['fallen'])
            np.testing.assert_array_equal(trace['scored'].sum(axis=0),row['scored_by_corner'])
            if len(trace['action'])>500:raise ValueError('physical trace exceeds round limit')
            for role in ((0,1) if corner is None else (corner,)):
                obs=trace['observation'][:,role]
                normalized=normalize(obs,saved['norm'])
                predicted=np.asarray(inference(jp.array(normalized)))
                np.testing.assert_allclose(predicted,trace['action'][:,role],atol=2e-6)
                distances=block_distances(data[roles==role],normalized)
                before=~np.any(obs[:,146:],axis=1)
                subsets={}
                for phase,mask in (('before_hit',before),('after_hit',~before)):
                    if mask.any():subsets[phase]=summarize(distances,mask,baselines[str(role)],trace['time_before'])
                cross=[c['elapsed'] for c in row['first_observed_contacts'] if
                    any(g.startswith('red/') for g in c['geoms']) and any(g.startswith('blue/') for g in c['geoms'])]
                own_force=[c['elapsed'] for c in row['first_observed_contacts'] if any(
                    g in (('red/' if role==0 else 'blue/')+hand for hand in ('left_hand_collision','right_hand_collision'))
                    for g in c['geoms']) and any(g in (('blue/' if role==0 else 'red/')+target
                    for target in ('head_collision','torso_collision')) for g in c['geoms'])]
                own_scores=[hand['score_time'] for event in row['impact_events'] for hand in event['hands'] if hand['corner']==role]
                rows.append(dict(evaluation=name,seed=row['seed'],swap=row['swap'],role=role,
                    pair_survived=row['survived'],own_fell=row['fallen'][role],controls=len(obs),
                    first_cross_body_force=min(cross) if cross else None,
                    first_own_fist_target_force=min(own_force) if own_force else None,
                    first_own_valid_score=min(own_scores) if own_scores else None,
                    first_pair_fall=row['first_fall_time'],terminal=row['terminal_state']['corners'][role],
                    minimum_pelvis=float(trace['pelvis_height'][:,role].min()),
                    minimum_upright=float(trace['upright'][:,role].min()),
                    phase_coverage=subsets))
        for file in args.output.glob(name+'*.log'):
            text=file.read_text().lower()
            if any(w in text for w in ('badqpos','badqvel','badqacc','cnstrfull','contactfull','nefc overflow','nonfinite','traceback')):
                raise ValueError('native diagnostic failure: '+str(file))
        for file in (args.output/(name+'.json'),args.output/(name+'-progress.json'),args.output/(name+'.log'),args.output/(name+'-errors.log')):
            shutil.copy2(file,archive/file.name)
        destination=archive/(name+'-traces');destination.mkdir(exist_ok=True)
        for file in (args.output/(name+'-traces')).glob('*.npz'):shutil.copy2(file,destination/file.name)
    if controls!=state['diagnostic_world_controls'] or controls>12000:raise ValueError('consumed control accounting mismatch')
    aggregate={}
    for name in ('neutral-0','neutral-1','selfplay'):
        selected=[r for r in rows if r['evaluation']==name]
        # Count pair outcomes once despite two controlled corners in self-play.
        pairs={(r['seed'],r['swap']):r for r in selected}
        aggregate[name]=dict(pair_survivors=sum(r['pair_survived'] for r in pairs.values()),
            pair_falls=sum(not r['pair_survived'] for r in pairs.values()),
            pair_falls_without_preceding_cross_body_force=sum(not r['pair_survived'] and
                (r['first_cross_body_force'] is None or r['first_cross_body_force']>r['first_pair_fall']) for r in pairs.values()),
            controlled_role_falls=sum(r['own_fell'] for r in selected),
            controlled_role_falls_without_own_valid_score=sum(r['own_fell'] and r['first_own_valid_score'] is None for r in selected),
            low_pelvis_flags=sum(r['own_fell'] and r['terminal']['low_pelvis'] for r in selected),
            tilt_flags=sum(r['own_fell'] and r['terminal']['tilted'] for r in selected),
            self_leg_contact_flags=sum(r['own_fell'] and r['terminal']['self_leg_contact'] for r in selected))
    result=dict(source_sha256=SOURCE_SHA,complete_original_physical_results_match_exactly=True,
        trace_rows_checked=len(rows),new_supervised_updates=0,new_ppo_transitions=0,
        diagnostic_world_controls=controls,formal_world_controls_total=385343+controls,
        diagnostic_ppo_retained=1188096,cumulative_supervised_updates=6000,
        foundation_consumed=474992640,main_boxing_consumed=1048576,main_project_consumed=476041216,
        native_numerical_or_capacity_warnings=False,policy_adoption='none',promotion=False,
        leave_trajectory_out_baselines=baselines,physical_summary=aggregate,controlled_role_trials=rows,
        limitation='Successful-data marginal distances are descriptive, not joint coverage, causal proof or changed gates.',
        decision='Review first-fall and coverage evidence before allocating learning; no automatic promotion or extension.')
    shutil.copy2(args.output/'coverage.json',archive/'coverage.json')
    atomic_json(archive/'analysis.json',result)
    print(json.dumps(dict(controls=controls,total=385343+controls,physical_summary=aggregate),indent=2))


if __name__=='__main__':main()
