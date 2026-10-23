"""Verify fresh matched prefixes and export deduplicated pre-contact candidates."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.striking_train import Actor,normalize
from tools.evaluate_striking import ablate_action
from tools.striking_anchor_compare import SEEDS
from tools.striking_contact_compare import load_checked
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA


def candidate_reasons(row,trace):
    reasons=[]
    if not row['survived'] or len(trace['action'])!=50:reasons.append('incomplete_or_fallen_prefix')
    if not all(np.isfinite(x).all() for x in trace.values()):reasons.append('nonfinite_state')
    if min(trace['root_distance_before'].min(),trace['root_distance_after'].min())<=.9:
        reasons.append('root_separation_at_or_below_0.9m')
    if any(any(g.startswith('red/') for g in c['geoms']) and any(g.startswith('blue/') for g in c['geoms'])
            for c in row['first_observed_contacts']):reasons.append('positive_force_inter_body_contact')
    return reasons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=json.loads((args.output/'anchors.json').read_text())
    if state['status']!='completed' or len(state['checks'])!=6:raise ValueError('all fresh prefixes required')
    saved,_=load_checked(SOURCE,SOURCE_SHA)
    predict=jax.jit(lambda x:jp.tanh(Actor().apply(saved['actor'],x)[0]))
    archive=Path('docs/results/striking-pilot/precontact-anchors-bc4000');archive.mkdir(parents=True,exist_ok=True)
    first={};accepted={};comparisons=[];controls=0;outcomes={}
    for arm in ('original','zero_both'):
        outcomes[arm]={}
        for corner in (0,1,None):
            mode='selfplay' if corner is None else 'neutral-'+str(corner)
            folder=args.output/arm;report=json.loads((folder/(mode+'.json')).read_text())
            progress=json.loads((folder/(mode+'-progress.json')).read_text())
            if (report['striking_sha256']!=SOURCE_SHA or report['simulated_world_transitions']!=progress['simulated_world_transitions'] or
                    progress['status']!='completed' or len(report['trials'])!=16 or report['episode_seconds']!=1. or
                    report['diagnostic_zero_command_offsets']!=(arm=='zero_both') or
                    report['diagnostic_zero_arm_residuals']!=(arm=='zero_both')):
                raise ValueError('fresh prefix provenance mismatch')
            if {(r['seed'],r['swap']) for r in report['trials']}!={(s,w) for s in SEEDS for w in (False,True)}:
                raise ValueError('fresh sample set mismatch')
            controls+=report['simulated_world_transitions']
            eligible=0
            for row in report['trials']:
                file=Path(row['trace_file'])
                if hashlib.sha256(file.read_bytes()).hexdigest()!=row['trace_sha256']:raise ValueError('trace checksum mismatch')
                with np.load(file) as payload:trace={k:payload[k] for k in payload.files}
                if not all(np.isfinite(x).all() for x in trace.values()):raise FloatingPointError('nonfinite trace')
                np.testing.assert_array_equal(trace['next_observation'][:-1],trace['observation'][1:])
                np.testing.assert_array_equal(trace['root_distance_after'][:-1],trace['root_distance_before'][1:])
                np.testing.assert_array_equal(trace['scored'].sum(axis=0),row['scored_by_corner'])
                np.testing.assert_array_equal(trace['fallen'][-1],row['fallen'])
                shape=trace['observation'].shape
                prediction=np.asarray(predict(jp.array(normalize(trace['observation'].reshape(-1,148),saved['norm']))))
                expected=ablate_action(prediction,arm=='zero_both',arm=='zero_both').reshape(shape[0],2,17).copy()
                if corner is not None:expected[:,1-corner]=0.
                np.testing.assert_allclose(expected,trace['action'],atol=2e-6)
                key=(mode,row['seed'],row['swap'])
                if arm=='original':first[key]=trace['observation'][0].copy()
                else:np.testing.assert_array_equal(first[key],trace['observation'][0])
                reasons=candidate_reasons(row,trace)
                eligible+=not reasons
                comparisons.append(dict(profile=arm,evaluation=mode,seed=row['seed'],swap=row['swap'],
                    survived=row['survived'],elapsed=row['elapsed'],exclusion_reasons=reasons,
                    minimum_separation=float(min(trace['root_distance_before'].min(),trace['root_distance_after'].min()))))
                if arm=='zero_both' and not reasons:
                    for role in ((0,1) if corner is None else (corner,)):
                        key=(row['seed'],row['swap'],role)
                        obs=trace['observation'][:,role].copy();actions=trace['action'][:,role].copy()
                        np.testing.assert_array_equal(actions,0.)
                        if key in accepted:
                            np.testing.assert_array_equal(obs,accepted[key]['observation'])
                            np.testing.assert_array_equal(actions,accepted[key]['action'])
                        else:accepted[key]=dict(observation=obs,action=actions,seed=row['seed'],swap=row['swap'],
                            role=role,trajectory=file.name,trace_sha256=row['trace_sha256'],evaluation=mode)
            outcomes[arm][mode]=dict(pair_survivors=sum(r['survived'] for r in report['trials']),
                eligible_precontact_prefixes=eligible)
            destination=archive/arm;destination.mkdir(exist_ok=True)
            for file in folder.glob(mode+'*.log'):
                text=file.read_text().lower()
                if any(w in text for w in ('badqpos','badqvel','badqacc','cnstrfull','contactfull','nefc overflow','nonfinite','traceback')):
                    raise ValueError('native diagnostic failure: '+str(file))
            for file in (folder/(mode+'.json'),folder/(mode+'-progress.json'),folder/(mode+'.log'),folder/(mode+'-errors.log')):
                shutil.copy2(file,destination/file.name)
            traces=destination/(mode+'-traces');traces.mkdir(exist_ok=True)
            for file in (folder/(mode+'-traces')).glob('*.npz'):shutil.copy2(file,traces/file.name)
    if controls!=state['diagnostic_world_controls'] or controls>4800:raise ValueError('physical accounting mismatch')
    dataset_manifest=None
    if accepted:
        records=list(accepted.values())
        arrays=dict(observation=np.concatenate([r['observation'] for r in records]),action=np.concatenate([r['action'] for r in records]),
            role=np.repeat([r['role'] for r in records],50),seed=np.repeat([r['seed'] for r in records],50),
            swap=np.repeat([r['swap'] for r in records],50),trajectory_id=np.repeat(np.arange(len(records)),50))
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue()
        atomic_bytes(archive/'anchors.npz',data)
        dataset_manifest=dict(file='anchors.npz',sha256=hashlib.sha256(data).hexdigest(),rows=len(arrays['observation']),
            unique_role_trajectories=len(records),rows_by_role={str(role):int((arrays['role']==role).sum()) for role in (0,1)},
            source_sha256=SOURCE_SHA,selection='complete finite precontact prefix; separation >0.9m; no interbody positive force',
            contexts_deduplicated=True,records=[{k:v for k,v in r.items() if k not in ('observation','action')} for r in records])
        atomic_json(archive/'dataset.json',dataset_manifest)
    result=dict(source_sha256=SOURCE_SHA,matched_initial_observations_exact=True,outcomes=outcomes,trials=comparisons,
        dataset=dataset_manifest,diagnostic_world_controls=controls,formal_world_controls_total=398434+controls,
        new_ppo_transitions=0,new_supervised_updates=0,diagnostic_ppo_retained=1188096,cumulative_supervised_updates=6000,
        main_project_consumed=476041216,policy_adoption='none',promotion=False,native_numerical_or_capacity_warnings=False,
        limitation='One-second selected precontact prefixes, not complete boxing demonstrations or certified safety.',
        decision='Review candidate coverage and both profile outcomes before any explicitly bounded anchoring update allocation.')
    shutil.copy2(args.output/'anchors.json',archive/'anchors.json');atomic_json(archive/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('outcomes','diagnostic_world_controls','formal_world_controls_total')},indent=2))
    print('unique candidate role trajectories:',len(accepted))


if __name__=='__main__':main()
