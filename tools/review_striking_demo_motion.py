"""Verify the complete retained-demo manifest and versioned motion measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU motion review required')
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_train import utc
from boxer.strike_motion import evaluate_motion,PROFILE
from boxer.striking_punch_trace import describe_events
from tools.striking_demo_motion_audit import BASE,LIMIT,DATA,CONTROL,CONTROL_SHA,SOURCE_SHA
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();live=args.input
    state=json.loads((live/'audit.json').read_text());progress=json.loads((live/'progress.json').read_text())
    assert state['status']=='completed' and len(state['trials'])==15
    assert state['source_sha256']==SOURCE_SHA and state['current_retention_checkpoint_sha256']==CONTROL_SHA
    assert state['maximum_world_controls']==LIMIT and state['formal_world_controls_before_job']==BASE
    assert state['diagnostic_world_controls']==progress['diagnostic_world_controls']==LIMIT
    assert state['new_optimizer_updates']==state['new_ppo_steps']==state['new_supervised_updates']==0
    assert not (live/'native-errors.log').read_text().strip()
    assert verify_prerequisites()==state['prerequisites']
    manifest=json.loads((DATA/'replay.json').read_text());original=json.loads((DATA/'feasibility.json').read_text())
    assert hashlib.sha256((DATA/'replay.json').read_bytes()).hexdigest()==state['replay_manifest_sha256']
    assert hashlib.sha256((DATA/'feasibility.json').read_bytes()).hexdigest()==state['feasibility_manifest_sha256']
    control,_=load_checked(CONTROL,CONTROL_SHA)
    assert state['demonstration_files']==control['retention_provenance']['demonstration_files']
    assert {r['trajectory'] for r in state['trials']}=={r['trajectory'] for r in manifest['trials']}
    all_events=[];learner_events=[];rows=[]
    for row in state['trials']:
        name=row['trajectory'];source=next(r for r in manifest['trials'] if r['trajectory']==name)
        expected=next(r for r in original['trials'] if r.get('diagnostic_trajectory')==name)
        assert row['observations_match_exactly'] and row['events_match_exactly'] and row['survived']
        assert row['learner_corner']==expected['learner_corner'] and row['seed']==expected['seed'] and row['swap']==expected['swap']
        assert row['impact_events']==source['impact_events']==expected['impact_events']
        assert row['scored_by_corner']==expected['scored_by_corner']
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==row['demonstration_sha256']==source['sha256']
        path=live/'native'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['native_trace_sha256']
        with np.load(path) as arrays:
            assert len(arrays['time_before'])==5000
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            np.testing.assert_allclose(np.diff(arrays['time_before']),.002,atol=1e-12,rtol=0)
            events=describe_events(arrays);assert events==row['all_hand_measurements']
        assert len(events)==sum(row['scored_by_corner'])
        learner=[e for e in events if e['corner']==row['learner_corner']]
        evidence=evaluate_motion(learner);assert evidence==row['learner_motion_evidence']
        rows.append(dict(trajectory=name,learner_corner=row['learner_corner'],legacy_events=len(learner),
            both_reference_events=evidence['both_reference_components_count']))
        all_events+=events;learner_events+=learner
    # Apply the very same versioned interface to the archived learned-policy audit,
    # without simulating again or substituting its selected episodes for holdouts.
    learned=Path('docs/results/striking-pilot/punch-quality-2080768/audit.json')
    learned_state=json.loads(learned.read_text());learned_rows=[]
    for case in learned_state['cases']:
        events=[e for trial in case['trials'] for e in trial['events']]
        learned_rows.append(dict(case=case['case'],sha256=case['sha256'],evidence=evaluate_motion(events)))
    result=dict(status='reviewed',motion_profile=dict(PROFILE),demonstrations=15,all_manifest_files_exactly_replayed=True,
        native_errors_empty=True,source_sha256=SOURCE_SHA,current_retention_checkpoint_sha256=CONTROL_SHA,
        all_hand_legacy_events=len(all_events),learner_legacy_events=len(learner_events),
        learner_both_reference_events=sum(e['onset_relative_speed']>=1 and e['onset_forward_cosine']>=.7 for e in learner_events),
        learner_below_speed_reference=sum(e['onset_relative_speed']<1 for e in learner_events),
        learner_below_forward_reference=sum(e['onset_forward_cosine']<.7 for e in learner_events),
        learner_censored_post_windows=sum(not e['post_window_complete'] for e in learner_events),
        new_physical_controls=LIMIT,formal_world_controls_total=BASE+LIMIT,diagnostic_ppo_total=2163968,
        new_optimizer_updates=0,new_supervised_updates=0,main_adoption=False,promotion=False,trials=rows,
        learned_evidence_source_sha256=hashlib.sha256(learned.read_bytes()).hexdigest(),learned_policy_evidence=learned_rows,
        decision='Do not run more unchanged retention/BC/PPO. Establish a physically validated positive strike-cycle target under the same free-body motor/arm authority before a versioned objective comparison. Reference components alone cannot certify boxing.',
        limitations=['Selected surviving legacy-contact demonstrations, not a generalization or failure-rate estimate.',
            'A component failure is not proof that every possible punch type is invalid.',
            'The generator stopping after its first contact is a verified target-design issue, not an isolated causal explanation of all learned behavior.',
            'No original boxer/env.py policy was replayed and no teacher was deployed as a trained boxer.'])
    for key in ('onset_relative_speed','onset_forward_cosine','pre_max_displacement_m','pre_elbow_excursion_rad'):
        x=np.asarray([e[key] for e in learner_events])
        result[key]=dict(minimum=float(x.min()),median=float(np.median(x)),maximum=float(x.max())) if len(x) else dict(count=0)
    archive=Path('docs/results/striking-pilot/demo-motion-2080768');archive.mkdir(parents=True,exist_ok=True)
    shutil.copytree(live/'native',archive/'native',dirs_exist_ok=True)
    for name in ('audit.json','progress.json','native-errors.log'):shutil.copy2(live/name,archive/name)
    atomic_json(archive/'analysis.json',result);atomic_json(live/'analysis.json',result)
    for folder in (archive,live):
        if not (folder/'completed-before-review.json').exists():atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        reviewed=dict(state,phase='independently_reviewed',reviewed_utc=utc(),continuation_decision=result['decision'])
        atomic_json(folder/'audit.json',reviewed)
    print(json.dumps({k:result[k] for k in ('status','learner_legacy_events','learner_both_reference_events','formal_world_controls_total')}))


if __name__=='__main__':main()
