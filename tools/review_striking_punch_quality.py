"""Verify native replay provenance and interpret contact-motion evidence without PPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU review required')
import numpy as np
import jax

from boxer.checkpoint_io import atomic_json
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_punch_quality_audit import BASE,LIMIT,CASES
from tools.striking_pool_compare import verify_prerequisites
from tools.striking_contact_compare import load_checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input
    state=json.loads((root/'audit.json').read_text())
    assert state['status']=='completed' and len(state['cases'])==len(CASES)==4
    assert state['formal_world_controls_before_job']==BASE and state['maximum_evaluation_world_controls']==LIMIT
    assert state['new_optimizer_updates']==state['new_collection_world_controls']==0
    assert verify_prerequisites()==state['prerequisites']
    assert all(not p.read_text().strip() for p in root.rglob('*errors.log'))
    controls=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in root.glob('*/evaluation-progress.json'))
    assert controls==state['diagnostic_evaluation_world_controls']<=LIMIT
    all_events=[];summaries=[]
    for case,spec in zip(state['cases'],CASES):
        name,checkpoint,sha,seed,reference=spec
        assert case['case']==name and case['sha256']==sha
        saved,_=load_checked(Path(checkpoint),sha);require_ppo_checkpoint(saved)
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
            tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
        refpath=Path(reference);baseline=json.loads(refpath.read_text())
        assert hashlib.sha256(refpath.read_bytes()).hexdigest()==case['reference_sha256']
        report=json.loads((root/name/'evaluation.json').read_text())
        assert report['striking_sha256']==sha and report['source_sha256']==baseline['source_sha256']
        assert not any(report[k] for k in ('assistance','sampled_policy','diagnostic_zero_command_offsets',
            'diagnostic_zero_arm_residuals','diagnostic_profile_override'))
        assert report['diagnostic_base_yaw_scale']==1 and report['episode_seconds']==10
        assert len(report['trials'])==2 and {r['swap'] for r in report['trials']}=={False,True}
        events=[]
        for row in report['trials']:
            reference_row=next(r for r in baseline['trials'] if r['seed']==seed and r['swap']==row['swap'])
            for key,value in reference_row.items():
                if key not in ('trace_file','trace_sha256','first_observed_contacts','first_fall_time','terminal_state'):
                    assert row[key]==value
            # Review archived bytes independently of live absolute trace paths.
            trace=root/name/'native'/Path(row['punch_trace_file']).name
            assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['punch_trace_sha256']
            with np.load(trace) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                np.testing.assert_allclose(np.diff(arrays['time_before']),.002,atol=1e-12,rtol=0)
                np.testing.assert_allclose(arrays['time_after']-arrays['time_before'],.002,atol=1e-12,rtol=0)
                np.testing.assert_allclose(arrays['time_after'][-1],row['elapsed'],atol=1e-12,rtol=0)
                measured=describe_events(arrays);assert measured==row['punch_measurements']
                assert len(measured)==int(sum(row['scored_by_corner']))
                for e in measured:
                    assert e['onset_contact_closing']>=.3 and e['onset_contact_advancing']>=.2
                    assert e['event_window_impulse_ns']>=.5
                    np.testing.assert_allclose(e['event_window_impulse_ns'],e['native_contact_window_impulse_ns'],atol=1e-12)
                events+=measured
        assert len(events)==case['legacy_contact_event_count']
        all_events+=events
        summaries.append(dict(case=name,sha256=sha,events=len(events),
            below_speed_reference=sum(e['onset_relative_speed']<1 for e in events),
            below_forward_reference=sum(e['onset_forward_cosine']<.7 for e in events),
            both_reference_components=sum(e['onset_relative_speed']>=1 and e['onset_forward_cosine']>=.7 for e in events),
            censored_post_windows=sum(not e['post_window_complete'] for e in events),
            descriptive_distributions=case['descriptive_distributions']))
    result=dict(status='reviewed',episodes=8,legacy_contact_events=len(all_events),
        all_original_physical_trial_fields_reproduced=True,native_error_logs_empty=True,
        new_physical_controls=controls,formal_world_controls_total=BASE+controls,
        diagnostic_ppo_total=2163968,new_optimizer_updates=0,main_adoption=False,promotion=False,
        below_speed_reference=sum(e['onset_relative_speed']<1 for e in all_events),
        below_forward_reference=sum(e['onset_forward_cosine']<.7 for e in all_events),
        both_reference_components=sum(e['onset_relative_speed']>=1 and e['onset_forward_cosine']>=.7 for e in all_events),
        censored_post_windows=sum(not e['post_window_complete'] for e in all_events),cases=summaries,
        decision='No unchanged PPO extension. Implement a separate versioned punch-motion evaluation and audit archived strike demonstrations before defining the next training objective. Preserve legacy gates and current learner states; do not manufacture a pass by changing historical metrics.',
        limitations=['Eight selected known episodes, not a generalization estimate.',
            'Legacy speed/cone components are descriptive references, not a universal punch classifier.',
            'Hooks and other punch types need appropriate motion criteria; no inference that every non-forward event is a touch.',
            'Retraction toward a pre-window pose does not establish a competent guard.',
            'Earlier long-GAE is a paired-pilot actor; older boxer/env.py video policies and directed demonstrations were not replayed in this allocation.'])
    atomic_json(root/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('status','episodes','legacy_contact_events','both_reference_components','formal_world_controls_total')}))


if __name__=='__main__':main()
