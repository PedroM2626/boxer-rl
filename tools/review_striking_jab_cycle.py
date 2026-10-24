"""Review every jab prototype condition without relaxing prospective criteria."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU prototype review required')
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_train import utc
from boxer.striking_punch_trace import describe_events
from boxer.jab_feasibility import PROFILE,cycle_fraction
from tools.striking_jab_cycle_audit import BASE,LIMIT,SEEDS,candidate
from tools.striking_pool_compare import verify_prerequisites


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input;state=json.loads((root/'audit.json').read_text())
    progress=json.loads((root/'progress.json').read_text())
    assert state['status']=='completed' and len(state['trials'])==16
    assert state['profile']==PROFILE and state['formal_world_controls_before_job']==BASE and state['maximum_world_controls']==LIMIT
    assert state['new_optimizer_updates']==state['new_ppo_steps']==state['new_supervised_updates']==0
    assert not state['learned_controller'] and not state['assistance'] and not state['promotion'] and state['policy_adoption']=='none'
    assert not (root/'native-errors.log').read_text().strip()
    assert verify_prerequisites()==state['prerequisites']
    previous=Path('docs/results/striking-pilot/demo-motion-2080768/analysis.json')
    assert hashlib.sha256(previous.read_bytes()).hexdigest()==state['source_demo_review_sha256']
    expected={(c,s,w) for c in (0,1) for s in SEEDS for w in (False,True)}
    assert {(r['corner'],r['seed'],r['swap']) for r in state['trials']}==expected
    assert sum(r['controls'] for r in state['trials'])==progress['diagnostic_world_controls']==state['diagnostic_world_controls']<=LIMIT
    acting=[];phases={};rows=[]
    for row in state['trials']:
        path=root/'native'/Path(row['native_trace_file']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['native_trace_sha256']
        with np.load(path) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            np.testing.assert_allclose(np.diff(arrays['time_before']),.002,atol=1e-12,rtol=0)
            np.testing.assert_allclose(arrays['time_after'][-1],row['elapsed'],atol=1e-12,rtol=0)
            assert len(arrays['time_after'])<=row['controls']*10 and len(arrays['time_after'])>(row['controls']-1)*10
            measured=describe_events(arrays)
        assert len(measured)==sum(row['scored_by_corner'])==len(row['events'])
        for e,saved in zip(measured,row['events']):
            assert dict(e,jab_cycle_candidate=candidate(e,row['corner']))==saved
            if e['corner']==row['corner'] and e['hand']==0:
                acting.append(e)
                start=row['phase_controls'].get('approach',0)*.02
                phase=cycle_fraction(max(0,e['onset_time']-start))[0]
                phases[phase]=phases.get(phase,0)+1
        candidates=sum(candidate(e,row['corner']) for e in measured)
        assert candidates==row['candidate_events']
        assert row['feasible_episode']==(row['survived'] and row['controls']==500 and candidates>=2)
        assert sum(row['phase_controls'].values())==row['controls']
        rows.append({k:row[k] for k in ('corner','seed','swap','survived','elapsed','controls','candidate_events','feasible_episode','phase_controls')})
    checks=dict(speed=lambda e:e['onset_relative_speed']>=1.,forward=lambda e:e['onset_forward_cosine']>=.7,
        travel=lambda e:e['pre_max_displacement_m']>=.12,
        complete_recovery=lambda e:e['post_window_complete'],
        return_distance=lambda e:e['post_window_complete'] and e['return_toward_pre_window_position_m']>=.06,
        contact_release=lambda e:e['post_contact_seconds']<=.10)
    result=dict(status='reviewed',profile=PROFILE,episodes=16,new_physical_controls=state['diagnostic_world_controls'],
        formal_world_controls_total=BASE+state['diagnostic_world_controls'],diagnostic_ppo_total=2163968,
        new_optimizer_updates=0,new_supervised_updates=0,learned_controller=False,main_adoption=False,promotion=False,
        survivors=sum(r['survived'] for r in rows),feasible_episodes=sum(r['feasible_episode'] for r in rows),
        candidate_events=sum(r['candidate_events'] for r in rows),acting_left_legacy_events=len(acting),
        reference_component_counts={k:sum(test(e) for e in acting) for k,test in checks.items()},
        inferred_command_phase_at_contact=phases,trials=rows,native_errors_empty=True,
        decision='Reject prototype v1 as a positive training target. No unchanged PPO/BC extension and no threshold relaxation. Before a bounded redesign, audit guard/strike workspace reachability inside the residual box and isolate first-fall/phase-tracking behavior; do not assume arm speed alone caused all falls.',
        limitations=['Command phase is inferred from approach-control count; it is not a measured force/action attribution.',
            'No full root/leg first-fall trace was saved in this arm-focused prototype; the physical cause of all falls is not established.',
            'One prototype does not prove the original authority or frozen motor can never support a punch.',
            'No learner was optimized and no positive strike-cycle target is certified.'])
    archive=Path('docs/results/striking-pilot/jab-cycle-v1');archive.mkdir(parents=True,exist_ok=True)
    shutil.copytree(root/'native',archive/'native',dirs_exist_ok=True)
    for filename in ('audit.json','progress.json','native-errors.log'):shutil.copy2(root/filename,archive/filename)
    atomic_json(archive/'analysis.json',result);atomic_json(root/'analysis.json',result)
    for folder in (archive,root):
        if not (folder/'completed-before-review.json').exists():atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        reviewed=dict(state,phase='independently_reviewed',reviewed_utc=utc(),continuation_decision=result['decision'])
        atomic_json(folder/'audit.json',reviewed)
    print(json.dumps({k:result[k] for k in ('status','survivors','feasible_episodes','new_physical_controls','formal_world_controls_total')}))


if __name__=='__main__':main()
