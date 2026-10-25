"""Measure saved-pose legal and blocking gaps without physical stepping."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU surface audit required')
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import PairedFoundation
from boxer.surface_audit import SurfaceBudget,pose_surfaces
from boxer.scene import UPPER_TARGET_GEOMS
from tools.striking_terminal_dwell import sha
from tools.striking_pool_compare import verify_prerequisites

SOURCE=Path('docs/results/striking-pilot/terminal-workspace-v1')
PHYSICAL=Path('docs/results/striking-pilot/terminal-dwell-v3')
BASE=6253
PER_FORWARDS=48
PER_DISTANCES=480


def sources():
    geometry=json.loads((SOURCE/'audit.json').read_text());analysis=json.loads((SOURCE/'analysis.json').read_text())
    physical=json.loads((PHYSICAL/'audit.json').read_text())
    assert geometry['phase']==physical['phase']=='independently_reviewed' and analysis['status']=='reviewed'
    assert analysis['cloned_forward_evaluations_total']==BASE and analysis['formal_world_controls_total']==2051072
    assert len(geometry['workspace'])==24
    assert geometry['source_audit_sha256']==sha(PHYSICAL/'audit.json')
    return geometry,physical


def source_pose(pair,w,kind,physical):
    row=next(r for r in physical['trials'] if r['case']==w['case']);full=PHYSICAL/'full'/Path(row['full_trace_file']).name
    assert sha(full)==row['full_trace_sha256']==w['source_full_sha256'] and row['corner']==w['corner']
    with np.load(full) as a:
        index=w['control_index'];np.testing.assert_array_equal(w['qpos'],a['qpos_before'][index])
        assert a['terminal_dwell'][index];time=float(a['time_before'][index]);qpos=a['qpos_before'][index].copy()
        np.testing.assert_array_equal(w['baseline'],a['jab_baseline_target'][index])
    c=w['corner'];arm=np.array([i for i in range(29) if 'left_' in pair.foundation.mj_model.actuator(i).name
        and any(s in pair.foundation.mj_model.actuator(i).name for s in ('shoulder','elbow','wrist'))]);assert len(arm)==7
    q=pair.maps[c]['q'][arm];np.testing.assert_array_equal(q,w['joint_coordinates'])
    joints=pair.model.actuator_trnid[pair.maps[c]['act'][arm],0]
    lower=np.maximum(np.asarray(w['baseline'])-.8,pair.model.jnt_range[joints,0]);upper=np.minimum(np.asarray(w['baseline'])+.8,pair.model.jnt_range[joints,1])
    np.testing.assert_array_equal(lower,w['lower']);np.testing.assert_array_equal(upper,w['upper'])
    if kind=='best':
        x=np.asarray(w['best_targets']);assert np.all(x>=lower) and np.all(x<=upper);qpos[q]=x
    else:assert kind=='recorded'
    assert np.isfinite(qpos).all()
    fist=pair.model.geom(int(pair.fists[c,0])).name;other=pair.maps[1-c]['side']
    blocking={other+name for name in UPPER_TARGET_GEOMS if name not in ('head_collision','torso_collision')}
    events=[dict(e,occurred_by_snapshot=e['elapsed']<=time) for e in row['first_contacts']
        if fist in e['geoms'] and any(name in blocking for name in e['geoms'])]
    return qpos,time,events


def budget_for(root,review=False):
    path=root/('review-progress.json' if review else 'progress.json')
    return SurfaceBudget(PER_FORWARDS,PER_DISTANCES,lambda counters:atomic_json(path,dict(counters,new_physical_controls=0,updated_utc=utc())))


def collect(root):
    geometry,physical=sources();prerequisites=verify_prerequisites()
    pair=PairedFoundation(UnassistedG1('approach',impl='jax'));budget=budget_for(root)
    state=dict(status='running',phase='signed_surface_queries',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(root),
        source_audit_sha256=sha(SOURCE/'audit.json'),source_analysis_sha256=sha(SOURCE/'analysis.json'),physical_audit_sha256=sha(PHYSICAL/'audit.json'),
        prerequisites=prerequisites,cloned_before_job=BASE,cloned_ceiling=BASE+2*PER_FORWARDS,
        maximum_new_cloned_forwards=2*PER_FORWARDS,maximum_surface_distance_queries=2*PER_DISTANCES,
        formal_world_controls_total=2051072,diagnostic_ppo_total=2163968,main_combined_total=476041216,standalone_bc_total=8000,
        new_physical_controls=0,new_optimizer_updates=0,producer_cloned_forwards=0,producer_surface_queries=0,poses=[])
    atomic_json(root/'audit.json',state)
    try:
        for w in geometry['workspace']:
            for kind in ('recorded','best'):
                qpos,time,events=source_pose(pair,w,kind,physical)
                surfaces=pose_surfaces(pair,qpos,w['corner'],budget)
                state['poses'].append(dict(case=w['case'],control_index=w['control_index'],corner=w['corner'],pose_kind=kind,
                    time=time,qpos_sha256=hashlib.sha256(qpos.tobytes()).hexdigest(),surfaces=surfaces,
                    physical_first_block_events=events))
                state.update(producer_cloned_forwards=budget.forwards,producer_surface_queries=budget.distances)
                atomic_json(root/'audit.json',state)
        assert budget.forwards==PER_FORWARDS and budget.distances==PER_DISTANCES and verify_prerequisites()==prerequisites
        state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc());atomic_json(root/'audit.json',state)
        print(json.dumps(dict(status='completed',cloned_forwards=budget.forwards,surface_queries=budget.distances)))
    except BaseException as error:
        state.update(status='failed',error=str(error),producer_cloned_forwards=budget.forwards,producer_surface_queries=budget.distances,updated_utc=utc())
        atomic_json(root/'audit.json',state);raise


def review(root):
    state=json.loads((root/'audit.json').read_text());geometry,physical=sources()
    assert state['status']=='completed' and state['phase']=='awaiting_interpretation' and len(state['poses'])==48
    assert state['source_audit_sha256']==sha(SOURCE/'audit.json') and state['source_analysis_sha256']==sha(SOURCE/'analysis.json')
    assert state['physical_audit_sha256']==sha(PHYSICAL/'audit.json') and verify_prerequisites()==state['prerequisites']
    assert state['cloned_before_job']==BASE and state['cloned_ceiling']==BASE+2*PER_FORWARDS
    assert state['maximum_new_cloned_forwards']==96 and state['maximum_surface_distance_queries']==960
    assert state['new_physical_controls']==state['new_optimizer_updates']==0
    assert state['formal_world_controls_total']==2051072 and state['main_combined_total']==476041216
    progress=json.loads((root/'progress.json').read_text())
    assert progress['cloned_forward_evaluations']==state['producer_cloned_forwards']==48
    assert progress['surface_distance_queries']==state['producer_surface_queries']==480
    assert not (root/'native-errors.log').read_text().strip()
    pair=PairedFoundation(UnassistedG1('approach',impl='jax'));budget=budget_for(root,True)
    expected={(w['case'],w['control_index'],kind) for w in geometry['workspace'] for kind in ('recorded','best')}
    assert {(p['case'],p['control_index'],p['pose_kind']) for p in state['poses']}==expected
    for p in state['poses']:
        w=next(w for w in geometry['workspace'] if (w['case'],w['control_index'])==(p['case'],p['control_index']))
        qpos,time,events=source_pose(pair,w,p['pose_kind'],physical)
        assert p['corner']==w['corner'] and p['time']==time and p['physical_first_block_events']==events
        assert p['qpos_sha256']==hashlib.sha256(qpos.tobytes()).hexdigest()
        measured=pose_surfaces(pair,qpos,p['corner'],budget);assert len(measured)==len(p['surfaces'])==10
        for a,b in zip(measured,p['surfaces']):
            assert a['target']==b['target'] and a['legal']==b['legal'] and a['collision_pair_declared']==b['collision_pair_declared']==True
            assert not a['censored'] and not b['censored']
            np.testing.assert_allclose(a['signed_gap_m'],b['signed_gap_m'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(a['nearest_segment'],b['nearest_segment'],atol=1e-12,rtol=0)
    groups={}
    for kind in ('recorded','best'):
        poses=[p for p in state['poses'] if p['pose_kind']==kind]
        legal=np.array([min(s['signed_gap_m'] for s in p['surfaces'] if s['legal']) for p in poses])
        block=np.array([min(s['signed_gap_m'] for s in p['surfaces'] if not s['legal']) for p in poses])
        groups[kind]=dict(poses=len(poses),minimum_legal_gap_m=dict(minimum=float(legal.min()),median=float(np.median(legal)),maximum=float(legal.max())),
            minimum_block_gap_m=dict(minimum=float(block.min()),median=float(np.median(block)),maximum=float(block.max())),
            legal_overlap_poses=int((legal<=0).sum()),blocking_overlap_poses=int((block<=0).sum()))
    result=dict(status='reviewed',groups=groups,new_physical_controls=0,new_optimizer_updates=0,
        formal_world_controls_total=2051072,diagnostic_ppo_total=2163968,main_combined_total=476041216,standalone_bc_total=8000,
        producer_cloned_forwards=48,review_cloned_forwards=budget.forwards,cloned_forward_evaluations_total=BASE+48+budget.forwards,
        producer_surface_queries=480,review_surface_queries=budget.distances,surface_distance_queries_total=480+budget.distances,
        promotion=False,main_adoption=False,
        limitations=['Saved local-best poses were optimized for a point, not for nearest collision-surface gap.',
            'Static overlap does not establish a balanced dynamic punch, contact impulse or collision-free path.',
            'An earlier physical block event does not prove it is active at a later snapshot.'],
        decision='Interpret legal surface gaps and blocking intersections before a bounded target/path/authority intervention; no automatic training extension.')
    assert budget.forwards==48 and budget.distances==480
    archive=Path('docs/results/striking-pilot/surface-audit-v1');archive.mkdir(parents=True,exist_ok=True)
    for name in ('audit.json','progress.json','review-progress.json','native-errors.log'):shutil.copy2(root/name,archive/name)
    for folder in (root,archive):
        atomic_json(folder/'analysis.json',result);atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,phase='independently_reviewed',reviewed_utc=utc(),review_cloned_forwards=48,review_surface_queries=480))
    print(json.dumps(result))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--review',action='store_true')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.review:
            if (root/'review-progress.json').exists():raise ValueError('review queries exist; preserve consumed accounting before recovery')
        elif (root/'audit.json').exists():raise ValueError('surface audit exists; inspect before recovery')
        errors=root/('review-native-errors.log' if args.review else 'native-errors.log')
        stream=stack.enter_context(errors.open('w'));original=os.dup(2)
        stack.callback(os.close,original);stack.callback(os.dup2,original,2);os.dup2(stream.fileno(),2)
        (review if args.review else collect)(root)
        stream.flush();assert not errors.read_text().strip(),'Unexpected native stderr'
        if args.review:shutil.copy2(errors,Path('docs/results/striking-pilot/surface-audit-v1')/errors.name)


if __name__=='__main__':main()
