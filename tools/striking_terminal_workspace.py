"""Bounded cloned terminal geometry, with independent no-physics review."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU static workspace audit required')
import mujoco
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import PairedFoundation
from boxer.jab_workspace import GeometryBudget,static_target_search
from tools.striking_terminal_dwell import sha
from tools.striking_pool_compare import verify_prerequisites

REFERENCE=Path('docs/results/striking-pilot/terminal-dwell-v3')
BASE=4339
LIMIT=6000


def selected(a):
    mask=a['terminal_dwell'].astype(bool);indices=np.flatnonzero(mask)
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
    return sorted({int(ends[0]),int(indices[np.argmin(a['jab_target_error_m'][indices])]),int(ends[-1])})


class DurableBudget(GeometryBudget):
    def __init__(self,root):super().__init__(LIMIT-24);self.root=root
    def consume(self):
        super().consume()
        atomic_json(self.root/'progress.json',dict(cloned_forward_evaluations=self.used,new_physical_controls=0,updated_utc=utc()))


def collect(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            handle=stack.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'audit.json'
        if path.exists():raise ValueError('existing workspace requires consumed-state review before recovery')
        stream=stack.enter_context((root/'native-errors.log').open('w'));original=os.dup(2)
        stack.callback(os.close,original);stack.callback(os.dup2,original,2);os.dup2(stream.fileno(),2)
        source=json.loads((REFERENCE/'audit.json').read_text());review=json.loads((REFERENCE/'analysis.json').read_text())
        assert source['phase']=='independently_reviewed' and review['status']=='reviewed'
        assert review['formal_world_controls_total']==2051072 and review['cloned_forward_evaluations_total']==BASE
        prerequisites=verify_prerequisites();pair=PairedFoundation(UnassistedG1('approach',impl='jax'))
        state=dict(status='running',phase='cloned_geometry',pid=os.getpid(),diagnostic_output=str(root),started_utc=utc(),
            source_audit_sha256=sha(REFERENCE/'audit.json'),source_review_sha256=sha(REFERENCE/'analysis.json'),
            prerequisites=prerequisites,new_physical_controls=0,new_optimizer_updates=0,formal_world_controls_total=2051072,
            diagnostic_ppo_total=2163968,cloned_forward_evaluations_before_job=BASE,maximum_new_cloned_evaluations=LIMIT,
            cumulative_cloned_ceiling=BASE+LIMIT,cloned_forward_evaluations=0,workspace=[])
        atomic_json(path,state);budget=DurableBudget(root)
        try:
            for row in source['trials']:
                full=REFERENCE/'full'/Path(row['full_trace_file']).name;assert sha(full)==row['full_trace_sha256']
                with np.load(full) as a:
                    for index in selected(a):
                        snapshot=dict(case=row['case'],corner=row['corner'],control_index=index,source_full_sha256=sha(full),
                            qpos=a['qpos_before'][index].tolist(),baseline=a['jab_baseline_target'][index].tolist(),
                            actual_error_m=float(a['jab_target_error_m'][index]),root_distance_m=float(a['root_distance_before'][index]))
                        result=static_target_search(pair,row['corner'],np.asarray(snapshot['qpos']),np.asarray(snapshot['baseline']),
                            a['jab_strike_world'][index],budget)
                        state['workspace'].append(dict(snapshot,**result));state['cloned_forward_evaluations']=budget.used
                        atomic_json(path,state)
            assert 8<=len(state['workspace'])<=24 and verify_prerequisites()==prerequisites
            stream.flush();assert not (root/'native-errors.log').read_text().strip()
            state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc());atomic_json(path,state)
            print(json.dumps(dict(status='completed',queries=len(state['workspace']),cloned_evaluations=budget.used)))
        except BaseException as error:
            state.update(status='failed',error=str(error),cloned_forward_evaluations=budget.used,updated_utc=utc());atomic_json(path,state);raise


def review(root):
    state=json.loads((root/'audit.json').read_text())
    assert state['status']=='completed' and state['phase']=='awaiting_interpretation'
    if (root/'review-progress.json').exists():raise ValueError('review compute exists; audit before recovery')
    assert not (root/'native-errors.log').read_text().strip()
    assert sha(REFERENCE/'audit.json')==state['source_audit_sha256'] and sha(REFERENCE/'analysis.json')==state['source_review_sha256']
    assert state['new_physical_controls']==state['new_optimizer_updates']==0 and state['formal_world_controls_total']==2051072
    assert state['cloned_forward_evaluations_before_job']==BASE and state['maximum_new_cloned_evaluations']==LIMIT
    assert state['cumulative_cloned_ceiling']==BASE+LIMIT
    assert state['cloned_forward_evaluations']==json.loads((root/'progress.json').read_text())['cloned_forward_evaluations']
    assert sum(w['cloned_forward_evaluations'] for w in state['workspace'])==state['cloned_forward_evaluations']
    assert verify_prerequisites()==state['prerequisites']
    pair=PairedFoundation(UnassistedG1('approach',impl='jax'));source=json.loads((REFERENCE/'audit.json').read_text())
    expected=set();count=0
    arm=np.array([i for i in range(29) if 'left_' in pair.foundation.mj_model.actuator(i).name
        and any(s in pair.foundation.mj_model.actuator(i).name for s in ('shoulder','elbow','wrist'))]);assert len(arm)==7
    for row in source['trials']:
        full=REFERENCE/'full'/Path(row['full_trace_file']).name;assert sha(full)==row['full_trace_sha256']
        with np.load(full) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            expected.update((row['case'],i) for i in selected(a))
    assert len(state['workspace'])==len(expected)<=24 and {(w['case'],w['control_index']) for w in state['workspace']}==expected
    for w in state['workspace']:
        row=next(r for r in source['trials'] if r['case']==w['case']);full=REFERENCE/'full'/Path(row['full_trace_file']).name
        assert w['corner']==row['corner'] and w['source_full_sha256']==row['full_trace_sha256']
        with np.load(full) as a:
            i=w['control_index'];assert a['terminal_dwell'][i] and a['jab_fraction'][i]==1.
            np.testing.assert_array_equal(w['qpos'],a['qpos_before'][i]);np.testing.assert_array_equal(w['baseline'],a['jab_baseline_target'][i])
            np.testing.assert_array_equal(w['target_world'],a['jab_strike_world'][i]);assert w['actual_error_m']==a['jab_target_error_m'][i]
            assert w['root_distance_m']==a['root_distance_before'][i]
        c=w['corner'];q=pair.maps[c]['q'][arm];joints=pair.model.actuator_trnid[pair.maps[c]['act'][arm],0]
        np.testing.assert_array_equal(w['joint_coordinates'],q)
        lower=np.maximum(np.asarray(w['baseline'])-.8,pair.model.jnt_range[joints,0]);upper=np.minimum(np.asarray(w['baseline'])+.8,pair.model.jnt_range[joints,1])
        np.testing.assert_array_equal(w['lower'],lower);np.testing.assert_array_equal(w['upper'],upper)
        x=np.asarray(w['best_targets']);assert np.all(x>=lower) and np.all(x<=upper)
        assert not w['physical_controls'] and not w['global_infeasibility_proven'] and not w['dynamic_feasibility_proven']
        count+=1;assert state['cloned_forward_evaluations']+count<=LIMIT
        atomic_json(root/'review-progress.json',dict(cloned_forward_evaluations=count,new_physical_controls=0,updated_utc=utc()))
        clone=mujoco.MjData(pair.model);clone.qpos[:]=w['qpos'];clone.qpos[q]=x
        mujoco.mj_kinematics(pair.model,clone);mujoco.mj_comPos(pair.model,clone)
        pos=clone.geom_xpos[pair.fists[c,0]];assert np.isfinite(pos).all()
        np.testing.assert_allclose(pos,w['best_fist_world'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(pos-np.asarray(w['target_world'])),w['best_error_m'],atol=1e-12,rtol=0)
    actual=np.array([w['actual_error_m'] for w in state['workspace']]);best=np.array([w['best_error_m'] for w in state['workspace']])
    result=dict(status='reviewed',queries=count,new_physical_controls=0,new_optimizer_updates=0,formal_world_controls_total=2051072,
        diagnostic_ppo_total=2163968,producer_cloned_evaluations=state['cloned_forward_evaluations'],review_cloned_evaluations=count,
        cloned_forward_evaluations_total=BASE+state['cloned_forward_evaluations']+count,
        actual_terminal_error_m=dict(minimum=float(actual.min()),median=float(np.median(actual)),maximum=float(actual.max())),
        best_static_error_m=dict(minimum=float(best.min()),median=float(np.median(best)),maximum=float(best.max())),
        limitations=['Two local searches do not prove global impossibility.',
            'Cloned static endpoints do not prove collision-safe dynamic reachability or balance.'],
        promotion=False,main_adoption=False,decision='Interpret terminal workspace before another bounded authority/target/controller intervention; no automatic learning extension.')
    archive=Path('docs/results/striking-pilot/terminal-workspace-v1');archive.mkdir(parents=True,exist_ok=True)
    for name in ('audit.json','progress.json','review-progress.json','native-errors.log'):shutil.copy2(root/name,archive/name)
    for folder in (archive,root):
        atomic_json(folder/'analysis.json',result);atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,phase='independently_reviewed',reviewed_utc=utc(),review_cloned_evaluations=count))
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--review',action='store_true')
    args=parser.parse_args();(review if args.review else collect)(args.output)


if __name__=='__main__':main()
