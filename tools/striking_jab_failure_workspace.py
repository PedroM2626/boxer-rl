"""Exact v1 first-fall replays plus separately bounded cloned geometry queries."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU failure/workspace audit required')
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,utc
from boxer.striking_env import StrikingEnv
from boxer.striking_trace import StrikingTrace,terminal_details
from boxer.striking_punch_trace import PunchTrace,describe_events
from boxer.jab_feasibility import JabFeasibility,PROFILE
from boxer.jab_workspace import MotorTap,controller_measurements,GeometryBudget,static_target_search
from tools.striking_jab_cycle_audit import candidate
from tools.striking_pool_compare import verify_prerequisites

BASE=2030927
LIMIT=4048
GEOMETRY_LIMIT=20000
REFERENCE=Path('docs/results/striking-pilot/jab-cycle-v1')


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in (args.output/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            handle=stack.enter_context(p.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists():raise ValueError('existing replay requires consumed-state review before recovery')
        errors=args.output/'native-errors.log';stream=stack.enter_context(errors.open('w'))
        original_fd=os.dup(2);stack.callback(os.close,original_fd);stack.callback(os.dup2,original_fd,2);os.dup2(stream.fileno(),2)
        review=json.loads((REFERENCE/'analysis.json').read_text());reference=json.loads((REFERENCE/'audit.json').read_text())
        assert review['status']=='reviewed' and review['formal_world_controls_total']==BASE and reference['profile']==PROFILE
        assert sum(r['controls'] for r in reference['trials'])==LIMIT and len(reference['trials'])==16
        prerequisites=verify_prerequisites()
        foundation=UnassistedG1('approach',impl='jax');params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
        assert motor['sha256']==reference['motor_sha256']
        env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45);env.motor=MotorTap(env.motor)
        budget=GeometryBudget(GEOMETRY_LIMIT)
        state=dict(status='running',phase='exact_physical_replay',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            reference_audit_sha256=hashlib.sha256((REFERENCE/'audit.json').read_bytes()).hexdigest(),
            reference_review_sha256=hashlib.sha256((REFERENCE/'analysis.json').read_bytes()).hexdigest(),
            prerequisites=prerequisites,profile=PROFILE,motor_sha256=motor['sha256'],
            formal_world_controls_before_job=BASE,maximum_world_controls=LIMIT,formal_world_controls_ceiling=BASE+LIMIT,
            diagnostic_world_controls=0,maximum_cloned_forward_evaluations=GEOMETRY_LIMIT,cloned_forward_evaluations=0,
            new_optimizer_updates=0,new_ppo_steps=0,new_supervised_updates=0,diagnostic_ppo_retained=2163968,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            assistance=False,learned_controller=False,policy_adoption='none',promotion=False,trials=[],workspace=[])
        atomic_json(path,state);count=0;snapshots=[]
        try:
            for previous in reference['trials']:
                corner=previous['corner'];seed=previous['seed'];swap=previous['swap']
                name=f'corner-{corner}-seed-{seed}-swap-{int(swap)}'
                obs=env.reset(seed,swap);controller=JabFeasibility(corner)
                punch=PunchTrace();env.pair.diagnostic_punch_trace=punch;full=StrikingTrace(env)
                hits=np.zeros(2);phases={};taken=set()
                state.update(active_case=name,updated_utc=utc());atomic_json(path,state)
                for step in range(previous['controls']):
                    action,phase=controller.action(env)
                    diagnostics=controller_measurements(env,controller,action,phase,env.motor.last)
                    phases[phase]=phases.get(phase,0)+1
                    full.before(env,obs,action)
                    full.pending.update(qpos_before=env.pair.data.qpos.copy(),qvel_before=env.pair.data.qvel.copy(),
                        **{'jab_'+k:np.asarray(v) for k,v in diagnostics.items()})
                    kind='guard' if phase=='guard' else 'extend_mid' if phase=='extend' and diagnostics['fraction']>=.5 else None
                    if kind is not None and kind not in taken:
                        snapshots.append(dict(case=name,corner=corner,kind=kind,control_index=step,
                            qpos=env.pair.data.qpos.copy().tolist(),baseline=diagnostics['baseline_target'].tolist(),
                            guard=diagnostics['guard_world'].tolist(),strike=diagnostics['strike_world'].tolist()))
                        taken.add(kind)
                    count+=1;assert count<=LIMIT
                    atomic_json(args.output/'progress.json',dict(diagnostic_world_controls=count,active_case=name,
                        attempted_control=step+1,cloned_forward_evaluations=budget.used,new_optimizer_updates=0,updated_utc=utc()))
                    obs,_,done,outcome=env.step(action);hits+=outcome['scored'];full.after(env,obs,outcome)
                    assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)
                    full.rows[-1]['external_force_norm']=np.array(0.)
                    if done:
                        assert step+1==previous['controls'];break
                assert env.pair.elapsed==previous['elapsed'] and (not env.pair.done)==previous['survived']
                assert hits.tolist()==previous['scored_by_corner'] and phases==previous['phase_controls']
                source=REFERENCE/'native'/Path(previous['native_trace_file']).name
                assert hashlib.sha256(source.read_bytes()).hexdigest()==previous['native_trace_sha256']
                arrays=punch.arrays()
                with np.load(source) as expected:
                    assert set(arrays)==set(expected.files)
                    for key in arrays:np.testing.assert_array_equal(arrays[key],expected[key])
                events=[dict(e,jab_cycle_candidate=candidate(e,corner)) for e in describe_events(arrays)]
                assert events==previous['events']
                output=args.output/'full'/f'{name}.npz';contacts=full.save(output,env)
                state['trials'].append(dict(case=name,corner=corner,seed=seed,swap=swap,controls=step+1,
                    elapsed=env.pair.elapsed,survived=not env.pair.done,all_original_native_arrays_equal=True,
                    scores=hits.tolist(),terminal=terminal_details(env),first_observed_contacts=contacts,
                    full_trace_file=str(output),full_trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    snapshot_kinds=sorted(taken)))
                state['diagnostic_world_controls']=count;atomic_json(path,state)
            assert count==LIMIT and len(snapshots)<=32
            atomic_json(args.output/'snapshots.json',snapshots)
            state.update(phase='cloned_geometry',updated_utc=utc());atomic_json(path,state)
            for snapshot in snapshots:
                for kind in ('guard','strike'):
                    result=static_target_search(env.pair,snapshot['corner'],np.asarray(snapshot['qpos']),
                        np.asarray(snapshot['baseline']),np.asarray(snapshot[kind]),budget)
                    state['workspace'].append(dict(case=snapshot['case'],snapshot_kind=snapshot['kind'],
                        control_index=snapshot['control_index'],target_kind=kind,**result))
                    state['cloned_forward_evaluations']=budget.used;atomic_json(path,state)
            assert verify_prerequisites()==prerequisites
            stream.flush();assert not errors.read_text().strip(),'Unexpected native stderr'
            state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc(),
                formal_world_controls_total=BASE+count,geometry_snapshot_count=len(snapshots))
            atomic_json(path,state);print(json.dumps(dict(status='completed',physical_controls=count,cloned_evaluations=budget.used)))
        except BaseException as error:
            state.update(status='failed',error=str(error),diagnostic_world_controls=count,
                cloned_forward_evaluations=budget.used,updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
