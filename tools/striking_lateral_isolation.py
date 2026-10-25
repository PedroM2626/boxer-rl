"""Isolate positive lateral command stability without scripted arm requests."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU comparison required')
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,utc
from boxer.striking_env import StrikingEnv
from boxer.striking_trace import StrikingTrace,terminal_details
from boxer.striking_punch_trace import PunchTrace,describe_events
from boxer.jab_feasibility import MotorOnlySideStep,PROFILE
from boxer.jab_workspace import MotorTap,controller_measurements
from tools.striking_pool_compare import verify_prerequisites
from tools.striking_jab_cycle_audit import candidate

BASE=2066307
LIMIT=4000
SEEDS=(7,23)
AMPLITUDES={'motor_only_positive':1}
MODES=tuple(AMPLITUDES)
REFERENCE=Path('docs/results/striking-pilot/lateral-command-v5')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            handle=stack.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'audit.json'
        if path.exists():raise ValueError('existing comparison requires consumed-state review before recovery')
        errors=root/'native-errors.log';stream=stack.enter_context(errors.open('w'))
        original=os.dup(2);stack.callback(os.close,original);stack.callback(os.dup2,original,2);os.dup2(stream.fileno(),2)
        previous=json.loads((REFERENCE/'analysis.json').read_text());source=json.loads((REFERENCE/'audit.json').read_text())
        assert previous['status']=='reviewed' and previous['formal_world_controls_total']==BASE
        assert source['profile']==PROFILE and previous['cloned_forward_evaluations_total']==6349
        surface_review=Path('docs/results/striking-pilot/lateral-command-v5/analysis.json')
        signed=json.loads(surface_review.read_text());assert signed['formal_world_controls_total']==BASE and signed['status']=='reviewed' and signed['cloned_forward_evaluations_total']==6349 and signed['surface_distance_queries_total']==960
        prerequisites=verify_prerequisites()
        foundation=UnassistedG1('approach',impl='jax');params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
        assert motor['sha256']==source['motor_sha256']
        env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45)
        env.motor=MotorTap(env.motor)
        state=dict(status='running',phase='lateral_isolation',controller_design_version=6,terminal_dwell_seconds=.12,pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(root),profile=PROFILE,prerequisites=prerequisites,motor_sha256=motor['sha256'],
            source_accounting_review_sha256=sha(surface_review),source_review_sha256=sha(REFERENCE/'analysis.json'),source_audit_sha256=sha(REFERENCE/'audit.json'),
            formal_world_controls_before_job=BASE,maximum_world_controls=LIMIT,formal_world_controls_ceiling=BASE+LIMIT,
            diagnostic_world_controls=0,new_ppo_steps=0,new_supervised_updates=0,new_optimizer_updates=0,
            diagnostic_ppo_retained=2163968,main_project_consumed_retained=476041216,main_boxing_retained=1048576,
            cloned_forward_evaluations_retained=6349,surface_distance_queries_retained=960,learned_controller=False,assistance=False,promotion=False,
            policy_adoption='none',trials=[])
        atomic_json(path,state);count=0
        try:
            for corner in (0,1):
                for seed in SEEDS:
                    for swap in (False,True):
                        reference=next(r for r in source['trials'] if r['mode']=='lateral_plus' and (r['corner'],r['seed'],r['swap'])==(corner,seed,swap))
                        reference_file=REFERENCE/'full'/Path(reference['full_trace_file']).name
                        assert sha(reference_file)==reference['full_trace_sha256']
                        with np.load(reference_file) as archive:
                            prefix=int(np.flatnonzero(archive['jab_active'])[0])
                            expected_before=archive['qpos_before'][:prefix].copy();expected_after=archive['qpos'][:prefix].copy()
                        for mode in MODES:
                            case=f'{mode}-corner-{corner}-seed-{seed}-swap-{int(swap)}'
                            obs=env.reset(seed,swap);controller=MotorOnlySideStep(corner,AMPLITUDES[mode])
                            native=PunchTrace();env.pair.diagnostic_punch_trace=native;full=StrikingTrace(env)
                            hits=np.zeros(2);phases={}
                            state.update(active_case=case,updated_utc=utc());atomic_json(path,state)
                            for step in range(500):
                                action,phase=controller.action(env)
                                telemetry=controller_measurements(env,controller,action,'approach',env.motor.last)
                                assert np.isfinite(action).all() and np.max(np.abs(action))<=1
                                full.before(env,obs,action)
                                full.pending.update(qpos_before=env.pair.data.qpos.copy(),qvel_before=env.pair.data.qvel.copy(),
                                    **{'jab_'+k:np.asarray(v) for k,v in telemetry.items()})
                                site=env.pair.data.site(env.pair.maps[corner]['side']+'imu_in_torso')
                                clock=0. if controller.started is None else env.pair.elapsed-controller.started
                                full.pending.update(base_commands=env.pair.approach_commands(stop_distance=.45).copy(),lateral_action=np.array(AMPLITUDES[mode]),cycle_entry_local=np.zeros(3) if controller.entry_local is None else controller.entry_local.copy(),
                                    terminal_dwell=np.array(phase=='extend' and clock%controller.cycle_duration()>=PROFILE['guard_seconds']+PROFILE['extend_seconds']),
                                    cycle_elapsed=np.array(clock),cycle_phase=np.array({'approach':0,'guard':1,'extend':2,'retract':3,'rest':4}[phase]),
                                    torso_origin_before=site.xpos.copy(),torso_rotation_before=site.xmat.reshape(3,3).copy())
                                phases[phase]=phases.get(phase,0)+1
                                if step<prefix:np.testing.assert_array_equal(env.pair.data.qpos,expected_before[step])
                                count+=1;assert count<=LIMIT
                                atomic_json(root/'progress.json',dict(diagnostic_world_controls=count,active_case=case,
                                    attempted_control=step+1,new_optimizer_updates=0,updated_utc=utc()))
                                obs,_,done,outcome=env.step(action);hits+=outcome['scored'];full.after(env,obs,outcome)
                                assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)
                                full.rows[-1]['external_force_norm']=np.array(0.)
                                if step<prefix:np.testing.assert_array_equal(env.pair.data.qpos,expected_after[step])
                                if done:break
                            assert step+1>=prefix
                            full_path=root/'full'/f'{case}.npz';contacts=full.save(full_path,env)
                            native_path=root/'native'/f'{case}.npz';native.save(native_path)
                            events=[dict(e,jab_cycle_candidate=candidate(e,corner)) for e in describe_events(native.arrays())]
                            n_candidates=sum(e['jab_cycle_candidate'] for e in events)
                            state['trials'].append(dict(case=case,mode=mode,lateral_action=AMPLITUDES[mode],corner=corner,seed=seed,swap=swap,
                                controls=step+1,elapsed=env.pair.elapsed,survived=not env.pair.done,
                                scores=hits.tolist(),terminal=terminal_details(env),events=events,candidate_events=n_candidates,feasible_episode=not env.pair.done and step==499 and n_candidates>=2,
                                phase_controls=phases,first_contacts=contacts,matched_prefix_controls=prefix,matched_prefix_kind='first_arm_request_removed',
                                reference_full_trace_sha256=reference['full_trace_sha256'],
                                full_trace_file=str(full_path),full_trace_sha256=sha(full_path),
                                native_trace_file=str(native_path),native_trace_sha256=sha(native_path)))
                            state['diagnostic_world_controls']=count;atomic_json(path,state)
            assert len(state['trials'])==len(MODES)*2*len(SEEDS)*2 and verify_prerequisites()==prerequisites
            stream.flush();assert not errors.read_text().strip(),'Unexpected native stderr'
            state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc(),formal_world_controls_total=BASE+count)
            atomic_json(path,state);print(json.dumps(dict(status=state['status'],controls=count)))
        except BaseException as error:
            state.update(status='failed',error=str(error),diagnostic_world_controls=count,updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
