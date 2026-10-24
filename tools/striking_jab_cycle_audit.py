"""Bounded explicit jab-cycle prototype, not learned-policy training."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU prototype required')
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,utc
from boxer.striking_env import StrikingEnv
from boxer.striking_punch_trace import PunchTrace,describe_events
from boxer.jab_feasibility import JabFeasibility,PROFILE
from tools.striking_pool_compare import verify_prerequisites

BASE=2026879
LIMIT=8000
SEEDS=(7,23,31,47)


def candidate(e,corner):
    return (e['corner']==corner and e['hand']==0 and e['onset_relative_speed']>=1.
        and e['onset_forward_cosine']>=.7 and e['pre_max_displacement_m']>=.12
        and e['post_window_complete'] and e['return_toward_pre_window_position_m'] is not None
        and e['return_toward_pre_window_position_m']>=.06 and e['post_contact_seconds']<=.10)


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in (args.output/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(p.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists():raise ValueError('existing jab prototype requires accounting review before recovery')
        errors=args.output/'native-errors.log';stream=stack.enter_context(errors.open('w'))
        original_fd=os.dup(2);stack.callback(os.close,original_fd);stack.callback(os.dup2,original_fd,2);os.dup2(stream.fileno(),2)
        previous=Path('docs/results/striking-pilot/demo-motion-2080768/analysis.json');review=json.loads(previous.read_text())
        assert review['status']=='reviewed' and review['formal_world_controls_total']==BASE and review['learner_both_reference_events']==0
        prerequisites=verify_prerequisites()
        foundation=UnassistedG1('approach',impl='jax');params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
        env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45)
        assert env.residual_scale==PROFILE['arm_residual_radians']
        state=dict(status='running',phase='physical_feasibility',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),profile=PROFILE,prerequisites=prerequisites,motor_sha256=motor['sha256'],
            source_demo_review_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
            maximum_world_controls=LIMIT,formal_world_controls_before_job=BASE,formal_world_controls_ceiling=BASE+LIMIT,
            diagnostic_world_controls=0,new_optimizer_updates=0,new_ppo_steps=0,new_supervised_updates=0,
            diagnostic_ppo_retained=2163968,main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            learned_controller=False,assistance=False,promotion=False,policy_adoption='none',trials=[])
        atomic_json(path,state);count=0
        try:
            for corner in (0,1):
                for seed in SEEDS:
                    for swap in (False,True):
                        env.reset(seed,swap);trace=PunchTrace();env.pair.diagnostic_punch_trace=trace
                        controller=JabFeasibility(corner);hits=np.zeros(2);phases={}
                        state.update(active_corner=corner,active_seed=seed,active_swap=swap,updated_utc=utc());atomic_json(path,state)
                        for step in range(500):
                            action,phase=controller.action(env);assert np.isfinite(action).all() and np.max(np.abs(action))<=1.
                            phases[phase]=phases.get(phase,0)+1;count+=1;assert count<=LIMIT
                            atomic_json(args.output/'progress.json',dict(diagnostic_world_controls=count,corner=corner,seed=seed,
                                swap=swap,attempted_control=step+1,new_optimizer_updates=0,updated_utc=utc()))
                            _,_,done,outcome=env.step(action);hits+=outcome['scored']
                            if done:break
                        file=args.output/'native'/f'corner-{corner}-seed-{seed}-swap-{int(swap)}.npz';trace.save(file)
                        events=describe_events(trace.arrays());assert len(events)==int(hits.sum())
                        rows=[dict(e,jab_cycle_candidate=candidate(e,corner)) for e in events]
                        n=sum(e['jab_cycle_candidate'] for e in rows)
                        row=dict(corner=corner,seed=seed,swap=swap,survived=not env.pair.done,elapsed=env.pair.elapsed,
                            controls=step+1,scored_by_corner=hits.tolist(),candidate_events=n,
                            feasible_episode=not env.pair.done and step==499 and n>=2,
                            phase_controls=phases,events=rows,native_trace_file=str(file),
                            native_trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                        state['trials'].append(row);state['diagnostic_world_controls']=count;atomic_json(path,state)
            assert len(state['trials'])==16 and verify_prerequisites()==prerequisites
            stream.flush();assert not errors.read_text().strip(),'Unexpected native stderr'
            state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc(),
                formal_world_controls_total=BASE+count,feasible_episodes=sum(r['feasible_episode'] for r in state['trials']),
                candidate_events=sum(r['candidate_events'] for r in state['trials']))
            atomic_json(path,state)
            print(json.dumps({k:state[k] for k in ('status','diagnostic_world_controls','feasible_episodes','candidate_events')}))
        except BaseException as error:
            state.update(status='failed',error=str(error),diagnostic_world_controls=count,updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
