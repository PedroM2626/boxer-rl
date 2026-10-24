"""Replay all retained actions exactly and measure versioned motion evidence."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU demonstration audit required')
import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,utc
from boxer.striking_env import StrikingEnv
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import PunchTrace,describe_events
from boxer.strike_motion import evaluate_motion,PROFILE
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

BASE=2019379
LIMIT=7500
DATA=Path('docs/results/striking-pilot/directed-holdout-1441792')
SOURCE=Path('checkpoints/striking-validation/partner-duration-1441792')
SOURCE_SHA='87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8'
CONTROL=Path('checkpoints/striking-validation/pool-2064384/control/gate-16384')
CONTROL_SHA='3a34aa85f0e9cf1ecb178c02f5fe945f8dd17f45dc1b323208b6083e2091b501'


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in (args.output/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            handle=stack.enter_context(p.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists():raise ValueError('existing demonstration audit requires accounting review before recovery')
        errors=args.output/'native-errors.log';native=stack.enter_context(errors.open('w'))
        original_fd=os.dup(2);stack.callback(os.close,original_fd)
        stack.callback(os.dup2,original_fd,2);os.dup2(native.fileno(),2)
        prior=json.loads(Path('docs/results/striking-pilot/punch-quality-2080768/analysis.json').read_text())
        assert prior['status']=='reviewed' and prior['formal_world_controls_total']==BASE and prior['diagnostic_ppo_total']==2163968
        prerequisites=verify_prerequisites()
        source,_=load_checked(SOURCE,SOURCE_SHA);require_ppo_checkpoint(source)
        control,_=load_checked(CONTROL,CONTROL_SHA);require_ppo_checkpoint(control)
        for saved in (source,control):
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
        replay=json.loads((DATA/'replay.json').read_text());feasibility=json.loads((DATA/'feasibility.json').read_text())
        assert replay['source_sha256']==feasibility['source_sha256']==SOURCE_SHA and len(replay['trials'])==15
        files={r['trajectory']:r['sha256'] for r in replay['trials']}
        assert files==control['retention_provenance']['demonstration_files']
        foundation=UnassistedG1('approach',impl='jax');params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
        assert motor['sha256']==feasibility['motor_sha256']==source['foundation_sha256']==control['foundation_sha256']
        env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45)
        state=dict(status='running',phase='native_replay',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            source_sha256=SOURCE_SHA,current_retention_checkpoint_sha256=CONTROL_SHA,demonstration_files=files,
            replay_manifest_sha256=hashlib.sha256((DATA/'replay.json').read_bytes()).hexdigest(),
            feasibility_manifest_sha256=hashlib.sha256((DATA/'feasibility.json').read_bytes()).hexdigest(),
            formal_world_controls_before_job=BASE,maximum_world_controls=LIMIT,formal_world_controls_ceiling=BASE+LIMIT,
            diagnostic_world_controls=0,new_optimizer_updates=0,new_ppo_steps=0,new_supervised_updates=0,
            diagnostic_ppo_retained=2163968,main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            assistance=False,policy_adoption='none',promotion=False,motion_profile=dict(PROFILE),
            prerequisites=prerequisites,trials=[])
        atomic_json(path,state);count=0
        try:
            for demo in replay['trials']:
                filename=demo['trajectory'];raw=(DATA/filename).read_bytes();assert hashlib.sha256(raw).hexdigest()==demo['sha256']
                original=next(r for r in feasibility['trials'] if r.get('diagnostic_trajectory')==filename)
                data=np.load(DATA/filename);assert data['observation'].shape==(500,146) and data['action'].shape==(500,17)
                assert np.isfinite(data['observation']).all() and np.isfinite(data['action']).all() and np.max(np.abs(data['action']))<=1
                assert demo['observations_match_exactly'] and demo['events_match_exactly'] and demo['survived']
                obs=env.reset(original['seed'],original['swap']);corner=original['learner_corner']
                trace=PunchTrace();env.pair.diagnostic_punch_trace=trace;hits=np.zeros(2);events=[]
                state.update(active_trajectory=filename,updated_utc=utc());atomic_json(path,state)
                for step,(expected,action) in enumerate(zip(data['observation'],data['action'])):
                    np.testing.assert_array_equal(obs[corner],expected)
                    paired=np.zeros((2,17),dtype=action.dtype);paired[corner]=action
                    previous=env.pair.scorecard.last_score.copy();count+=1
                    assert count<=LIMIT
                    atomic_json(args.output/'progress.json',dict(diagnostic_world_controls=count,active_trajectory=filename,
                        attempted_control=step+1,new_optimizer_updates=0,updated_utc=utc()))
                    obs,_,done,outcome=env.step(paired);hits+=outcome['scored']
                    if np.any(outcome['scored']):
                        recent=env.pair.scorecard.last_score>previous
                        events.append(dict(elapsed=env.pair.elapsed,scored_by_corner=outcome['scored'].tolist(),
                            hands=[dict(corner=i,hand=h,time=float(env.pair.scorecard.last_score[i,h]),
                                contact_window_impulse=float(env.pair.scorecard.impulse[i,h]))
                                for i in range(2) for h in range(2) if recent[i,h]]))
                    if done and step!=499:raise ValueError('complete demonstration stopped early')
                assert not env.pair.done and env.pair.elapsed==original['elapsed']
                assert hits.tolist()==original['scored_by_corner'] and events==demo['impact_events']==original['impact_events']
                output=args.output/'native'/filename;trace.save(output)
                measured=describe_events(trace.arrays());assert len(measured)==int(hits.sum())
                learner=[e for e in measured if e['corner']==corner]
                evidence=evaluate_motion(learner)
                state['trials'].append(dict(trajectory=filename,demonstration_sha256=demo['sha256'],
                    learner_corner=corner,seed=original['seed'],swap=original['swap'],survived=True,
                    observations_match_exactly=True,events_match_exactly=True,scored_by_corner=hits.tolist(),
                    impact_events=events,native_trace_file=str(output),native_trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    all_hand_measurements=measured,learner_motion_evidence=evidence))
                state['diagnostic_world_controls']=count;atomic_json(path,state)
            assert count==LIMIT and len(state['trials'])==15 and verify_prerequisites()==prerequisites
            native.flush();assert not errors.read_text().strip(),'Unexpected native stderr'
            state.update(status='completed',phase='awaiting_interpretation',updated_utc=utc(),diagnostic_world_controls=count,
                formal_world_controls_total=BASE+count,learner_legacy_events=sum(r['learner_motion_evidence']['legacy_event_count'] for r in state['trials']),
                learner_both_reference_events=sum(r['learner_motion_evidence']['both_reference_components_count'] for r in state['trials']))
            atomic_json(path,state)
            print(json.dumps({k:state[k] for k in ('status','diagnostic_world_controls','learner_legacy_events','learner_both_reference_events')}))
        except BaseException as error:
            state.update(status='failed',error=str(error),diagnostic_world_controls=count,updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
