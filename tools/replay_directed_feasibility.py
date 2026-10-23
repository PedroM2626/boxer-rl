"""Verify saved diagnostic actions in unchanged physics; optionally render one."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.striking_env import StrikingEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    progress_path=args.output.with_suffix('.progress.json')
    if args.output.exists() or progress_path.exists():
        raise ValueError('replay result/progress exists; preserve attempted controls before recovery')
    source=json.loads((args.run/'feasibility.json').read_text())
    if source['status']!='completed':raise ValueError('complete diagnostic required')
    foundation=UnassistedG1('approach',impl='jax')
    params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
    if motor['sha256']!=source['motor_sha256']:raise ValueError('motor provenance mismatch')
    env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45)
    report=dict(updated_utc=utc(),optimization_transitions=0,diagnostic_world_controls=0,
        learned_controller=False,promotion_eligible=False,source_sha256=source['source_sha256'],trials=[])
    frames=[];times=[];video_events=[]
    for row in source['trials']:
        if 'diagnostic_trajectory' not in row:continue
        file=args.run/row['diagnostic_trajectory'];data=np.load(file)
        obs=env.reset(row['seed'],row['swap']);corner=row['learner_corner'];hits=np.zeros(2);events=[]
        if len(data['action'])!=500 or data['observation'].shape!=(500,146):
            raise ValueError('complete demonstration arrays required')
        for step,(expected,action) in enumerate(zip(data['observation'],data['action'])):
            np.testing.assert_array_equal(obs[corner],expected)
            paired=np.zeros((2,17),dtype=action.dtype);paired[corner]=action
            previous=env.pair.scorecard.last_score.copy()
            report['diagnostic_world_controls']+=1
            atomic_json(progress_path,dict(diagnostic_world_controls=report['diagnostic_world_controls'],
                active_trajectory=file.name,attempted_control=step+1,optimization_transitions=0,updated_utc=utc()))
            obs,_,done,outcome=env.step(paired)
            hits+=outcome['scored']
            if np.any(outcome['scored']):
                recent=env.pair.scorecard.last_score>previous
                events.append(dict(elapsed=env.pair.elapsed,scored_by_corner=outcome['scored'].tolist(),
                    hands=[dict(corner=i,hand=h,time=float(env.pair.scorecard.last_score[i,h]),
                        contact_window_impulse=float(env.pair.scorecard.impulse[i,h]))
                        for i in range(2) for h in range(2) if recent[i,h]]))
            if args.video and not report['trials'] and step%2==0:
                frames.append(env.pair.data.qpos.copy());times.append(env.pair.elapsed)
            if done and step!=499:raise ValueError('recorded complete trajectory fell or stopped early')
        if env.pair.done or env.pair.elapsed!=row['elapsed'] or hits.tolist()!=row['scored_by_corner'] or events!=row['impact_events']:
            raise ValueError('physical demonstration replay mismatch')
        if not report['trials']:video_events=events
        report['trials'].append(dict(trajectory=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            observations_match_exactly=True,events_match_exactly=True,survived=True,impact_events=events))
    if frames:
        from tools.evaluate_paired_foundation import render
        render(env.pair,frames,args.video,caption='Geometric feasibility | Not a trained boxer | Free bodies',
               frame_times=times,impact_events=video_events)
    atomic_json(args.output,report)


if __name__=='__main__':main()
