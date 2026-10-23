"""Bounded single-learner practice; neutral partner has fully active free physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('checkpoints/striking-pilot/step-1048576'))
    parser.add_argument('--steps', type=int, default=131072)
    parser.add_argument('--train-child', action='store_true')
    parser.add_argument('--ceiling', type=int)
    args = parser.parse_args()
    if args.train_child:
        import boxer.striking_train as training
        from boxer.striking_partner import PartnerStrikingEnv
        training.StrikingEnv = PartnerStrikingEnv
        training.main(['--output', str(args.output), '--resume', '--ceiling', str(args.ceiling),
                       '--worlds', '4', '--unroll', '128'])
        return
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.output/'partner.json'
        if state_path.exists():
            raise ValueError('partner diagnostic exists; inspect it before another launch')
        if args.steps <= 0 or args.steps % 512:
            raise ValueError('complete 512-transition PPO blocks required')
        manifest = json.loads((args.source/'latest.json').read_text())
        payload = (args.source/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
            raise ValueError('source hash mismatch')
        config = json.loads((args.source/'config.json').read_text())
        if (config['actor_inputs'], config['actions']) != (146,17):
            raise ValueError('original unmodified striking interface required')
        offset = manifest['consumed_steps']
        saved = pickle.loads(payload)
        world_offset = saved.get('world_control_steps', offset//2)
        state = dict(status='running', phase='source_evaluation', pid=os.getpid(), started_utc=utc(),
            diagnostic_output=str(args.output), source_sha256=manifest['sha256'],
            source_offset=offset, source_world_offset=world_offset, diagnostic_ceiling=args.steps,
            diagnostic_consumed_steps=0, diagnostic_evaluation_world_controls=0,
            assistance=False, learners_per_world=1, opponent_mode='neutral_motor',
            main_consumed_retained=1048576, policy_adoption='none', evaluations={})
        atomic_json(state_path,state)
        def evaluate(name, checkpoint, corner=None, sampled=False):
            command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                     '--output',str(args.output/(name+'.json'))]
            if corner is not None:command += ['--neutral-partner',str(corner)]
            if sampled:command.append('--sampled')
            state.update(active_evaluation=name,updated_utc=utc());atomic_json(state_path,state)
            with (args.output/(name+'.log')).open('w') as out, (args.output/(name+'-errors.log')).open('w') as err:
                subprocess.run(command,stdout=out,stderr=err,check=True)
            report=json.loads((args.output/(name+'.json')).read_text())
            rows=report['trials']
            if (len(rows)!=8 or {(x['seed'],x['swap']) for x in rows} !=
                    {(s,w) for s in (7,23,31,47) for w in (False,True)} or not all(x['finite'] for x in rows)):
                raise ValueError('incomplete physical evaluation')
            state['diagnostic_evaluation_world_controls']+=report['simulated_world_transitions']
            state['evaluations'][name]=dict(pair_survivors=sum(x['survived'] for x in rows),
                complete_episode_impacts=sum(sum(x['scored_by_corner']) for x in rows if x['survived']),
                learner_corner=corner,source_sha256=report['striking_sha256'])
            atomic_json(state_path,state)
        folder=args.output/'learner'; checkpoint=folder/'checkpoint'
        try:
            for corner in (0,1):evaluate('source-neutral-'+str(corner),args.source,corner)
            checkpoint.mkdir(parents=True,exist_ok=False)
            (checkpoint/manifest['file']).write_bytes(payload)
            atomic_json(checkpoint/'latest.json',manifest)
            atomic_json(folder/'config.json',dict(config,learners_per_world=1,opponent_mode='neutral_motor'))
            atomic_json(folder/'progress.json',dict(consumed_steps=offset,world_control_steps=world_offset))
            atomic_json(folder/'diagnostic-profile.json',dict(source_sha256=manifest['sha256'],
                source_offset=offset,source_world_offset=world_offset,main_training=False,
                no_parameter_migration=True,reward='unchanged original',assistance=False,
                learner_role='seed parity; balanced corners',opponent='zero high-level commands, reactive motor'))
            state.update(phase='training',updated_utc=utc());atomic_json(state_path,state)
            with (folder/'training.log').open('w') as out,(folder/'errors.log').open('w') as err:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'--output',str(folder),
                    '--train-child','--ceiling',str(offset+args.steps)],stdout=out,stderr=err,check=True)
            progress=json.loads((folder/'progress.json').read_text())
            delta=progress['consumed_steps']-offset
            if (delta!=args.steps or progress['optimized_steps']!=offset+args.steps or
                    progress['world_control_steps']-world_offset!=delta):
                raise ValueError('single-learner/physical counters disagree')
            state.update(phase='evaluation',diagnostic_consumed_steps=delta,
                         diagnostic_training_world_controls=delta,updated_utc=utc())
            atomic_json(state_path,state)
            for corner in (0,1):
                evaluate('trained-neutral-'+str(corner),checkpoint,corner)
                evaluate('sampled-neutral-'+str(corner),checkpoint,corner,True)
            evaluate('trained-selfplay',checkpoint)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc())
            atomic_json(state_path,state)
        except BaseException as error:
            p=folder/'progress.json'
            if p.exists():
                progress=json.loads(p.read_text())
                state.update(diagnostic_consumed_steps=max(0,progress['consumed_steps']-offset),
                    diagnostic_training_world_controls=max(0,progress['world_control_steps']-world_offset))
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(state_path,state)
            raise


if __name__=='__main__':main()
