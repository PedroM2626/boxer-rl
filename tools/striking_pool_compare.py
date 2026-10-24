"""Bounded matched archived-pool versus fixed-adversary continuation from bea8147f."""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import jax
import numpy as np

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_train import utc
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_fixed_opponent import fingerprint, FROZEN_FINGERPRINT
from boxer.striking_opponent_pool import (SOURCE, SOURCE_SHA, SPECS, OFFSET,
    WORLD_OFFSET, prepare_pool_state, pool_config, validate_pool_ledger)
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from tools.review_striking_credit_horizon import full_profiles

SEEDS = list(range(32000003, 32000007))
PPO_BEFORE = 2131200
EVALUATION_BEFORE = 1936816
MAX_PPO = 32768
MAX_EVALUATION = 84000


def verify_prerequisites():
    review = json.loads(Path('docs/results/striking-pilot/gae-generalization-2064384/analysis.json').read_text())
    assert review['status']=='reviewed' and review['source_sha256']==SOURCE_SHA
    assert review['diagnostic_ppo_total']==PPO_BEFORE and review['formal_world_controls_total']==EVALUATION_BEFORE
    result={}
    for stage in ('balance','walk_easy','walk','approach'):
        history=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert history==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text())
        assert promotion_ready(history,stage)
        folder=Path('checkpoints/curriculum')/stage
        manifest=json.loads((folder/'latest.json').read_text());data=(folder/manifest['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==manifest['sha256']
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result[stage]=dict(records=len(history),sha256=manifest['sha256'])
    _,main=load_checked(Path('checkpoints/striking-pilot/step-1048576'),
        'efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0')
    return dict(foundation=result,main_sha256=main['sha256'])


def verify_checkpoint(folder, target, pool):
    cp=folder/'checkpoint';manifest=json.loads((cp/'latest.json').read_text())
    saved,_=load_checked(cp,manifest['sha256']);require_ppo_checkpoint(saved)
    assert saved['consumed_steps']==saved['optimized_steps']==OFFSET+target
    assert saved['world_control_steps']==WORLD_OFFSET+target
    assert int(saved['optimizer']['actor_adam'][0].count)==14728+target//1024*16
    assert int(saved['optimizer']['critic_adam'][0].count)==38400+target//1024*16
    assert saved.get('pool_opponent_ppo',False)==pool
    assert fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
        tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
    np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1),
                                 [163840+target//2]*2)
    if pool:validate_pool_ledger(saved['pool_credits_by_opponent_and_corner'],saved['mixed_credits_by_mode_and_corner'])
    return saved,manifest


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for name in (args.output/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock')):
            handle=stack.enter_context(name.open('w'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Probe the established training namespace before any allocation, without
        # holding it across children which acquire this lock themselves.
        with Path('runs/unassisted_boxing.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('comparison exists; inspect consumed progress before recovery')
        prerequisites=verify_prerequisites()
        source,manifest=load_checked(Path(SOURCE),SOURCE_SHA);require_ppo_checkpoint(source)
        config=json.loads((Path(SOURCE)/'config.json').read_text())
        snapshots=[]
        for location,sha,_ in SPECS:
            archived,_=load_checked(Path(location),sha);require_ppo_checkpoint(archived)
            assert archived['foundation_sha256']==source['foundation_sha256']
            snapshots.append(dict(source_sha256=sha,actor=archived['actor'],norm=archived['norm']))
        state=dict(status='running',phase='initializing',pid=os.getpid(),started_utc=utc(),
            diagnostic_output=str(args.output),source_sha256=SOURCE_SHA,
            source_consumed_offset=OFFSET,source_world_offset=WORLD_OFFSET,
            maximum_new_ppo_steps=MAX_PPO,new_ppo_steps_per_arm=16384,
            maximum_evaluation_world_controls=MAX_EVALUATION,
            diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0,diagnostic_ppo_before_job=PPO_BEFORE,
            formal_evaluation_world_controls_before_job=EVALUATION_BEFORE,
            diagnostic_ppo_ceiling=PPO_BEFORE+MAX_PPO,
            formal_evaluation_world_controls_ceiling=EVALUATION_BEFORE+MAX_EVALUATION,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000,new_separate_supervised_updates=0,
            unchanged_physics=True,unchanged_rewards=True,on_policy_sampling=True,
            assistance=False,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,
            pool_original_complete_pass_streak=0,prerequisites=prerequisites,arms=[])
        atomic_json(path,state)

        def account():
            rows=[json.loads(p.read_text()) for p in (args.output/'pool/progress.json',args.output/'control/progress.json') if p.exists()]
            state['diagnostic_consumed_steps']=sum(p['consumed_steps']-OFFSET for p in rows)
            state['diagnostic_collection_world_controls']=sum(p['world_control_steps']-WORLD_OFFSET for p in rows)
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions']
                for p in args.output.rglob('*-progress.json'))
            assert 0<=state['diagnostic_consumed_steps']<=MAX_PPO
            assert 0<=state['diagnostic_evaluation_world_controls']<=MAX_EVALUATION

        def physical(checkpoint,destination,seeds,sha):
            evaluate(checkpoint,destination,seeds,state,path)
            assert all(not p.read_text().strip() for p in destination.glob('*errors.log'))
            full,_=full_profiles(destination,seeds,sha);account();atomic_json(path,state)
            return full

        def train(arm,target):
            name=arm['arm'];folder=args.output/name
            state.update(phase='training',active_arm=name,active_training_output=str(folder),updated_utc=utc())
            atomic_json(path,state)
            cmd=[sys.executable,'-u','-m','boxer.striking_train','--output',str(folder),'--resume',
                '--contact-ppo','--demonstration-ppo','--neutral-contact-ppo','--credit-horizon-ppo',
                '--mixed-partner-ppo','--fixed-opponent-ppo','--long-gae-ppo','--training-gamma','.999',
                '--retention-loss-weight','1','--ceiling',str(OFFSET+target),'--worlds','8','--unroll','128','--seed','53']
            if name=='pool':cmd+=['--pool-opponent-ppo']
            if target==1024:cmd+=['--update-audit-directory',str(folder/'update-audit')]
            with (folder/f'train-{target}.log').open('w') as out,(folder/f'train-{target}-errors.log').open('w') as err:
                subprocess.run(cmd,stdout=out,stderr=err,check=True)
            account();assert not (folder/f'train-{target}-errors.log').read_text().strip()
            saved,m=verify_checkpoint(folder,target,name=='pool')
            if target==1024:
                audit=json.loads((folder/'update-audit/minibatch-measurements.json').read_text())
                assert len(audit['records'])==18 and audit['records'][0]['profiles']['all']['ratio_clip_fraction']==0.
                arm.update(integration_block_verified=True,initial_update_geometry=audit['records'][-1],
                    initial_sampling_sha256=audit['sampling_sha256'])
                if name=='pool':arm['initial_pool_credits']=saved['pool_credits_by_opponent_and_corner']
            arm['latest_sha256']=m['sha256'];arm['new_ppo_steps']=target
            atomic_json(path,state);return m

        try:
            for name in ('pool','control'):
                folder=args.output/name;cp=folder/'checkpoint';cp.mkdir(parents=True,exist_ok=False)
                initial=prepare_pool_state(source,SOURCE_SHA,snapshots) if name=='pool' else copy.deepcopy(source)
                for key in source.keys()-{'opponent_mode'}:
                    for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(initial[key])):
                        np.testing.assert_equal(a,b)
                payload=pickle.dumps(initial,protocol=pickle.HIGHEST_PROTOCOL)
                atomic_bytes(cp/'initial.pkl',payload)
                extra=dict(pool_opponent_ppo=True,opponent_mode='mixed_pool_mean') if name=='pool' else {}
                atomic_json(cp/'latest.json',dict(manifest,file='initial.pkl',sha256=hashlib.sha256(payload).hexdigest(),**extra))
                atomic_json(folder/'config.json',pool_config(config) if name=='pool' else config)
                atomic_json(folder/'progress.json',dict(consumed_steps=OFFSET,optimized_steps=OFFSET,world_control_steps=WORLD_OFFSET,
                    neutral_credits_by_corner=source['neutral_credits_by_corner'],
                    mixed_credits_by_mode_and_corner=source['mixed_credits_by_mode_and_corner']))
                arm=dict(arm=name,initial_all_learned_source_leaves_retained=True,gates=[])
                state['arms'].append(arm);train(arm,1024)
            state.update(phase='source_fresh_evaluation',updated_utc=utc());atomic_json(path,state)
            state['source_fresh']=physical(Path(SOURCE),args.output/'source-fresh',SEEDS,SOURCE_SHA)
            for arm in state['arms']:
                folder=args.output/arm['arm'];cp=folder/'checkpoint'
                for target in (8192,16384):
                    m=train(arm,target)
                    gate=folder/f'gate-{target}'/'checkpoint';gate.mkdir(parents=True,exist_ok=False)
                    for filename in (m['file'],'latest.json'):shutil.copy2(cp/filename,gate/filename)
                    shutil.copy2(folder/'config.json',gate/'config.json')
                    state.update(phase='screen_evaluation',updated_utc=utc());atomic_json(path,state)
                    full=physical(gate,folder/f'gate-{target}'/'screen',[7,23,31,47],m['sha256'])
                    arm['gates'].append(dict(new_ppo_steps=target,sha256=m['sha256'],screen=full))
                    if min(p['pair_survivors'] for p in full['profiles'].values())<=3:
                        arm['early_stop_reason']='Severe survival regression: at most three paired survivors in a full profile'
                        break
                arm['final_sha256']=arm['latest_sha256']
                state.update(phase='final_fresh_evaluation',updated_utc=utc());atomic_json(path,state)
                arm['fresh']=physical(cp,folder/'fresh',SEEDS,arm['final_sha256'])
                atomic_json(path,state)
            verify_prerequisites();account()
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state)
            raise


if __name__=='__main__':main()
