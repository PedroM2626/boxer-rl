"""Independently verify and archive both completed pool-comparison arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil

# Read-only validation must not reserve GPU memory beside persistent CPU jobs.
if os.environ.get('JAX_PLATFORMS','cpu')!='cpu':
    raise ValueError('opponent-pool checkpoint review requires JAX_PLATFORMS=cpu')
os.environ.setdefault('JAX_PLATFORMS','cpu')

import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_opponent_pool import OFFSET, WORLD_OFFSET, SOURCE_SHA, SOURCE, TRANSITION
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import (SEEDS, PPO_BEFORE, EVALUATION_BEFORE,
    MAX_PPO, MAX_EVALUATION, verify_prerequisites, verify_checkpoint)
from tools.review_striking_credit_horizon import full_profiles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();live=args.input
    state=json.loads((live/'comparison.json').read_text())
    assert state['status']=='completed' and state['phase']=='awaiting_review'
    assert state['source_sha256']==SOURCE_SHA and state['fresh_seeds']==SEEDS
    assert [a['arm'] for a in state['arms']]==['pool','control']
    assert state['maximum_new_ppo_steps']==MAX_PPO and state['maximum_evaluation_world_controls']==MAX_EVALUATION
    assert state['diagnostic_ppo_before_job']==PPO_BEFORE and state['formal_evaluation_world_controls_before_job']==EVALUATION_BEFORE
    assert not state['promotion'] and state['policy_adoption']=='none'
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log'))
    prerequisites=verify_prerequisites();assert prerequisites==state['prerequisites']
    source,_=load_checked(Path(SOURCE),SOURCE_SHA);require_ppo_checkpoint(source)
    source_full,controls=full_profiles(live/'source-fresh',SEEDS,SOURCE_SHA)
    assert source_full==state['source_fresh']
    report=Path('docs/results/striking-pilot/pool-2064384')
    models=Path('checkpoints/striking-validation/pool-2064384')
    result=dict(status='reviewed',source_sha256=SOURCE_SHA,source_fresh=source_full,
        main_adoption=False,promotion=False,assistance=False,unchanged_full_gates=True,
        native_error_logs_empty=True,prerequisites=prerequisites,
        new_separate_supervised_updates=0,cumulative_supervised_updates=8000,
        main_consumed_retained=1048576,main_project_consumed_retained=476041216,arms=[])
    new_steps=0;episodes=24
    for arm in state['arms']:
        name=arm['arm'];folder=live/name;target=arm['new_ppo_steps']
        assert target in (8192,16384) and arm['integration_block_verified']
        saved,manifest=verify_checkpoint(folder,target,name=='pool')
        assert manifest['sha256']==arm['final_sha256']==arm['latest_sha256']
        initial=pickle.loads((folder/'checkpoint/initial.pkl').read_bytes())
        require_ppo_checkpoint(initial)
        for key in source.keys()-{'opponent_mode'}:
            for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(initial[key])):
                np.testing.assert_equal(a,b)
        if name=='pool':assert initial['opponent_pool_transition']==TRANSITION
        block_hashes=[]
        for step in range(1024,target+1,1024):
            payload=(folder/f'checkpoint/step-{OFFSET+step}.pkl').read_bytes()
            cp=pickle.loads(payload);require_ppo_checkpoint(cp)
            assert cp['consumed_steps']==cp['optimized_steps']==OFFSET+step
            assert cp['world_control_steps']==WORLD_OFFSET+step
            assert cp.get('pool_opponent_ppo',False)==(name=='pool')
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(cp[k] for k in ('actor','critic','optimizer','norm','key'))))
            block_hashes.append(dict(new_ppo_steps=step,sha256=hashlib.sha256(payload).hexdigest(),
                actor_count=int(cp['optimizer']['actor_adam'][0].count),
                critic_count=int(cp['optimizer']['critic_adam'][0].count)))
        gates=[];streak=0
        for gate in arm['gates']:
            location=folder/f"gate-{gate['new_ppo_steps']}"
            assert gate['sha256']==next(b['sha256'] for b in block_hashes if b['new_ppo_steps']==gate['new_ppo_steps'])
            full,n=full_profiles(location/'screen',[7,23,31,47],gate['sha256']);controls+=n;episodes+=24
            assert full==gate['screen']
            cp,m=load_checked(location/'checkpoint',gate['sha256']);require_ppo_checkpoint(cp)
            assert cp['optimized_steps']==OFFSET+gate['new_ppo_steps']
            streak=streak+1 if full['passes_complete'] else 0
            gates.append(dict(gate,consecutive_complete_pass_streak=streak))
        assert len({g['sha256'] for g in gates})==len(gates)
        assert len(gates)==(1 if target==8192 else 2)
        if target==8192:assert min(p['pair_survivors'] for p in gates[-1]['screen']['profiles'].values())<=3
        fresh,n=full_profiles(folder/'fresh',SEEDS,manifest['sha256']);controls+=n;episodes+=24
        assert fresh==arm['fresh']
        progress=json.loads((folder/'progress.json').read_text())
        assert progress['consumed_steps']==progress['optimized_steps']==OFFSET+target
        assert progress['world_control_steps']==WORLD_OFFSET+target
        rows=[]
        for block in (1024,8192,16384):
            log=folder/f'train-{block}.log'
            if log.exists():rows += [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        assert [r['optimized_steps'] for r in rows]==list(range(OFFSET+1024,OFFSET+target+1,1024))
        assert all(np.isfinite(r['mean_loss']) for r in rows)
        new_steps+=target
        result['arms'].append(dict(arm=name,sha256=manifest['sha256'],new_ppo_steps=target,
            original_complete_pass_streak=streak,gates=gates,fresh=fresh,block_hashes=block_hashes,
            pool_credits_by_opponent_and_corner=saved.get('pool_credits_by_opponent_and_corner')))
    actual=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in live.rglob('*-progress.json'))
    assert controls==actual==state['diagnostic_evaluation_world_controls']<=MAX_EVALUATION
    assert new_steps==state['diagnostic_consumed_steps']==state['diagnostic_collection_world_controls']<=MAX_PPO
    result.update(episodes=episodes,actual_new_evaluation_world_controls=actual,
        new_collection_world_controls=new_steps,new_integrated_replay_presentations=4*new_steps,
        diagnostic_ppo_total=PPO_BEFORE+new_steps,formal_world_controls_total=EVALUATION_BEFORE+actual)
    pool,control=result['arms']
    result['fresh_pair_survival_difference_pool_minus_control']=sum(p['pair_survivors'] for p in pool['fresh']['profiles'].values())-sum(p['pair_survivors'] for p in control['fresh']['profiles'].values())
    result['decision']='Freeze both actual finals for review; no automatic allocation or adoption. Inspect full profiles, scoring breadth, reciprocal contacts and pool coverage before any continuation.'
    # Archive only after all independent assertions pass. Intermediate payloads
    # remain durable locally; all block hashes and evaluated snapshots are published.
    shutil.copytree(live/'source-fresh',report/'source-fresh',dirs_exist_ok=True)
    for arm in state['arms']:
        name=arm['arm'];folder=live/name
        for gate in arm['gates']:
            label=f"gate-{gate['new_ppo_steps']}"
            shutil.copytree(folder/label/'screen',report/name/label/'screen',dirs_exist_ok=True)
            shutil.copytree(folder/label/'checkpoint',models/name/label,dirs_exist_ok=True)
        shutil.copytree(folder/'fresh',report/name/'fresh',dirs_exist_ok=True)
        shutil.copytree(folder/'update-audit',report/name/'update-audit',dirs_exist_ok=True)
        for filename in ('progress.json','state.json','config.json'):
            shutil.copy2(folder/filename,report/name/filename)
    atomic_json(report/'analysis.json',result);atomic_json(report/'comparison.json',state)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
