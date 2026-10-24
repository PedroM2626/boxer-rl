"""Verify every KL-guard/control block and complete unchanged physical evaluation."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
import jax,numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_fixed_opponent import fingerprint,FROZEN_FINGERPRINT
from tools.striking_contact_compare import load_checked
from tools.striking_kl_guard_compare import SOURCE,SHA
from boxer.striking_kl_guard import expected_actor_count
from tools.review_striking_credit_horizon import full_profiles


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;state=json.loads((live/'comparison.json').read_text())
    assert state['status']=='completed' and len(state['arms'])==2 and state['diagnostic_consumed_steps']==state['diagnostic_collection_world_controls']==32768
    errors=list(live.rglob('*errors.log'));assert len(errors)==27 and all(not p.read_text().strip() for p in errors)
    source,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(source)
    out=Path('docs/results/striking-pilot/kl-guard-1949696');models=Path('checkpoints/striking-validation/kl-guard-1949696')
    result=dict(status='reviewed',diagnostic_ppo_total=2016512,formal_world_controls_total=1633434,
        actual_new_evaluation_world_controls=74819,new_ppo_collection_world_controls=32768,
        new_integrated_replay_presentations=131072,cumulative_supervised_updates=8000,
        main_adoption=False,promotion=False,native_error_logs_empty=True,arms=[],foundation_history_records={})
    fresh,controls=full_profiles(live/'source-fresh',state['fresh_seeds'],SHA);assert fresh==state['source_fresh'];result['source_fresh']=fresh
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert h==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        p=Path('checkpoints/curriculum')/stage;m=json.loads((p/'latest.json').read_text());data=(p/m['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage]=len(h)
    p=Path('checkpoints/striking-pilot/step-1048576');m=json.loads((p/'latest.json').read_text())
    assert hashlib.sha256((p/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    original=[];cost_history=[]
    for arm in state['arms']:
        name=arm['arm'];folder=live/name;cost=name=='kl_guard';review=dict(arm=name,gates=[])
        metrics=[]
        for target in range(1024,16385,1024):
            saved=pickle.loads((folder/'checkpoint'/f'step-{1949696+target}.pkl').read_bytes());require_ppo_checkpoint(saved)
            assert saved.get('kl_guard_ppo',False)==cost and fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
            assert saved['consumed_steps']==saved['optimized_steps']==1949696+target and saved['world_control_steps']==1327104+target
            assert int(saved['optimizer']['actor_adam'][0].count)==(expected_actor_count(saved) if cost else 12936+target//1024*16) and int(saved['optimizer']['critic_adam'][0].count)==36608+target//1024*16
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
            np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1),[106496+target//2]*2)
            assert saved['neutral_credits_by_corner']==[50229,48075]
        for target in (1024,8192,16384):
            rows=[json.loads(line) for line in (folder/f'train-{target}.log').read_text().splitlines() if line.strip()]
            assert all(np.isfinite(t['mean_loss']) for t in rows);metrics.extend(rows)
        assert [r['optimized_steps'] for r in metrics]==list(range(1950720,1966081,1024))
        initial=pickle.loads((folder/'checkpoint'/('initial.pkl' if cost else 'step-1949696.pkl')).read_bytes())
        for k in source:
            for a,b in zip(jax.tree_util.tree_leaves(source[k]),jax.tree_util.tree_leaves(initial[k])):np.testing.assert_equal(a,b)
        first=pickle.loads((folder/'checkpoint/step-1950720.pkl').read_bytes())
        audit=json.loads((folder/'update-audit/minibatch-measurements.json').read_text())
        assert len(audit['records'])==18 and audit['records'][0]['profiles']['all']['ratio_clip_fraction']==0.
        assert abs(audit['records'][0]['profiles']['all']['sampled_approx_kl'])<1e-8
        review['initial_update_geometry']=audit['records'][-1]['profiles']['all']
        shutil.copy2(folder/'update-audit/minibatch-measurements.json',out/(name+'-geometry.json'))
        for gate in arm['gates']:
            target=gate['new_ppo_steps'];p=folder/f'gate-{target}'/'checkpoint';saved,m=load_checked(p,gate['sha256']);require_ppo_checkpoint(saved)
            full,n=full_profiles(folder/f'gate-{target}'/'screen',[7,23,31,47],m['sha256']);controls+=n
            assert full==gate['screen']
            review['gates'].append(dict(sha256=m['sha256'],new_ppo_steps=target,screen=full))
            (cost_history if cost else original).append(dict(sha256=m['sha256'],screen=full))
            dest=models/name/f'gate-{target}';dest.mkdir(parents=True,exist_ok=True)
            for f in (m['file'],'latest.json','config.json'):shutil.copy2(p/f,dest/f)
        final,n=full_profiles(folder/'fresh',state['fresh_seeds'],arm['final_sha256']);controls+=n;assert final==arm['fresh'] and not final['passes_complete']
        review.update(final_sha256=arm['final_sha256'],fresh=final,initial_all_source_leaves_retained=True,
            actor_steps_by_rollout=saved.get('kl_guard_actor_steps_by_rollout',[]),actor_adam_count=int(saved['optimizer']['actor_adam'][0].count),critic_adam_count=int(saved['optimizer']['critic_adam'][0].count))
        dest=out/name;dest.mkdir(parents=True,exist_ok=True)
        (dest/'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
        for f in ('config.json','state.json','progress.json'):shutil.copy2(folder/f,dest/f)
        for sub in ('gate-8192/screen','gate-16384/screen','fresh'):
            for f in (folder/sub).glob('*.json'):
                target=dest/sub/f.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
        result['arms'].append(review)
    assert controls==state['diagnostic_evaluation_world_controls']==74819
    audits=[json.loads((live/name/'update-audit/minibatch-measurements.json').read_text()) for name in ('kl_guard','control')]
    assert audits[0]['collection_sha256']==audits[1]['collection_sha256']
    result['matching_initial_collected_batch_sha256']=audits[0]['collection_sha256']
    history_path=Path('docs/results/striking-pilot/fixed-opponent-original-gate-history.json');history=json.loads(history_path.read_text())
    records=history['records'];assert len(records) in (9,11)
    if len(records)==11:assert records[-2:]==original
    else:records+=original
    assert len({r['sha256'] for r in records})==11 and [r['screen']['passes_complete'] for r in records]==[False,False,False,True,True,False,False,False,False,True,False]
    atomic_json(history_path,dict(kind='version_8_original_full_screens',required_distinct_consecutive_passes=2,pass_streak=0,
        original_screen_repeatability_met=False,historical_two_distinct_consecutive_passes=True,independent_fresh_final_passes=False,main_adoption=False,promotion=False,records=records))
    atomic_json(out/'version-11-original-gate-history.json',dict(kind='version_11_original_full_screens',pass_streak=0,required_distinct_consecutive_passes=2,records=cost_history,promotion=False))
    result.update(current_version_8_pass_streak=0,current_version_11_pass_streak=0)
    result['decision']=('Reject this .02 actor KL stop as a validated stability remedy. Both guarded original screens fail; control passes then regresses severely. Final fresh survival is 19/24 guarded, 18/24 control versus 23/24 source, with all complete fresh gates failing. Do not extend PPO, adopt a failed actor, restore earlier checkpoints, or tune another KL target blindly. Preserve latest endpoints and complete consumed ledgers. Permit a passive matched failure-timing replay of source and both latest endpoints on the same 25M seed group, all three unchanged physical profiles, maximum 72 episodes / 36000 evaluation controls and zero optimizer updates. Verify every original trial physical field exactly, classify first-fall mechanisms relative to physical contact and scoring, archive traces and decide the next intervention from that evidence. Diagnostic PPO remains 2016512. Formal evaluation starts 1633434 with a bounded diagnostic ceiling 1669434. Main counters and all gates remain unchanged; no automatic further extension.')
    for p in errors:
        dest=out/p.relative_to(live).with_suffix('.txt');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(live/'comparison.json',out/'completed-comparison.json');atomic_json(out/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','source_fresh')},indent=2))


if __name__=='__main__':main()
