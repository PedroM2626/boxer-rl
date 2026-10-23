"""Verify and archive every complete actor-rate comparison gate and consumed prefix."""
import argparse, hashlib, json, pickle, shutil
from pathlib import Path
import jax
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;state=json.loads((live/'comparison.json').read_text())
    assert state['status']=='completed' and len(state['arms'])==2
    assert state['diagnostic_consumed_steps']==state['diagnostic_collection_world_controls']==24576
    errors=list(live.rglob('*errors.log'));assert errors and all(not p.read_text().strip() for p in errors)
    report=Path('docs/results/striking-pilot/actor-rate-1835008');models=Path('checkpoints/striking-validation/actor-rate-1835008')
    result=dict(status='reviewed',main_adoption=False,promotion=False,diagnostic_ppo_total=1836288,
        formal_world_controls_total=1052177,new_collection_world_controls=24576,
        new_integrated_replay_presentations=98304,cumulative_supervised_updates=8000,
        native_error_logs_empty=True,foundation_history_records={},arms=[])
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert h==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        f=Path('checkpoints/curriculum')/stage;m=json.loads((f/'latest.json').read_text());b=(f/m['file']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(b)))
        result['foundation_history_records'][stage]=len(h)
    f=Path('checkpoints/striking-pilot/step-1048576');m=json.loads((f/'latest.json').read_text())
    assert hashlib.sha256((f/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    result['source_fresh'],controls=full_profiles(live/'source-fresh',state['fresh_seeds'],state['source_sha256'])
    for arm in state['arms']:
        name=arm['arm'];folder=live/name;row=dict(arm=name,gates=[])
        for gate in arm['gates']:
            target=gate['new_ppo_steps'];f=folder/f'gate-{target}'
            saved,m=load_checked(f/'checkpoint',gate['sha256']);require_ppo_checkpoint(saved)
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((saved['actor'],saved['critic'],saved['optimizer'],saved['norm'])))
            assert saved['consumed_steps']==saved['optimized_steps']==1835008+target
            assert saved['world_control_steps']==1212416+target
            assert int(saved['optimizer']['actor_adam'][0].count)==11144+target//1024*16
            assert int(saved['optimizer']['critic_adam'][0].count)==34816+target//1024*16
            assert saved['neutral_credits_by_corner']==[50229,48075]
            np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1),[49152+target//2]*2)
            assert bool(saved.get('conservative_actor_ppo'))==(name=='conservative_actor')
            full,n=full_profiles(f/'screen',[7,23,31,47],m['sha256']);controls+=n
            assert full['passes_complete']==gate['passes_complete']==False
            for profile,short in gate['screen'].items():
                assert full['profiles'][profile]['pair_survivors']==short['pair_survivors']
                assert full['profiles'][profile]['own_surviving_scoring_seeds']==short['own_surviving_scoring_seeds']
            row['gates'].append(dict(new_ppo_steps=target,sha256=m['sha256'],screen=full))
            shutil.copytree(f/'screen',report/name/f'gate-{target}'/'screen',dirs_exist_ok=True)
            out=models/name/f'gate-{target}';out.mkdir(parents=True,exist_ok=True)
            for fn in (m['file'],'latest.json','config.json'):shutil.copy2(f/'checkpoint'/fn,out/fn)
        fresh,n=full_profiles(folder/'fresh',state['fresh_seeds'],arm['final_checkpoint_sha256']);controls+=n
        assert not fresh['passes_complete'];row.update(fresh=fresh,final_sha256=arm['final_checkpoint_sha256'])
        shutil.copytree(folder/'fresh',report/name/'fresh',dirs_exist_ok=True)
        for fn in ('progress.json','state.json','initial-block-verified.json'):shutil.copy2(folder/fn,report/name/fn)
        shutil.copy2(folder/'update-audit/minibatch-measurements.json',report/name/'minibatch-measurements.json')
        metrics=[json.loads(line) for target in (1024,8192,16384) if (folder/f'train-{target}.log').exists() for line in (folder/f'train-{target}.log').read_text().splitlines() if line.strip()]
        total=8192 if name=='conservative_actor' else 16384
        assert len(metrics)==total//1024 and all(np.isfinite(r['mean_loss']) for r in metrics)
        assert [r['optimized_steps'] for r in metrics]==list(range(1836032,1835008+total+1,1024))
        (report/name/'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
        result['arms'].append(row)
    assert controls==state['diagnostic_evaluation_world_controls']==62822
    result['actual_new_evaluation_world_controls']=controls
    result['decision']='Reject actor-rate reduction: smaller measured KL did not improve physical gates. Unchanged control final improves matched fresh survival 16/24 to 19/24 and surviving active events 4 to 17 across all four seeds, but active survival falls 7/8 to 6/8 and no complete gate passes. Permit one 16384-credit ordinary version-6 diagnostic continuation from this latest control final only, full gates after 8192/16384, severe-regression and first-gate stagnation stops, fresh 16M source/final comparison. No main adoption, promotion or automatic further allocation.'
    shutil.copy2(live/'comparison.json',report/'completed-comparison.json');atomic_json(report/'analysis.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
