"""Archive the completed mixed continuation and verify unchanged full physical gates."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked
from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT,fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); live = args.input
    state = json.loads((live / 'continuation.json').read_text())
    assert state['status'] == 'completed' and len(state['gates']) == 4
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 65536
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log'))
    report = Path('docs/results/striking-pilot/gae-consolidate-1998848')
    models = Path('checkpoints/striking-validation/gae-consolidate-1998848')
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['source_sha256'])
    shutil.copytree(live/'source-fresh',report/'source-fresh',dirs_exist_ok=True)
    initial=(live/'checkpoint/step-1998848.pkl').read_bytes()
    assert hashlib.sha256(initial).hexdigest()==state['source_sha256']=='3052e0f69fcadc823be14069ed462ee3057ae75a8028cb2d8aa5bf3615aa48b5'
    result = dict(status='reviewed', episodes=144, main_adoption=False, promotion=False, diagnostic_ppo_total=2131200,
        formal_world_controls_total=1874882, actual_new_evaluation_world_controls=70222,
        new_collection_world_controls=65536, new_integrated_replay_presentations=262144,
        cumulative_supervised_updates=8000, native_error_logs_empty=True,
        source_fresh=source, gates=[], foundation_history_records={})
    for stage in ('balance','walk_easy','walk','approach'):
        history = json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert history == json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text())
        assert promotion_ready(history, stage)
        folder = Path('checkpoints/curriculum') / stage; m = json.loads((folder / 'latest.json').read_text())
        data = (folder / m['file']).read_bytes(); assert hashlib.sha256(data).hexdigest() == m['sha256']
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage] = len(history)
    folder = Path('checkpoints/striking-pilot/step-1048576'); m = json.loads((folder / 'latest.json').read_text())
    assert hashlib.sha256((folder / m['file']).read_bytes()).hexdigest() == m['sha256'] == 'efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    for gate in state['gates']:
        target = gate['new_ppo_steps']; folder = live / f'gate-{target}'
        saved, manifest = load_checked(folder / 'checkpoint', gate['sha256']); require_ppo_checkpoint(saved)
        assert fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((saved['actor'],saved['critic'],saved['optimizer'],saved['norm'])))
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1998848 + target
        assert saved['world_control_steps'] == 1376256 + target
        assert int(saved['optimizer']['actor_adam'][0].count) == 13704 + target // 1024 * 16
        assert int(saved['optimizer']['critic_adam'][0].count) == 37376 + target // 1024 * 16
        assert saved['neutral_credits_by_corner'] == [50229,48075]
        np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1), [131072+target//2]*2)
        full, n = full_profiles(folder / 'screen', [7,23,31,47], manifest['sha256']); controls += n
        assert full == gate['screen'] and full['passes_complete']==(target!=32768)
        shutil.copytree(folder / 'screen', report / f'gate-{target}' / 'screen', dirs_exist_ok=True)
        out = models / f'gate-{target}'; out.mkdir(parents=True, exist_ok=True)
        for filename in (manifest['file'],'latest.json','config.json'):
            shutil.copy2(folder / 'checkpoint' / filename, out / filename)
        result['gates'].append(dict(new_ppo_steps=target, sha256=manifest['sha256'], screen=full,
            mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
    final, n = full_profiles(live / 'final-fresh', state['fresh_seeds'], state['gates'][-1]['sha256']); controls += n
    assert final['passes_complete'] and controls == state['diagnostic_evaluation_world_controls'] == 70222
    result['final_fresh'] = final
    shutil.copytree(live / 'final-fresh', report / 'final-fresh', dirs_exist_ok=True)
    metrics = [json.loads(line) for target in (16384,32768,49152,65536)
        for line in (live / f'train-{target}.log').read_text().splitlines() if line.strip()]
    assert len(metrics) == 64 and all(np.isfinite(r['mean_loss']) for r in metrics)
    assert [r['optimized_steps'] for r in metrics] == list(range(1999872,2064385,1024))
    (report / 'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    for filename in ('state.json','progress.json'):
        shutil.copy2(live / filename, report / filename)
    shutil.copy2(live / 'continuation.json', report / 'completed-continuation.json')
    history_path=Path('docs/results/striking-pilot/gae-1966080/version-12-original-gate-history.json')
    history=json.loads(history_path.read_text());records=history['records']
    new=[dict(sha256=g['sha256'],screen=g['screen']) for g in result['gates']]
    assert len(records) in (4,8)
    if len(records)==8:assert records[-4:]==new
    else:records+=new
    assert len({r['sha256'] for r in records})==8 and [r['screen']['passes_complete'] for r in records]==[False,False,True,False,True,False,True,True]
    result['original_complete_pass_streak']=2
    atomic_json(history_path,dict(kind='version_12_original_full_screens',required_distinct_consecutive_passes=2,pass_streak=2,original_screen_repeatability_met=True,independent_fresh_final_passes=True,main_adoption=False,promotion=False,records=records))
    # Every included update is retained, not only favorable gate checkpoints.
    for target in range(1024,65537,1024):
        current=pickle.loads((live/'checkpoint'/f'step-{1998848+target}.pkl').read_bytes());require_ppo_checkpoint(current)
        assert current['consumed_steps']==current['optimized_steps']==1998848+target and current['world_control_steps']==1376256+target
        assert int(current['optimizer']['actor_adam'][0].count)==13704+target//1024*16 and int(current['optimizer']['critic_adam'][0].count)==37376+target//1024*16
        assert current.get('long_gae_ppo') and current['training_gae_lambda']==.99 and fingerprint(current['frozen_opponent'])==FROZEN_FINGERPRINT
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(current[k] for k in ('actor','critic','optimizer','norm','key'))))
        np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1),[131072+target//2]*2)
    assert len(list(live.rglob('*errors.log')))==22
    for p in live.rglob('*errors.log'):
        out=report/p.relative_to(live).with_suffix('.txt');out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    shutil.copy2(live/'config.json',report/'config.json')
    result.update(screen_milestone_verified=True,general_boxing_certified=False)
    result['decision']=('Verify the original ten-second version-12 readiness milestone: two distinct consecutive complete original passes, followed by a complete fresh 29M pass (23/24 pairs). Freeze latest bea8147f without further PPO or main adoption. This is not reliable general boxing: final original mirror has one reciprocal surviving episode and fresh mirror three. Authorize a broader frozen-candidate audit before opponent-pool training or longer-round certification: four untouched 30M full groups (96 episodes), two archived opponents in both evaluated corners on fresh 31M seeds (32 episodes), and one exact two-episode first-29M-seed video replay (selected illustration, not a new holdout). Original full gates stay unchanged. Cross-policy diagnostic criteria are declared before results. At most 65000 new evaluation controls, zero optimization. Diagnostic PPO stays 2131200; formal simulation starts 1874882, ceiling 1939882. Preserve all failed history, hashes and consumed compute. No new training, main adoption, automatic promotion or further allocation from this milestone alone.')
    milestone=Path('docs/results/striking-pilot/gae-readiness-certification');milestone.mkdir(parents=True,exist_ok=True)
    atomic_json(milestone/'manifest.json',dict(scope='Original ten-second complete screen repeatability plus one fresh four-seed group',
        profile_version=12,training_gae_lambda=.99,original_last_two=[g['sha256'] for g in result['gates'][-2:]],
        fresh_seeds=state['fresh_seeds'],fresh_complete_pass=True,latest_sha256=result['gates'][-1]['sha256'],
        consumed_learner_steps=2064384,world_control_steps=1441792,diagnostic_ppo_total=2131200,
        formal_world_controls_total=1874882,main_adoption=False,promotion=False,general_boxing_certified=False,
        evidence='../gae-consolidate-1998848/analysis.json'))

    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
