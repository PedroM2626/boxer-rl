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
    assert state['status'] == 'completed' and len(state['gates']) == 1
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 8192
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log') if p.name!='replay-errors.log')
    assert hashlib.sha256((live/'replay-errors.log').read_bytes()).hexdigest()=='5f745dc5b2c316c0cfcc27eaa60e9f2de3001e10ce05a73e26c4207b6073eebe'
    assert hashlib.sha256((live/'first-full-screen-pass-1908736.mp4').read_bytes()).hexdigest()=='69270d39a2b33e0da2df490ec2deb57e2a1e1b2cc09b010b8eb14aa1ab0944fa'
    report = Path('docs/results/striking-pilot/fixed-repeat-1908736')
    models = Path('checkpoints/striking-validation/fixed-repeat-1908736')
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['source_sha256'])
    shutil.copytree(live/'source-fresh',report/'source-fresh',dirs_exist_ok=True)
    replay_progress=json.loads((live/'replay-progress.json').read_text());assert replay_progress['status']=='completed' and replay_progress['simulated_world_transitions']==1000
    controls+=replay_progress['simulated_world_transitions']
    result = dict(status='reviewed', main_adoption=False, promotion=False, diagnostic_ppo_total=1918208,
        formal_world_controls_total=1296474, actual_new_evaluation_world_controls=35525,
        new_collection_world_controls=8192, new_integrated_replay_presentations=32768,
        cumulative_supervised_updates=8000, native_physics_training_error_logs_empty=True, reviewed_renderer_warning_retained=True,
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
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1908736 + target
        assert saved['world_control_steps'] == 1286144 + target
        assert int(saved['optimizer']['actor_adam'][0].count) == 12296 + target // 1024 * 16
        assert int(saved['optimizer']['critic_adam'][0].count) == 35968 + target // 1024 * 16
        assert saved['neutral_credits_by_corner'] == [50229,48075]
        np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1), [86016+target//2]*2)
        full, n = full_profiles(folder / 'screen', [7,23,31,47], manifest['sha256']); controls += n
        assert full == gate['screen'] and full['passes_complete']==True
        shutil.copytree(folder / 'screen', report / f'gate-{target}' / 'screen', dirs_exist_ok=True)
        out = models / f'gate-{target}'; out.mkdir(parents=True, exist_ok=True)
        for filename in (manifest['file'],'latest.json','config.json'):
            shutil.copy2(folder / 'checkpoint' / filename, out / filename)
        result['gates'].append(dict(new_ppo_steps=target, sha256=manifest['sha256'], screen=full,
            mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
    final, n = full_profiles(live / 'final-fresh', state['fresh_seeds'], state['gates'][-1]['sha256']); controls += n
    assert not final['passes_complete'] and controls == state['diagnostic_evaluation_world_controls'] == 35525
    result['final_fresh'] = final
    shutil.copytree(live / 'final-fresh', report / 'final-fresh', dirs_exist_ok=True)
    metrics = [json.loads(line) for target in (8192,)
        for line in (live / f'train-{target}.log').read_text().splitlines() if line.strip()]
    assert len(metrics) == 8 and all(np.isfinite(r['mean_loss']) for r in metrics)
    assert [r['optimized_steps'] for r in metrics] == list(range(1909760,1916929,1024))
    (report / 'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    for filename in ('state.json','progress.json'):
        shutil.copy2(live / filename, report / filename)
    shutil.copy2(live / 'continuation.json', report / 'completed-continuation.json')
    history_path=Path('docs/results/striking-pilot/fixed-opponent-original-gate-history.json')
    old=json.loads(history_path.read_text());records=old['records']
    assert len(records) in (4,5)
    if len(records)==5:
        assert records[-1]['sha256']==state['gates'][-1]['sha256'];records=records[:4]
    assert records[-1]['sha256']==state['source_sha256'] and records[-1]['screen']['passes_complete']
    records.append(dict(sha256=state['gates'][-1]['sha256'],screen=result['gates'][-1]['screen']))
    assert len({r['sha256'] for r in records})==5 and [r['screen']['passes_complete'] for r in records]==[False,False,False,True,True]
    result['original_complete_pass_streak']=2
    atomic_json(history_path,dict(kind='version_8_original_full_screens',required_distinct_consecutive_passes=2,pass_streak=2,original_screen_repeatability_met=True,independent_fresh_final_passes=False,main_adoption=False,promotion=False,records=records))
    result['decision'] = ('Two distinct consecutive original full screens pass, but fresh 20M neutral-0 survival is 6/8 and the complete fresh set fails (22/24 pairs versus 21/24 source). No phase promotion or main adoption. Retain the latest final and frozen opponent. Allocate no further unchanged PPO yet; perform one larger independent generalization and cross-policy audit: four disjoint 21M full seed groups plus two archived opposing policies across both corner assignments on 22M seeds. Maximum 128 episodes/64000 evaluation controls, zero optimizer or collection updates. Preserve exact physical gates; audit groups and mixed-policy matches cannot replace consecutive distinct-checkpoint certification. Decide training coverage only after complete physical evidence.')

    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
