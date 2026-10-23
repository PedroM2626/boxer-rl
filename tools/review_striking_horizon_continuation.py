"""Verify and archive the bounded longer-horizon continuation without optimization."""
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
from boxer.striking_credit_horizon import validate_horizon_state
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    live = args.input
    state = json.loads((live / 'continuation.json').read_text())
    assert state['status'] == 'completed' and len(state['gates']) == 2
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 32768
    assert state['source_sha256'] == '22f4908af40b7c5269952d40a98f35630c2300d50d2bf8ba54c125f36cb6dd55'
    report = Path('docs/results/striking-pilot/credit-horizon-continue-1703936')
    models = Path('checkpoints/striking-validation/credit-horizon-continue-1703936')
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['source_sha256'])
    shutil.copytree(live / 'source-fresh', report / 'source-fresh', dirs_exist_ok=True)
    errors = list(live.rglob('*errors.log'))
    assert errors and all(not p.read_text().strip() for p in errors)
    result = dict(status='reviewed', main_adoption=False, promotion=False,
        diagnostic_ppo_total=1679616, formal_world_controls_total=813050,
        new_ppo_collection_world_controls=32768, new_integrated_replay_presentations=131072,
        cumulative_supervised_updates=8000, native_error_logs_empty=True,
        source_fresh=source, gates=[], foundation_history_records={})
    for stage in ('balance', 'walk_easy', 'walk', 'approach'):
        history = json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert history == json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text())
        assert promotion_ready(history, stage)
        folder = Path('checkpoints/curriculum') / stage
        m = json.loads((folder / 'latest.json').read_text())
        data = (folder / m['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m['sha256']
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage] = len(history)
    for gate in state['gates']:
        target = gate['new_ppo_steps']
        folder = live / f'gate-{target}'
        saved, manifest = load_checked(folder / 'checkpoint', gate['sha256'])
        validate_horizon_state(saved)
        assert saved['training_gamma'] == .999
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1703936 + target
        assert saved['world_control_steps'] == 1081344 + target
        assert int(saved['optimizer']['actor_adam'][0].count) == 9096 + target // 1024 * 16
        assert int(saved['optimizer']['critic_adam'][0].count) == 32768 + target // 1024 * 16
        assert sum(saved['neutral_credits_by_corner']) == 65536 + target
        assert all(a >= b for a, b in zip(saved['neutral_credits_by_corner'], [33273, 32263]))
        screen, count = full_profiles(folder / 'screen', [7, 23, 31, 47], manifest['sha256'])
        controls += count
        assert screen['passes_complete'] == gate['passes_complete'] == False
        out = models / f'gate-{target}'
        out.mkdir(parents=True, exist_ok=True)
        for filename in (manifest['file'], 'latest.json', 'config.json'):
            shutil.copy2(folder / 'checkpoint' / filename, out / filename)
        shutil.copytree(folder / 'screen', report / f'gate-{target}' / 'screen', dirs_exist_ok=True)
        result['gates'].append(dict(new_ppo_steps=target, sha256=manifest['sha256'], screen=screen,
            neutral_credits_by_corner=saved['neutral_credits_by_corner']))
    final, count = full_profiles(live / 'final-fresh', state['fresh_seeds'], state['gates'][-1]['sha256'])
    controls += count
    assert not final['passes_complete']
    assert controls == state['diagnostic_evaluation_world_controls'] == 42964
    result['final_fresh'] = final
    shutil.copytree(live / 'final-fresh', report / 'final-fresh', dirs_exist_ok=True)
    metrics = [json.loads(line) for target in (16384, 32768) for line in (live / f'train-{target}.log').read_text().splitlines() if line.strip()]
    assert len(metrics) == 32 and all(np.isfinite(r['mean_loss']) for r in metrics)
    assert [r['optimized_steps'] for r in metrics] == list(range(1704960, 1736705, 1024))
    (report / 'training-metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics))
    for filename in ('progress.json', 'state.json'):
        shutil.copy2(live / filename, report / filename)
    shutil.copy2(live / 'continuation.json', report / 'completed-continuation.json')
    result['decision'] = ('No additional optimizer allocation. Both complete generations fail. Neutral-role-1 scoring '
        'improves but final active survival and surviving event breadth regress on the same fresh seeds. '
        'Run a read-only matched source/final trace of all active and neutral-role-1 fresh episodes before '
        'choosing any training change. Retain the latest final and all consumed work; no rollback or adoption.')
    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
