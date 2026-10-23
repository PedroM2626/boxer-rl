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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); live = args.input
    state = json.loads((live / 'continuation.json').read_text())
    assert state['status'] == 'completed' and len(state['gates']) == 2
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 65536
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log'))
    report = Path('docs/results/striking-pilot/mixed-partner-continue-1769472')
    models = Path('checkpoints/striking-validation/mixed-partner-continue-1769472')
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['source_sha256'])
    result = dict(status='reviewed', main_adoption=False, promotion=False, diagnostic_ppo_total=1810688,
        formal_world_controls_total=989355, actual_new_evaluation_world_controls=42608,
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
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1769472 + target
        assert saved['world_control_steps'] == 1146880 + target
        assert int(saved['optimizer']['actor_adam'][0].count) == 10120 + target // 1024 * 16
        assert int(saved['optimizer']['critic_adam'][0].count) == 33792 + target // 1024 * 16
        assert saved['neutral_credits_by_corner'] == [50229,48075]
        np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1), [16384+target//2]*2)
        full, n = full_profiles(folder / 'screen', [7,23,31,47], manifest['sha256']); controls += n
        assert full == gate['screen'] and not full['passes_complete']
        shutil.copytree(folder / 'screen', report / f'gate-{target}' / 'screen', dirs_exist_ok=True)
        out = models / f'gate-{target}'; out.mkdir(parents=True, exist_ok=True)
        for filename in (manifest['file'],'latest.json','config.json'):
            shutil.copy2(folder / 'checkpoint' / filename, out / filename)
        result['gates'].append(dict(new_ppo_steps=target, sha256=manifest['sha256'], screen=full,
            mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
    final, n = full_profiles(live / 'final-fresh', state['fresh_seeds'], state['gates'][-1]['sha256']); controls += n
    assert not final['passes_complete'] and controls == state['diagnostic_evaluation_world_controls'] == 42608
    result['final_fresh'] = final
    shutil.copytree(live / 'final-fresh', report / 'final-fresh', dirs_exist_ok=True)
    metrics = [json.loads(line) for target in (32768,65536)
        for line in (live / f'train-{target}.log').read_text().splitlines() if line.strip()]
    assert len(metrics) == 64 and all(np.isfinite(r['mean_loss']) for r in metrics)
    assert [r['optimized_steps'] for r in metrics] == list(range(1770496,1835009,1024))
    (report / 'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    for filename in ('state.json','progress.json'):
        shutil.copy2(live / filename, report / filename)
    shutil.copy2(live / 'continuation.json', report / 'completed-continuation.json')
    result['decision'] = ('No further unchanged PPO allocation. Both complete generations fail. Intermediate neutral '
        'profiles pass but active survival does not; the final regresses to 15/24 fresh survivors versus 21/24 source. '
        'Retain latest final without rollback. Audit actor versus normalizer drift and consecutive saved PPO blocks '
        'on fixed archived states without physics, optimization or new supervised compute before selecting a correction.')
    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
