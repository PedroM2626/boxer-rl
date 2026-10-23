"""Verify all complete mixed/control gates, exact checkpoints and diagnostic accounting."""
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
    args = parser.parse_args()
    live = args.input
    state = json.loads((live / 'comparison.json').read_text())
    assert state['status'] == 'completed' and {a['arm'] for a in state['arms']} == {'mixed', 'neutral'}
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 65536
    report = Path('docs/results/striking-pilot/mixed-partner-1736704')
    models = Path('checkpoints/striking-validation/mixed-partner-1736704')
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log'))
    result = dict(status='reviewed', main_adoption=False, promotion=False, diagnostic_ppo_total=1745152,
        formal_world_controls_total=946747, actual_new_evaluation_world_controls=76692,
        new_collection_world_controls=65536, new_integrated_replay_presentations=262144,
        cumulative_supervised_updates=8000, native_error_logs_empty=True, arms=[], foundation_history_records={})
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
    folder = Path('checkpoints/striking-pilot/step-1048576')
    m = json.loads((folder / 'latest.json').read_text())
    assert hashlib.sha256((folder / m['file']).read_bytes()).hexdigest() == m['sha256'] == 'efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['source_sha256'])
    result['source_fresh'] = source
    finals = {}
    for arm in state['arms']:
        name = arm['arm']; folder = live / name
        assert len(arm['gates']) == 2
        checks = []
        for gate in arm['gates']:
            target = gate['new_ppo_steps']
            cp = folder / f'gate-{target}' / 'checkpoint'
            saved, manifest = load_checked(cp, gate['sha256'])
            require_ppo_checkpoint(saved)
            assert saved['consumed_steps'] == saved['optimized_steps'] == 1736704 + target
            assert saved['world_control_steps'] == 1114112 + target
            assert int(saved['optimizer']['actor_adam'][0].count) == 9608 + target // 1024 * 16
            assert int(saved['optimizer']['critic_adam'][0].count) == 33280 + target // 1024 * 16
            if name == 'mixed':
                assert saved['neutral_credits_by_corner'] == [50229, 48075]
                np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1), [target//2]*2)
            else: assert sum(saved['neutral_credits_by_corner']) == 98304 + target
            screen, n = full_profiles(folder / f'gate-{target}' / 'screen', [7,23,31,47], manifest['sha256'])
            controls += n
            assert screen['passes_complete'] == gate['passes_complete'] == False
            shutil.copytree(folder / f'gate-{target}' / 'screen', report / name / f'gate-{target}' / 'screen', dirs_exist_ok=True)
            out = models / name / f'gate-{target}'; out.mkdir(parents=True, exist_ok=True)
            for filename in (manifest['file'], 'latest.json', 'config.json'):
                shutil.copy2(cp / filename, out / filename)
            checks.append(dict(new_ppo_steps=target, sha256=manifest['sha256'], screen=screen))
        final, manifest = load_checked(folder / 'checkpoint', arm['final_checkpoint_sha256'])
        require_ppo_checkpoint(final); finals[name] = final
        fresh, n = full_profiles(folder / 'fresh', state['fresh_seeds'], manifest['sha256'])
        controls += n
        assert not fresh['passes_complete']
        shutil.copytree(folder / 'fresh', report / name / 'fresh', dirs_exist_ok=True)
        metrics = [json.loads(line) for target in (1024,16384,32768)
            for line in (folder / f'train-{target}.log').read_text().splitlines() if line.strip()]
        assert len(metrics) == 32 and all(np.isfinite(r['mean_loss']) for r in metrics)
        assert [r['optimized_steps'] for r in metrics] == list(range(1737728,1769473,1024))
        dest = report / name; dest.mkdir(parents=True, exist_ok=True)
        (dest / 'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
        for filename in ('state.json','progress.json','initial-block-verified.json'):
            shutil.copy2(folder / filename, dest / filename)
        result['arms'].append(dict(arm=name, final_sha256=manifest['sha256'], gates=checks, fresh=fresh,
            actor_adam_count=10120, critic_adam_count=33792,
            neutral_credits_by_corner=final['neutral_credits_by_corner'],
            mixed_credits_by_mode_and_corner=final.get('mixed_credits_by_mode_and_corner'),
            severe_survival_flag=arm.get('early_stop_reason')))
    assert controls == state['diagnostic_evaluation_world_controls'] == 76692
    for key in ('bc_sampler_state','aux_sampler_state','retention_sampler_state'):
        assert finals['mixed'][key] == finals['neutral'][key]
    np.testing.assert_array_equal(finals['mixed']['key'], finals['neutral']['key'])
    result['matched_sampler_and_training_rng_streams'] = True
    result['actor_parameter_l2_between_arms'] = float(np.sqrt(sum(np.sum((a-b)**2) for a,b in zip(
        jax.tree_util.tree_leaves(finals['mixed']['actor']), jax.tree_util.tree_leaves(finals['neutral']['actor'])))))
    result['decision'] = ('Neither arm passes either complete generation. Stop the severely regressed neutral control. '
        'Mixed fresh survival improves to 24/24 versus 18/24 control and 22/24 source, with sparse valid active strikes. '
        'Permit one ordinary 65536-credit bounded mixed-profile diagnostic continuation from its latest final only, '
        'full screens at 32768/65536, severe-regression and no-measured-improvement early stops. '
        'Keep all source/optimizer/mode/consumed prefixes, unchanged gates and main ceilings. No promotion or adoption.')
    shutil.copy2(live / 'comparison.json', report / 'completed-comparison.json')
    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
