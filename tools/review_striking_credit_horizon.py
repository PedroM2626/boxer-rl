"""Archive and independently verify the completed fixed horizon comparison."""
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
from tools.striking_contact_compare import load_checked


def full_profiles(folder, seeds, sha):
    profiles = {}
    controls = 0
    for role in (0, 1, None):
        name = 'selfplay' if role is None else f'neutral-{role}'
        report = json.loads((folder / (name + '.json')).read_text())
        progress = json.loads((folder / (name + '-progress.json')).read_text())
        rows = report['trials']
        assert progress['status'] == 'completed'
        assert report['striking_sha256'] == progress['striking_sha256'] == sha
        assert len(rows) == 8 and {(r['seed'], r['swap']) for r in rows} == {(s, w) for s in seeds for w in (False, True)}
        assert not report['assistance'] and not report['sampled_policy']
        assert report.get('diagnostic_base_yaw_scale',1.)==1., 'Modified automatic controller is diagnostic, not a promotable full gate'
        assert all(r['finite'] for r in rows)
        assert report['simulated_world_transitions'] == progress['simulated_world_transitions']
        controls += report['simulated_world_transitions']
        survivors = [r for r in rows if r['survived']]
        scoring_seeds = sorted({r['seed'] for r in survivors if (sum(r['scored_by_corner']) if role is None else r['scored_by_corner'][role]) > 0})
        profiles[name] = dict(pair_survivors=len(survivors), own_surviving_scoring_seeds=scoring_seeds,
            surviving_events_by_corner=np.sum([r['scored_by_corner'] for r in survivors], axis=0).tolist(),
            surviving_two_way_episodes=sum(all(v > 0 for v in r['scored_by_corner']) for r in survivors),
            passes_profile=len(survivors) >= (8 if role == 1 else 7) and (role is None or len(scoring_seeds) >= 2))
    return dict(profiles=profiles, passes_complete=all(p['passes_profile'] for p in profiles.values())), controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    live = args.input
    state = json.loads((live / 'comparison.json').read_text())
    assert state['status'] == 'completed' and len(state['arms']) == 2
    assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == 65536
    report = Path('docs/results/striking-pilot/credit-horizon-1671168')
    models = Path('checkpoints/striking-validation/credit-horizon-1671168')
    canonical, _ = load_checked(live / 'transition', state['transition_sha256'])
    validate_horizon_state(canonical)
    errors = list(live.rglob('*errors.log'))
    assert errors and all(not p.read_text().strip() for p in errors)
    result = dict(status='reviewed', main_adoption=False, promotion=False,
        diagnostic_ppo_total=1646848, formal_world_controls_total=770086,
        new_ppo_collection_world_controls=65536, cumulative_supervised_updates=8000,
        new_integrated_replay_presentations=262144, native_error_logs_empty=True,
        foundation_history_records={}, arms=[])
    for stage in ('balance', 'walk_easy', 'walk', 'approach'):
        history = json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert history == json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text())
        assert promotion_ready(history, stage)
        folder = Path('checkpoints/curriculum') / stage
        manifest = json.loads((folder / 'latest.json').read_text())
        data = (folder / manifest['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['sha256']
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage] = len(history)
    main_folder = Path('checkpoints/striking-pilot/step-1048576')
    main_manifest = json.loads((main_folder / 'latest.json').read_text())
    assert hashlib.sha256((main_folder / main_manifest['file']).read_bytes()).hexdigest() == main_manifest['sha256'] == 'efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    source, controls = full_profiles(live / 'source-fresh', state['fresh_seeds'], state['transition_sha256'])
    result['source_fresh'] = source
    finals = []
    for arm in state['arms']:
        name = arm['arm']
        folder = live / name
        saved, manifest = load_checked(folder / 'checkpoint', arm['final_checkpoint_sha256'])
        validate_horizon_state(saved)
        assert saved['training_gamma'] == arm['gamma']
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1703936
        assert saved['world_control_steps'] == 1081344
        assert int(saved['optimizer']['actor_adam'][0].count) == 9096
        assert int(saved['optimizer']['critic_adam'][0].count) == 32768
        assert sum(saved['neutral_credits_by_corner']) == 65536
        assert all(n >= old for n, old in zip(saved['neutral_credits_by_corner'], canonical['neutral_credits_by_corner']))
        assert saved['bc_sampler_state'] == canonical['bc_sampler_state']
        checks = {}
        for sample, seeds in (('screen', [7, 23, 31, 47]), ('fresh', state['fresh_seeds'])):
            checks[sample], count = full_profiles(folder / sample, seeds, manifest['sha256'])
            controls += count
            shutil.copytree(folder / sample, report / name / sample, dirs_exist_ok=True)
        assert not any(c['passes_complete'] for c in checks.values())
        out = models / name
        out.mkdir(parents=True, exist_ok=True)
        for filename in (manifest['file'], 'latest.json'):
            shutil.copy2(folder / 'checkpoint' / filename, out / filename)
        shutil.copy2(folder / 'config.json', out / 'config.json')
        for filename in ('progress.json', 'state.json', 'initial-block-verified.json'):
            shutil.copy2(folder / filename, report / name / filename)
        metrics = [json.loads(line) for label in ('initial-block', 'remaining') for line in (folder / (label + '.log')).read_text().splitlines() if line.strip()]
        assert len(metrics) == 32 and all(np.isfinite(r['mean_loss']) for r in metrics)
        assert [r['optimized_steps'] for r in metrics] == list(range(1672192, 1703937, 1024))
        (report / name / 'training-metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics))
        result['arms'].append(dict(arm=name, sha256=manifest['sha256'], checks=checks,
            actor_adam_count=9096, critic_adam_count=32768,
            neutral_credits_by_corner=saved['neutral_credits_by_corner']))
        finals.append(saved)
    assert controls == state['diagnostic_evaluation_world_controls'] == 58073
    for key in ('retention_sampler_state', 'aux_sampler_state'):
        assert finals[0][key] == finals[1][key]
    result['auxiliary_sampler_streams_match'] = True
    result['decision'] = ('Neither arm passes. Gamma 0.999 increases surviving active strikes to four fresh seeds, '
        'including one two-way episode, versus zero control fresh strikes. Permit one 32768-credit diagnostic '
        'ordinary continuation from the latest gamma-0.999 final only, retaining its complete profile and optimizer. '
        'Check full screens after 16384/32768 and fresh seeds at the end; stop on severe survival regression. '
        'No further automatic extension, main adoption or promotion.')
    shutil.copy2(live / 'comparison.json', report / 'completed-comparison.json')
    atomic_json(report / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
