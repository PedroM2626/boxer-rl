"""Zero-simulation retrospective GAE diagnostics on verified deterministic traces.

These are off-policy observations, not a recovered stochastic PPO training batch.
The direct own-fall component excludes learned critic predictions by construction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_train import Critic, advantages, normalize, require_ppo_checkpoint
from tools.striking_contact_compare import load_checked


def fragmented_advantages(rewards, dones, values, following, lam, unroll=128):
    """Diagnostic fragments start at reset; actual training boundaries differ."""
    result = np.zeros_like(rewards)
    for start in range(0, len(rewards), unroll):
        end = min(start + unroll, len(rewards))
        result[start:end], _ = advantages(rewards[start:end], dones[start:end],
            values[start:end], following[end-1], gamma=.999, lam=lam)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Terminal-credit audit exists; inspect before repeating')
    root = Path('docs/results/striking-pilot/kl-failure-trace-1966080')
    review = json.loads((root/'analysis.json').read_text())
    assert review['status'] == 'reviewed' and review['episodes'] == 72
    assert review['formal_world_controls_total'] == 1666653
    critic = Critic()
    predict = jax.jit(lambda params, obs: critic.apply(params, obs))
    models = {}
    rows = []
    for check in review['checks']:
        variant, profile = check['name'].split('-', 1)
        if variant not in models:
            folder = Path('checkpoints/striking-validation') / (
                'sampling-1933312/control/gate-16384' if variant == 'source'
                else 'kl-guard-1949696/'+variant+'/gate-16384')
            manifest = json.loads((folder/'latest.json').read_text())
            saved, _ = load_checked(folder, manifest['sha256'])
            require_ppo_checkpoint(saved)
            config = json.loads((folder/'config.json').read_text())
            assert config['gamma'] == .999 and config['gae_lambda'] == .95 and config['unroll'] == 128
            models[variant] = saved, manifest['sha256']
        saved, sha = models[variant]
        report = json.loads((root/check['name']/'evaluation.json').read_text())
        assert report['striking_sha256'] == sha
        for trial in report['trials']:
            path = root/check['name']/'traces'/Path(trial['trace_file']).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == trial['trace_sha256']
            with np.load(path) as data:
                assert all(np.isfinite(x).all() for x in data.values())
                reward = data['reward_base_total']
                fallen = data['fallen'].astype(bool)
                direct = -5. * fallen.astype(float)
                # First fall cuts both learners' bootstrap; timeout does so too.
                done = np.repeat(fallen.any(axis=1)[:, None], 2, axis=1)
                done[-1] = True
                values = np.asarray(predict(saved['critic'], jp.array(normalize(
                    data['observation'].reshape(-1,148), saved['norm'])))).reshape(-1,2)
                following = np.asarray(predict(saved['critic'], jp.array(normalize(
                    data['next_observation'].reshape(-1,148), saved['norm'])))).reshape(-1,2)
                assert np.allclose(data['reward_fall'], direct)
                assert len(reward) <= 500 and not fallen[:-1].any()
                eligible = (0,1) if profile == 'selfplay' else (int(profile[-1]),)
                row = dict(variant=variant, profile=profile, seed=trial['seed'], swap=trial['swap'],
                    sha256=sha, trace_sha256=trial['trace_sha256'], controls=len(reward),
                    survived=trial['survived'], retrospective_only=True, lambdas={})
                for lam in (.95,.99):
                    adv = fragmented_advantages(reward, done, values, following, lam)
                    terminal = fragmented_advantages(direct, done, np.zeros_like(values),
                        np.zeros_like(following), lam)
                    assert np.isfinite(adv).all() and np.isfinite(terminal).all()
                    corners = []
                    for corner in eligible:
                        samples = {}
                        if fallen[-1,corner]:
                            for lag in (0,10,25,50,100):
                                index = len(reward)-1-lag
                                if index >= 0:
                                    samples[str(lag)] = dict(
                                        direct_terminal_component=float(terminal[index,corner]),
                                        total_gae=float(adv[index,corner]),
                                        crosses_fragment_boundary=index//128 != (len(reward)-1)//128,
                                        pelvis_height=float(data['pelvis_height'][index,corner]))
                        corners.append(dict(corner=corner, own_fall=bool(fallen[-1,corner]),
                            mean_abs_gae=float(np.abs(adv[:,corner]).mean()),
                            terminal_credit_at_lags=samples))
                    row['lambdas'][str(lam)] = corners
                rows.append(row)
    assert len(rows) == 72
    result = dict(status='completed', new_world_controls=0, new_optimizer_updates=0,
        new_supervised_updates=0, diagnostic_ppo_total_retained=2016512,
        formal_world_controls_total_retained=1666653, adoption=False, promotion=False,
        analytic_direct_kernel={str(lam):{str(lag):(.999*lam)**lag for lag in (10,25,50,100,128)}
            for lam in (.95,.99)},
        half_life_controls={str(lam):float(np.log(.5)/np.log(.999*lam)) for lam in (.95,.99)},
        trials=rows,
        limitations='Deterministic evaluation traces are off-policy. Reset-aligned 128-control fragments differ from actual staggered training boundaries. Direct terminal-reward attenuation is not the total effective credit horizon: the learned critic can propagate value across fragments and earlier TD errors. Lambda .99 raises variance and is an offline counterfactual only; neither lower attenuation nor these traces establish causal benefit or authorize adoption.')
    atomic_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'trials'}, indent=2))


if __name__ == '__main__':
    main()
