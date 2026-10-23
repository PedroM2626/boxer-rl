"""Read-only Gaussian policy/normalizer drift on a fixed, hashed archived-state corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_policy_diagnostics import gaussian_kl
from boxer.striking_retention import load_retention_data
from boxer.striking_train import Actor, normalize, require_ppo_checkpoint


def kl_summary(a, b):
    parts = gaussian_kl(a[0], a[1], b[0], b[1])
    total = parts.sum(axis=-1)
    return dict(mean=float(total.mean()), median=float(np.median(total)), p95=float(np.quantile(total,.95)),
        p99=float(np.quantile(total,.99)), maximum=float(total.max()),
        command_mean=float(parts[:,:3].sum(axis=-1).mean()), arm_mean=float(parts[:,3:].sum(axis=-1).mean()),
        bounded_mean_action_rms=float(np.sqrt(np.mean((np.tanh(a[0])-np.tanh(b[0]))**2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Drift audit exists; inspect before rerunning')
    state = json.loads((args.input / 'continuation.json').read_text())
    assert state['status'] == 'completed' and state['diagnostic_consumed_steps'] == 65536
    data, provenance = load_retention_data(); corpus = {}; manifest = {}
    for name, key, label in (('demonstrations','demo_obs','demo_labels'),('motor_anchors','anchor_obs','anchor_labels')):
        indices = np.linspace(0,len(data[key])-1,1024,dtype=int)
        corpus[name] = (data[key][indices], data[label][indices])
        manifest[name] = dict(original_rows=len(data[key]), selected_rows=1024,
            indices=indices.tolist(), selected_observation_sha256=hashlib.sha256(corpus[name][0].tobytes()).hexdigest())
    rows = []; labels = []; trace_files = []
    root = Path('docs/results/striking-pilot/horizon-regression-trace-1736704')
    for profile in ('final-selfplay','final-neutral-1'):
        report = json.loads((root / profile / 'evaluation.json').read_text())
        for trial in report['trials']:
            path = root / profile / 'traces' / Path(trial['trace_file']).name
            payload = path.read_bytes(); assert hashlib.sha256(payload).hexdigest() == trial['trace_sha256']
            with np.load(path) as arrays:
                obs = arrays['observation'].reshape(-1,148); action = arrays['action'].reshape(-1,17)
                indices = np.linspace(0,len(obs)-1,64,dtype=int)
                rows.append(obs[indices]); labels.append(action[indices])
            trace_files.append(dict(file=str(path),sha256=trial['trace_sha256'],indices=indices.tolist()))
    corpus['archived_contact_states'] = (np.concatenate(rows),np.concatenate(labels))
    manifest['archived_contact_states'] = dict(selected_rows=len(corpus['archived_contact_states'][0]),
        files=trace_files, selected_observation_sha256=hashlib.sha256(corpus['archived_contact_states'][0].tobytes()).hexdigest())
    actor = Actor(); predict = jax.jit(lambda params,x: actor.apply(params,x))

    def predictions(saved, actor_params=None, norm=None):
        return {name:tuple(np.asarray(x,dtype=np.float64) for x in predict(
            saved['actor'] if actor_params is None else actor_params,
            jp.array(normalize(obs,saved['norm'] if norm is None else norm))))
            for name,(obs,_) in corpus.items()}

    paths = sorted((args.input / 'checkpoint').glob('step-*.pkl'), key=lambda p:int(p.stem.split('-')[-1]))
    assert len(paths) == 65 and int(paths[0].stem.split('-')[-1]) == 1769472
    source = pickle.loads(paths[0].read_bytes()); require_ppo_checkpoint(source)
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == state['source_sha256']
    baseline = predictions(source); previous = baseline
    checkpoints = []; endpoints = {}
    for i, path in enumerate(paths):
        saved = pickle.loads(path.read_bytes()); require_ppo_checkpoint(saved)
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1769472+i*1024
        current = baseline if i == 0 else predictions(saved)
        checkpoints.append(dict(consumed_steps=saved['consumed_steps'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            local_kl={name:kl_summary(previous[name],current[name]) for name in corpus},
            cumulative_kl={name:kl_summary(baseline[name],current[name]) for name in corpus}))
        if i in (0,32,64):
            parameters_only = predictions(saved,norm=source['norm'])
            norm_only = predictions(saved,actor_params=source['actor'])
            endpoints[str(saved['consumed_steps'])] = dict(normalizer_count=float(saved['norm']['count']),
                log_std=current['demonstrations'][1].tolist(),
                parameter_only_kl={name:kl_summary(baseline[name],parameters_only[name]) for name in corpus},
                normalizer_only_kl={name:kl_summary(baseline[name],norm_only[name]) for name in corpus},
                action_reference_mse={name:float(np.mean((np.tanh(current[name][0])-labels)**2))
                    for name,(_,labels) in corpus.items()})
        previous = current
    result = dict(status='completed', new_world_controls=0, new_optimizer_updates=0,
        new_supervised_updates=0, diagnostic_ppo_total_retained=1810688,
        formal_world_controls_total_retained=989355, main_adoption=False, promotion=False,
        corpus=manifest, retention_provenance=provenance, checkpoints=checkpoints, endpoints=endpoints,
        limitations='Fixed archived states are off-policy probes. KL is exact between Gaussian policies on these inputs, '
            'not the actual rollout minibatch KL or a causal explanation of falls. Decomposition holds actor or '
            'normalizer fixed without changing trajectories. Demonstration/anchor MSE does not certify physical boxing.')
    atomic_json(args.output,result)
    print(json.dumps(dict(endpoints=endpoints,maximum_local_mean_kl={name:max(c['local_kl'][name]['mean'] for c in checkpoints) for name in corpus}),indent=2))


if __name__ == '__main__':
    main()
