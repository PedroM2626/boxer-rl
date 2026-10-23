"""Read-only policy ablations with unchanged free-body physics and impact rules.

Interventions are diagnostic actions, never replacement policies or training.
The sampled arm uses the saved Gaussian scale and a separate diagnostic RNG.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, load_policy_params, utc
from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.checkpoint / 'latest.json').read_text())
    payload = (args.checkpoint / manifest['file']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    saved = pickle.loads(payload)
    if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
            tuple(saved[k] for k in ('actor', 'critic', 'optimizer', 'norm', 'key')))):
        raise FloatingPointError('nonfinite checkpoint')
    foundation = UnassistedG1('approach', impl='jax')
    params, source = load_policy_params(Path('checkpoints/curriculum/approach'))
    if contract(foundation) != source['contract'] or saved['foundation_sha256'] != source['sha256']:
        raise ValueError('foundation provenance mismatch')
    env = StrikingEnv(foundation, params,
        engagement_distance=manifest['engagement_distance'],
        engagement_reward_distance=manifest['engagement_reward_distance'])
    actor = Actor()
    policy = jax.jit(lambda obs: actor.apply(saved['actor'], obs))
    results = []
    transitions = 0
    for mode in ('sampled', 'zero_navigation', 'zero_arms'):
        for seed in (7, 23, 31, 47):
            for swap in (False, True):
                obs = env.reset(seed, swap)
                rng = np.random.default_rng(seed + 1009 * int(swap) + 524288)
                hits = np.zeros(2)
                raw_impulse = np.zeros(2)
                commands = []
                distances = []
                for step in range(500):
                    mean, log_std = map(np.asarray, policy(jp.array(normalize(obs, saved['norm']))))
                    action = np.tanh(mean + np.exp(log_std) * rng.normal(size=mean.shape)) if mode == 'sampled' else np.tanh(mean)
                    if mode == 'zero_navigation':
                        action[:, :3] = 0
                    if mode == 'zero_arms':
                        action[:, 3:] = 0
                    commands.append(env.commands(action))
                    obs, _, done, outcome = env.step(action)
                    transitions += 1
                    hits += outcome['scored']
                    raw_impulse += outcome['legal_contact_impulse']
                    distances.append(outcome['distance'])
                    if done:
                        break
                row = dict(mode=mode, seed=seed, swap=swap, survived=not env.pair.done,
                    elapsed=env.pair.elapsed, finite=bool(np.isfinite(obs).all()),
                    scored_by_corner=hits.tolist(), raw_legal_impulse=raw_impulse.tolist(),
                    fallen=outcome['fallen'].tolist(), final_distance=distances[-1],
                    minimum_distance=min(distances),
                    final_second_mean_commands=np.mean(commands[-50:], axis=0).tolist())
                results.append(row)
                print(json.dumps(row), flush=True)
    atomic_json(args.output, dict(kind='striking_behavior_diagnostic', updated_utc=utc(),
        checkpoint_sha256=manifest['sha256'], foundation_sha256=source['sha256'],
        assistance=False, scorecard_version=1, certification=False,
        engagement_distance=env.engagement_distance,
        simulated_world_transitions=transitions, optimization_transitions=0, trials=results))


if __name__ == '__main__':
    main()
