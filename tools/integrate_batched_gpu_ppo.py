"""One predeclared 1,024-credit warm-start integration, never an open-ended trainer."""
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import numpy as np
from boxer.batched_striking_env import BatchedStrikingEnv
from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.result_archive import archive_tree
from boxer.striking_gpu_profile import SOURCE, SOURCE_SHA, PROFILE, prepare_gpu_state
from boxer.striking_train import (Actor, Critic, normalize, log_probability, advantages,
    make_ppo_update, update_norm, require_ppo_checkpoint)
from boxer.striking_ppo_warmstart import SplitAdam
from boxer.striking_sampling_profile import sampling_offset
from boxer.striking_anchor import load_anchor_candidates
from boxer.striking_retention import load_retention_data, retention_batch, validate_retention_state
from boxer.striking_event_evidence import save_completed_evidence
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

OUT = Path('/home/pedro/boxing-runs/gpu-batched-integration-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-batched-integration-v1')
MODELS = Path('checkpoints/striking-validation/gpu-batched-integration-v1')
GLOBAL_BEFORE = 9504000
BLOCK = 1024


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (OUT/'.lock', Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock')):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'progress.json').exists(): raise ValueError('Existing integration requires consumed-state review, never silent retry')
        mechanics = json.loads(Path('/home/pedro/boxing-runs/batched-gpu-mechanics-v1/attempt-2/progress.json').read_text())
        bench = json.loads(Path('/home/pedro/boxing-runs/batched-gpu-benchmark-v1/progress.json').read_text())
        assert mechanics['status'] == bench['status'] == 'completed' and len(mechanics['checks']) == 5
        assert [r['worlds'] for r in bench['cases']] == [8, 128]
        original, manifest = load_checked(Path(SOURCE), SOURCE_SHA)
        initial = prepare_gpu_state(original, SOURCE_SHA)
        config = json.loads((Path(SOURCE)/'config.json').read_text())
        assert config['worlds'] == 8 and config['unroll'] == 128 and not original.get('kl_guard_ppo')
        assert config['anchor_loss_weight'] == config['retention_loss_weight'] == 0.
        assert (config['learning_rate'], config['epochs'], config['minibatches'],
                config['gamma'], config['gae_lambda']) == (.0001, 4, 4, .999, .99)
        config = dict(config, collection_backend='batched_mujoco_warp', gpu_collection_profile=PROFILE)
        state = dict(status='running', phase='initializing', started_utc=utc(), pid=os.getpid(),
            source_sha256=SOURCE_SHA, maximum_new_ppo_steps=BLOCK, new_ppo_steps=0,
            diagnostic_ppo_before=GLOBAL_BEFORE, diagnostic_ppo_ceiling=GLOBAL_BEFORE+BLOCK,
            diagnostic_ppo_total=GLOBAL_BEFORE, consumed_steps=original['consumed_steps'],
            optimized_steps=original['optimized_steps'], world_control_steps=original['world_control_steps'],
            main_retained=476041216, formal_retained=2298509, new_formal_controls=0,
            optimizer_reset=False, physics_state_saved=False, prerequisites=verify_prerequisites(), engine_ledger={})
        atomic_json(OUT/'progress.json', state)
        cp = OUT/'checkpoint'; cp.mkdir(exist_ok=True)
        def save_checkpoint(saved, name):
            require_ppo_checkpoint(saved)
            payload = pickle.dumps(jax.device_get(saved), protocol=pickle.HIGHEST_PROTOCOL)
            atomic_bytes(cp/name, payload)
            metadata = dict(manifest, file=name, sha256=hashlib.sha256(payload).hexdigest(),
                consumed_steps=saved['consumed_steps'], optimized_steps=saved['optimized_steps'],
                world_control_steps=saved['world_control_steps'], gpu_batched_ppo=True,
                gpu_collection_profile=saved['gpu_collection_profile'], gpu_collection_transition=saved['gpu_collection_transition'])
            atomic_json(cp/'latest.json', metadata)
            return metadata
        initial_manifest = save_checkpoint(initial, 'initial.pkl')
        state['initial_sha256'] = initial_manifest['sha256']
        atomic_json(OUT/'config.json', config)
        begun = time.perf_counter()
        try:
            foundation = UnassistedG1('approach', impl='jax')
            motor, motor_manifest = load_policy_params('checkpoints/curriculum/approach')
            assert motor_manifest['sha256'] == original['foundation_sha256']
            ap, vp, optimizer, norm, key = (initial[k] for k in ('actor', 'critic', 'optimizer', 'norm', 'key'))
            actor = Actor(sampling_log_std_offset=sampling_offset(initial)); critic = Critic()
            tx = SplitAdam()
            update = make_ppo_update(actor, critic, tx, 0., 0.)
            @jax.jit
            def sample(weights, obs, rng):
                mean, std = actor.apply(weights, obs)
                raw = mean+jp.exp(std)*jax.random.normal(rng, mean.shape)
                return jp.tanh(raw), raw, log_probability(raw, mean, std)
            predict = jax.jit(lambda weights, obs: jp.tanh(actor.apply(weights, obs)[0]))
            value = jax.jit(critic.apply)
            anchors, anchor_manifest = load_anchor_candidates()
            retention, provenance = load_retention_data()
            assert anchor_manifest['sha256'] == initial['anchor_dataset_sha256']
            validate_retention_state(initial, provenance)
            aux_rng = np.random.default_rng(); aux_rng.bit_generator.state = copy.deepcopy(initial['aux_sampler_state'])
            retention_rng = np.random.default_rng(); retention_rng.bit_generator.state = copy.deepcopy(initial['retention_sampler_state'])
            def account(ledger):
                state['engine_ledger'] = ledger
                assert ledger['gpu_world_substeps'] <= 10248
                atomic_json(OUT/'progress.json', state)
            env = BatchedStrikingEnv(foundation, motor, worlds=8,
                engagement_distance=config['engagement_distance'],
                engagement_reward_distance=config['engagement_reward_distance'], account=account)
            episode = np.full(8, initial['consumed_steps']//8, np.int64)
            seed = config['seed']+1009*np.arange(8)+episode
            swaps = np.arange(8) % 2 == 1
            obs = env.reset(np.arange(8), seed, swaps)
            corner = seed % 2
            active = np.isin(np.arange(8), config['active_world_indices'])
            mixed = np.asarray(initial['mixed_credits_by_mode_and_corner'], np.int64).copy()
            storage = {k: [] for k in ('obs', 'raw', 'logp', 'reward', 'done', 'value')}
            collection_start = time.perf_counter(); state['phase'] = 'collecting'
            for control in range(128):
                own = obs[np.arange(8), corner]
                normalized = normalize(own, norm)
                key, sub = jax.random.split(key)
                action, raw, logp = sample(ap, jp.asarray(normalized), sub)
                predicted = np.asarray(value(vp, jp.asarray(normalized)))
                other = obs[np.arange(8), 1-corner]
                opponent = np.asarray(predict(initial['frozen_opponent']['actor'],
                    jp.asarray(normalize(other, initial['frozen_opponent']['norm']))))
                paired = np.zeros((8, 2, 17), np.float32)
                paired[np.arange(8), corner] = np.asarray(action)
                paired[np.flatnonzero(active), 1-corner[active]] = opponent[active]
                # A CUDA batch is one concurrent attempted group. Debit all
                # submitted learner/world transitions before physics, never refund.
                state['new_ppo_steps'] += 8
                state['diagnostic_ppo_total'] = GLOBAL_BEFORE+state['new_ppo_steps']
                state['consumed_steps'] += 8; state['world_control_steps'] += 8
                np.add.at(mixed, (active.astype(int), corner), 1)
                atomic_json(OUT/'progress.json', state)
                following, reward, done, outcome = env.step(paired, optimized_steps=initial['optimized_steps'])
                state['physics_health'] = dict(env.health_summary)
                selected_reward = reward[np.arange(8), corner]
                for world, monitor in enumerate(env.monitors):
                    save_completed_evidence(monitor, OUT/'event-evidence', cp, consumed=state['consumed_steps'],
                        world=world, learner_corner=int(corner[world]), reset_seed=int(seed[world]), swap=bool(swaps[world]),
                        active_opponent=bool(active[world]), foundation_sha256=original['foundation_sha256'])
                for name, data in dict(obs=own.copy(), raw=np.asarray(raw), logp=np.asarray(logp),
                    reward=selected_reward.copy(), done=done.astype(np.float32), value=predicted).items():
                    storage[name].append(data)
                rows = np.flatnonzero(done)
                if len(rows):
                    episode[rows] += 1
                    seed[rows] = config['seed']+1009*rows+episode[rows]
                    swaps[rows] = episode[rows] % 2 == 1
                    following = env.reset(rows, seed[rows], swaps[rows])
                    corner[rows] = seed[rows] % 2
                obs = following
            state['collection_seconds_including_initial_jit'] = time.perf_counter()-collection_start
            state['phase'] = 'updating'; atomic_json(OUT/'progress.json', state)
            rollout = {k: np.asarray(v) for k, v in storage.items()}
            buffer = io_bytes(rollout); atomic_bytes(OUT/'rollout.npz', buffer)
            bootstrap = np.asarray(value(vp, jp.asarray(normalize(obs[np.arange(8), corner], norm))))
            adv, returns = advantages(rollout['reward'], rollout['done'], rollout['value'], bootstrap,
                gamma=config['gamma'], lam=config['gae_lambda'])
            flat = rollout['obs'].reshape(BLOCK, 148)
            batch = dict(obs=normalize(flat, norm), raw=rollout['raw'].reshape(BLOCK, 17),
                logp=rollout['logp'].reshape(BLOCK), adv=((adv-adv.mean())/(adv.std()+1e-8)).reshape(BLOCK),
                returns=returns.reshape(BLOCK))
            update_start = time.perf_counter(); losses = []
            for _ in range(4):
                key, sub = jax.random.split(key)
                order = np.asarray(jax.random.permutation(sub, BLOCK))
                for indices in np.split(order, 4):
                    mini = {k: jp.asarray(v[indices]) for k, v in batch.items()}
                    ai = aux_rng.integers(0, len(anchors['observation']), 64)
                    mini.update(anchor_obs=jp.asarray(normalize(anchors['observation'][ai], norm)),
                                anchor_labels=jp.asarray(anchors['action'][ai]))
                    mini.update(retention_batch(retention, retention_rng, norm))
                    ap, vp, optimizer, loss = update(ap, vp, optimizer, mini)
                    losses.append(float(loss))
            if not np.isfinite(losses).all() or not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((ap, vp, optimizer))):
                raise FloatingPointError('nonfinite integration update; retained source remains authoritative')
            state['ppo_update_seconds_including_initial_jit'] = time.perf_counter()-update_start
            final = dict(initial, actor=ap, critic=vp, optimizer=optimizer, norm=update_norm(norm, flat), key=key,
                consumed_steps=state['consumed_steps'], optimized_steps=initial['optimized_steps']+BLOCK,
                world_control_steps=state['world_control_steps'], mixed_credits_by_mode_and_corner=mixed.tolist(),
                aux_sampler_state=aux_rng.bit_generator.state, retention_sampler_state=retention_rng.bit_generator.state)
            validate_retention_state(final, provenance); require_ppo_checkpoint(final)
            final_manifest = save_checkpoint(final, f"step-{final['consumed_steps']}.pkl")
            state.update(status='completed', phase='awaiting_independent_review', optimized_steps=final['optimized_steps'],
                final_sha256=final_manifest['sha256'], actor_adam_count=int(optimizer['actor_adam'][0].count),
                critic_adam_count=int(optimizer['critic_adam'][0].count), mean_loss=float(np.mean(losses)),
                elapsed_seconds=time.perf_counter()-begun, completed_utc=utc(), decision='Review integration; no larger allocation or native promotion implied')
        except BaseException as error:
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(OUT/'progress.json', state)
            archive_tree(OUT, ARCHIVE)
            archive_tree(cp, MODELS)
            atomic_json(MODELS/'config.json', config)


def io_bytes(arrays):
    import io
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


if __name__ == '__main__': main()
