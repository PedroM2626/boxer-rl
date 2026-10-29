"""Original split-Adam PPO update/data ratio over an explicit larger rollout."""
import copy
import jax
import jax.numpy as jp
import numpy as np
from boxer.striking_train import (Actor, Critic, normalize, advantages,
    update_norm, make_ppo_update, require_ppo_checkpoint)
from boxer.striking_sampling_profile import sampling_offset
from boxer.striking_ppo_warmstart import SplitAdam
from boxer.striking_anchor import load_anchor_candidates
from boxer.striking_retention import load_retention_data, retention_batch, validate_retention_state


class BatchedPPOUpdate:
    def __init__(self, saved, config):
        require_ppo_checkpoint(saved)
        if (config['learning_rate'], config['epochs'], config['gamma'], config['gae_lambda'],
                config['anchor_loss_weight'], config['retention_loss_weight']) != (.0001, 4, .999, .99, 0., 0.):
            raise ValueError('unchanged v17 training coefficients required')
        self.actor = Actor(sampling_log_std_offset=sampling_offset(saved)); self.critic = Critic()
        self.update = make_ppo_update(self.actor, self.critic, SplitAdam(), 0., 0.)
        self.anchors, manifest = load_anchor_candidates()
        assert manifest['sha256'] == saved['anchor_dataset_sha256']
        self.retention, self.provenance = load_retention_data()
        validate_retention_state(saved, self.provenance)

    def apply(self, saved, rollout, bootstrap, key_after_sampling):
        """No simulation/new learner data; replayable from rollout and source."""
        ap, vp, optimizer, norm = (saved[k] for k in ('actor', 'critic', 'optimizer', 'norm'))
        flat = rollout['obs'].reshape(-1, 148); block = len(flat)
        if block % 256: raise ValueError('complete 256-row minibatches required')
        aux = np.random.default_rng(); aux.bit_generator.state = copy.deepcopy(saved['aux_sampler_state'])
        retention = np.random.default_rng(); retention.bit_generator.state = copy.deepcopy(saved['retention_sampler_state'])
        adv, returns = advantages(rollout['reward'], rollout['done'], rollout['value'], bootstrap, gamma=.999, lam=.99)
        batch = dict(obs=normalize(flat, norm), raw=rollout['raw'].reshape(block, 17),
            logp=rollout['logp'].reshape(block), adv=((adv-adv.mean())/(adv.std()+1e-8)).reshape(block),
            returns=returns.reshape(block))
        key = key_after_sampling; losses = []
        for _ in range(4):
            key, sub = jax.random.split(key)
            order = np.asarray(jax.random.permutation(sub, block))
            for indices in np.split(order, block//256):
                mini = {k: jp.asarray(v[indices]) for k, v in batch.items()}
                ai = aux.integers(0, len(self.anchors['observation']), 64)
                mini.update(anchor_obs=jp.asarray(normalize(self.anchors['observation'][ai], norm)),
                            anchor_labels=jp.asarray(self.anchors['action'][ai]))
                mini.update(retention_batch(self.retention, retention, norm))
                ap, vp, optimizer, loss = self.update(ap, vp, optimizer, mini)
                losses.append(float(loss))
        if not np.isfinite(losses).all() or not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((ap, vp, optimizer))):
            raise FloatingPointError('nonfinite PPO update; retain prior prefix and all attempted data')
        return dict(actor=ap, critic=vp, optimizer=optimizer, norm=update_norm(norm, flat), key=key,
            aux_sampler_state=aux.bit_generator.state, retention_sampler_state=retention.bit_generator.state), losses
