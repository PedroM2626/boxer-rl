"""Passive per-minibatch PPO measurements for one explicitly bounded diagnostic block."""
import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_policy_diagnostics import gaussian_kl


class UpdateAudit:
    def __init__(self, actor, log_probability, folder):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder
        self.predict = jax.jit(lambda params, obs: actor.apply(params,obs))
        self.logp = jax.jit(log_probability)
        self.records = []

    def begin(self, params, batch, mode_labels):
        self.obs = jp.array(batch['obs']); self.raw = jp.array(batch['raw'])
        self.old_logp = np.asarray(batch['logp'],dtype=np.float64)
        self.mode_labels = np.asarray(mode_labels)
        if self.mode_labels.shape != self.old_logp.shape:
            raise ValueError('one mode label per actual learner rollout row required')
        self.baseline = tuple(np.asarray(x,dtype=np.float64) for x in self.predict(params,self.obs))
        self.observe(params, -1, -1,phase='baseline')

    def observe(self, params, epoch, minibatch, phase='optimizer', loss=None):
        mean, std = self.predict(params,self.obs)
        exact = gaussian_kl(*self.baseline,np.asarray(mean),np.asarray(std))
        log_ratio = np.asarray(self.logp(self.raw,mean,std),dtype=np.float64)-self.old_logp
        ratio = np.exp(log_ratio)
        row = dict(epoch=epoch,minibatch=minibatch,phase=phase,loss=loss,profiles={})
        for name, mask in (('all',np.ones(len(log_ratio),dtype=bool)),
                ('neutral',self.mode_labels==0),('active',self.mode_labels==1)):
            if not np.any(mask): continue
            kl = exact[mask].sum(axis=1)
            row['profiles'][name] = dict(rows=int(mask.sum()),exact_gaussian_kl_mean=float(kl.mean()),
                exact_gaussian_kl_p95=float(np.quantile(kl,.95)),
                command_kl_mean=float(exact[mask,:3].sum(axis=1).mean()),
                arm_kl_mean=float(exact[mask,3:].sum(axis=1).mean()),
                sampled_approx_kl=float(np.mean(ratio[mask]-1.-log_ratio[mask])),
                ratio_clip_fraction=float(np.mean(np.abs(ratio[mask]-1.)>.2)))
            if not all(np.isfinite(v) for v in row['profiles'][name].values()):
                raise FloatingPointError('nonfinite PPO update diagnostic')
        self.records.append(row)
        atomic_json(self.folder/'minibatch-measurements.json',dict(read_only=True,
            changes_training_objective=False,rollout_rows=len(log_ratio),records=self.records,
            note='Exact Gaussian KL and sampled ratios on the same collected learner observations; no trust-region stop or optimizer change.'))

    def after_normalizer(self,params,normalized_observations):
        self.obs=jp.array(normalized_observations)
        self.observe(params,4,-1,phase='after_normalizer')
