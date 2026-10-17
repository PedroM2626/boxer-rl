"""Probe the brax PPO networks the MJX arms trained with, so the warp trainer
can replicate the parameterisation rather than guess it from source fragments.

    JAX_PLATFORMS=cpu /home/pedro/venvs/boxer313/bin/python tools/brax_ppo_probe.py

Reports: the distribution class and parameter layout, what the rollout policy
actually hands the env (tanh-bounded or not), and the log-prob the loss
recomputes from the stored action. Findings encoded in boxer/warp_train.py:
NormalTanhDistribution with param_size = 2 x action_size (loc ∥ raw scale,
scale = softplus(raw) + 0.001), actions = tanh(sample), the Normal evaluated
at the stored post-tanh action, plain MLPs with silu and a 58-wide final Dense.
"""
import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

OBS, ACT = 128, 29

net = ppo_networks.make_ppo_networks(
    (OBS,), ACT,
    policy_hidden_layer_sizes=(128, 128, 128, 128),
    value_hidden_layer_sizes=(256, 256, 256, 256, 256),
    preprocess_observations_fn=lambda x, stats: x,
)
dist = net.parametric_action_distribution
print("distribution:", type(dist).__name__, "param_size:", dist.param_size,
      "-> loc/scale halves of", dist.param_size // 2)

params = net.policy_network.init(jax.random.PRNGKey(0))
print("policy tree:", jax.tree_util.tree_map(lambda x: x.shape, params))
out = net.policy_network.apply(None, params, jp.zeros((1, OBS)))
print("policy output (concatenated loc ∥ raw scale):", np.asarray(out).shape)

v_params = net.value_network.init(jax.random.PRNGKey(1))
print("value tree:", jax.tree_util.tree_map(lambda x: x.shape, v_params))

# What the rollout hands the env, and how wide it ranges.
inference = ppo_networks.make_inference_fn(net)(params, deterministic=False)
obs = jp.zeros((64, OBS))
keys = jax.random.split(jax.random.PRNGKey(2), 64)
acts, extras = jax.vmap(inference, in_axes=(0, 0))(obs, keys)
a = np.asarray(acts)
print(f"action shape {a.shape}, min {a.min():.3f}, max {a.max():.3f} "
      "(bounded by tanh)")

# The formula the loss recomputes from the stored (post-tanh) action.
lp = dist.log_prob(out, jp.asarray(a[:1]))
print("log_prob at the stored action:", np.asarray(lp).shape,
      float(np.asarray(lp).mean()))
ent = dist.entropy(out, jax.random.PRNGKey(3))
print("entropy term:", float(np.asarray(ent).mean()))
print("extras keys:", sorted(extras.keys()) if isinstance(extras, dict)
      else type(extras))
