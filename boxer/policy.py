"""Policies that can drive a boxer: trained checkpoints and scripted opponents.

Everything here shares one call signature::

    action = policy(obs, time, key)

where ``obs`` is a batch of egocentric observations with shape ``(batch, D)``,
``time`` is the per-batch-element simulation clock (shape ``(batch,)``), and
``key`` is a PRNG key. The clock is part of the signature because a scripted
opponent has to derive its rhythm from something, and it must stay a pure
function of its inputs to survive ``jit``; trained policies simply ignore it.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jp
from brax.training import types as training_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

from boxer import scene

# Kwargs that Brax's own checkpoint reader cannot round-trip: it looks every
# stored kernel-initialiser name up in KERNEL_INITIALIZER, and the default saved
# config carries ``mean_kernel_init_fn: null``, which raises KeyError.
INITIALIZER_KEYS = (
    "policy_network_kernel_init_fn", "policy_network_kernel_init_kwargs",
    "value_network_kernel_init_fn", "value_network_kernel_init_kwargs",
    "mean_kernel_init_fn", "mean_kernel_init_kwargs",
)

ACTIVATIONS = {
    "elu": linen.elu,
    "gelu": linen.gelu,
    "relu": linen.relu,
    "selu": linen.selu,
    "silu": linen.silu,
    "swish": linen.silu,
    "tanh": linen.tanh,
}

#: ``obs`` is ``(batch, D)``, ``time`` is ``(batch,)`` simulated seconds, and the
#: result is ``(batch, action_size)`` in normalised ``[-1, 1]`` actuator units.
Policy = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def resolve_checkpoint(path: str | Path) -> Path:
    """Accepts either Brax's checkpoint root or a specific step directory.

    ``ppo_checkpoint.load`` wants the step directory that holds the network
    config, but the thing a run leaves behind and a person types is the root, so
    pick the highest numbered step.
    """
    path = Path(path)
    if (path / "ppo_network_config.json").exists():
        return path.resolve()
    steps = sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise SystemExit(f"no orbax step directory under {path}")
    return steps[-1].resolve()


def load_policy(
    checkpoint: str | Path,
    deterministic: bool = True,
    ignore_time: bool = True,
):
    """Rebuilds an inference policy from a Brax PPO checkpoint directory.

    Brax's ``load_config``/``load_policy`` pair cannot read back the config that
    ``ppo.train`` writes: deserialisation maps ``mean_kernel_init_fn: null``
    through ``KERNEL_INITIALIZER`` and raises KeyError, and even past that it
    hands ``make_ppo_networks`` the *string* ``"silu"`` for ``activation``, which
    the network then tries to call. Reading the plain JSON that orbax left on
    disk and converting the serialised names back to callables reproduces the
    network that was actually trained.
    """
    checkpoint = resolve_checkpoint(checkpoint)
    config = json.loads((checkpoint / "ppo_network_config.json").read_text())
    kwargs = {
        key: value
        for key, value in config["network_factory_kwargs"].items()
        if value is not None and key not in INITIALIZER_KEYS
    }
    if isinstance(kwargs.get("activation"), str):
        kwargs["activation"] = ACTIVATIONS[kwargs["activation"]]

    normalize = (
        running_statistics.normalize
        if config["normalize_observations"]
        else training_types.identity_observation_preprocessor
    )
    network = ppo_networks.make_ppo_networks(
        tuple(config["observation_size"]["shape"]),
        config["action_size"],
        preprocess_observations_fn=normalize,
        **kwargs,
    )
    params = ppo_checkpoint.load(checkpoint)
    inference = ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)
    if not ignore_time:
        return inference
    # Adapt Brax's (obs, key) -> (action, extras) into this module's
    # (obs, time, key) -> action, so trained and scripted opponents are
    # interchangeable at every call site.
    return lambda obs, time, key: inference(obs, key)[0]


def load_warp_policy(checkpoint: str | Path, deterministic: bool = True):
    """Rebuild an inference policy from a ``boxer.warp_train`` run.

    The checkpoint is a pickle of (params, value params, optax state,
    normalizer, step) with numpy leaves; the run directory's
    ``warp_config.json`` carries the layer widths and the observation/action
    contract. The forward is written against those arrays directly (no flax
    needed), mirroring ``boxer/warp_train``: running-mean normalisation, silu
    MLP, head split into ``loc ∥ raw scale``, actions through tanh.

    Accepts a run root (``runs/warp1``), its ``checkpoint`` directory, or a
    specific ``step-*.pkl``. One caveat when the policy drives the MJX env:
    its guard-error observation block was built from torso/palm SITE frames
    there while the warp env reads the same quantities from sensors -- close,
    but not bit-identical, so arena numbers on a warp arm are approximate
    until the arms are scored natively in warp.
    """
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pkl":
        step_file = p
        root = p.parent.parent if p.parent.name == "checkpoint" else p.parent
    else:
        root, step_file = p, None
        latest = root / "checkpoint" / "latest.txt"
        if not latest.exists():
            raise SystemExit(f"{root}: not a warp run (no checkpoint/latest.txt)")
        step_file = root / "checkpoint" / f"step-{latest.read_text().strip()}.pkl"
    config = json.loads((root / "warp_config.json").read_text())
    with step_file.open("rb") as handle:
        params_p, _params_v, _opt, norm_state, step = pickle.load(handle)

    mean = jp.asarray(norm_state["mean"])
    var = jp.asarray(norm_state["var"])
    kernels = [(jp.asarray(params_p["params"][f"Dense_{i}"]["kernel"]),
                jp.asarray(params_p["params"][f"Dense_{i}"]["bias"]))
               for i in range(len(config["policy_layers"]) + 1)]
    action_size = config["action_size"]

    def policy(obs, time, key):
        del time
        x = (jp.asarray(obs) - mean) / jp.sqrt(var + 1e-8)      # (batch, D)
        for kernel, bias in kernels[:-1]:
            x = x @ kernel + bias
            x = x * jax.nn.sigmoid(x)                           # silu
        kernel, bias = kernels[-1]
        out = x @ kernel + bias                                 # (batch, 2A)
        loc = out[:, :action_size]
        if deterministic:
            return jp.tanh(loc)                                 # mode of the dist
        scale = jax.nn.softplus(out[:, action_size:]) + 0.001
        return jp.tanh(loc + scale * jax.random.normal(key, loc.shape))

    return policy


def passive(action_size: int):
    """Holds the guard pose and does nothing else.

    The learner's action is relative to that same stance, so a zero action is
    literally "stand still with your hands up" -- the baseline that answers
    whether the policy can land a punch at all.
    """

    def policy(obs, time, key):
        del time, key
        return jp.zeros((obs.shape[0], action_size))

    return policy


def jabber(action_size: int, pitch_adr: int, elbow_adr: int, rate: float = 1.6,
           amplitude: float = 1.0):
    """A metronome lead hand: throws a straight jab every ``1 / rate`` simulated seconds.

    Gives the learner something rhythmic to attack and to learn to slip, which a
    fully passive opponent never asks of it.

    It used to be a sine on shoulder pitch alone, which rotates the whole arm about
    the shoulder and sends the fist up in an arc -- a greeting, not a jab. That is why
    a scripted "punch" in this project kept arriving on the opponent's gloves and
    reading as an upward wave on video, and why it is now an elbow extension with the
    upper arm brought round at the same time: the fist travels along the axis the
    boxer faces, which is the shape the gate requires.
    """

    def policy(obs, time, key):
        del key
        phase = 2 * jp.pi * rate * time
        # Half a cycle out, half back. Clipping the sine keeps the extension fast and
        # the return slow, so there is an unmistakable punch phase for the gate to
        # catch rather than a symmetric oscillation.
        strike = jp.clip(jp.sin(phase), 0.0, 1.0)
        actions = jp.zeros((obs.shape[0], action_size))
        actions = actions.at[:, pitch_adr].set(-0.32 * amplitude * strike)
        actions = actions.at[:, elbow_adr].set(-0.55 * amplitude * strike)
        return actions

    return policy


def make_opponent(spec: str, env, deterministic: bool = True):
    """Builds an opponent from ``passive``, ``jabber``, or a checkpoint path.

    The scripted policies need the action width and the arm's actuator indices,
    which belong to the env rather than to this module -- hard-coding them here
    would silently misfire the moment the robot description changes.

    ``spec`` is coerced because a CLI hands over a ``Path`` and the scripted names
    are compared as text.
    """
    spec = str(spec)
    if spec == "passive":
        return passive(env.action_size)
    if spec == "jabber":
        return jabber(
            env.action_size,
            pitch_adr=env.actuator_index(scene.RED + "left_shoulder_pitch_joint"),
            elbow_adr=env.actuator_index(scene.RED + "left_elbow_joint"),
        )
    path = Path(spec)
    warp_root = (path.parent.parent if path.suffix == ".pkl" else path)
    if (warp_root / "warp_config.json").exists():
        return load_warp_policy(spec, deterministic=deterministic)
    return load_policy(spec, deterministic=deterministic)
