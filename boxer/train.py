"""Trains one shared policy to box against its own current copy.

Both corners are driven by the same network and see the same egocentric
observation, so this is symmetric self-play: anything the policy learns to do it
also has to learn to defend against. Brax's PPO is used unmodified -- the duel is
presented to it as an ordinary single-agent env with twice as many learners as
physics worlds (see :mod:`boxer.selfplay`).

Run it on the GPU with ``JAX_PLATFORMS=cuda,cpu`` -- not ``cuda`` alone, which
leaves no CPU device for Brax's ``jax.debug.callback`` metrics logger. On CPU it
works but is roughly forty times slower. ``python -m boxer.bench`` measures both.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from boxer import policy as policy_module
from boxer import selfplay
from boxer.env import NUM_AGENTS, BoxingEnv


def _jax_compat_for_brax() -> None:
    """Restore the one JAX API Brax 0.14.2's PPO loop still calls.

    ``jax.device_put_replicated`` was removed from the JAX that Brax now resolves
    against, and Brax's own deprecation notice says its training stack is not
    actively maintained. The legacy function returned a *stacked* array whose
    leading axis indexed the devices -- Brax's ``_unpmap`` then reads
    ``addressable_shards[0].data`` and squeezes that axis away -- so the
    replacement has to reproduce the leading axis, not just replicate into a
    sharding.
    """
    if hasattr(jax, "device_put_replicated"):
        return

    def device_put_replicated(tree, devices):
        stacked = jax.tree.map(lambda x: jp.stack([x] * len(devices)), tree)
        mesh = jax.sharding.Mesh(tuple(devices), axis_names=("replica",))
        return jax.device_put(
            stacked, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("replica"))
        )

    jax.device_put_replicated = device_put_replicated


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run = parser.add_argument_group("run")
    run.add_argument("--mode", choices=("selfplay", "frozen"), default="selfplay",
                     help="selfplay: one shared policy drives both corners. "
                          "frozen: the learner fights a fixed opponent.")
    run.add_argument("--opponent", default="passive",
                     help="frozen mode only: 'passive', 'jabber', or a checkpoint dir")
    run.add_argument("--learner-side", choices=("red", "blue"), default="red",
                     help="frozen mode only: which corner the learner occupies")
    run.add_argument("--total-steps", type=int, default=200_000_000,
                     help="learner steps; in self-play that is twice the world steps")
    run.add_argument("--worlds", type=int, default=64,
                     help="parallel physics worlds; PPO sees twice as many learners")
    run.add_argument("--episode-length", type=int, default=500,
                     help="control steps per round")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=Path("runs/boxer"))
    run.add_argument("--resume", action="store_true",
                     help="continue from the checkpoint in --output")
    run.add_argument("--init-from", type=Path, default=None,
                     help="checkpoint to take weights from when --output has none, "
                          "e.g. starting a league corner from the frozen arm's net")

    env = parser.add_argument_group("environment")
    env.add_argument("--n-frames", type=int, default=5,
                     help="physics substeps per control step (substep is 4 ms)")
    env.add_argument("--spawn-distance", type=float, default=1.3)
    env.add_argument("--target-range", type=float, default=0.9)
    # Position actuators are commanded at 50 Hz. A scale of 1.0 lets one action
    # target the whole joint range, which topples the boxers within ~15 steps of
    # random exploration and leaves no signal to learn from; 0.25 keeps the
    # control authority in the range humanoid PPO normally operates in.
    env.add_argument("--action-scale", type=float, default=0.25)

    ppo_group = parser.add_argument_group("ppo")
    ppo_group.add_argument("--unroll-length", type=int, default=20)
    ppo_group.add_argument("--batch-size", type=int, default=2048)
    ppo_group.add_argument("--num-minibatches", type=int, default=8)
    ppo_group.add_argument("--num-updates-per-batch", type=int, default=4)
    ppo_group.add_argument("--learning-rate", type=float, default=3e-4)
    ppo_group.add_argument("--entropy-cost", type=float, default=1e-2)
    ppo_group.add_argument("--discounting", type=float, default=0.97)
    ppo_group.add_argument("--gae-lambda", type=float, default=0.95)
    ppo_group.add_argument("--clipping-epsilon", type=float, default=0.2)
    ppo_group.add_argument("--reward-scaling", type=float, default=1.0)
    # Brax's default policy is 32 units wide, which is sized for an ant. Driving
    # 58 position-controlled joints while balancing needs more capacity, and the
    # default visibly caps what the self-play policy can express.
    ppo_group.add_argument("--policy-layers", type=int, nargs="+", default=[128, 128, 128, 128])
    ppo_group.add_argument("--value-layers", type=int, nargs="+", default=[256, 256, 256, 256, 256])
    ppo_group.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction,
                           default=True, help="running-mean input normalisation")
    ppo_group.add_argument("--num-evals", type=int, default=40,
                           help="progress + checkpoint points over the run")
    ppo_group.add_argument("--run-evals", action="store_true",
                           help="also run Brax's in-loop evaluator each point")
    ppo_group.add_argument("--num-eval-envs", type=int, default=64)
    return parser.parse_args(argv)


def _latest_step(root: Path) -> Path | None:
    """The newest orbax step directory under ``root``, or None if there is none."""
    if not root.is_dir():
        return None
    steps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else None


def eval_schedule(args: argparse.Namespace) -> tuple[int, int]:
    """The horizon Brax will actually run for, given the requested eval count.

    Brax splits ``num_timesteps`` across the eval points with ``ceil()``, so a
    request smaller than ``num_evals * block`` does not shorten the run -- it
    silently becomes ``num_evals * block``, where one block is
    ``batch_size * unroll_length * num_minibatches`` learner steps. Asking for
    4.19M with 16 evals on this configuration runs 10.5M. Returns the block
    count and the resulting horizon so the log can state what is really happening
    instead of repeating the number that was typed.
    """
    block = args.batch_size * args.unroll_length * args.num_minibatches
    evals = max(args.num_evals - 1, 1)
    per_epoch = max(-(-args.total_steps // (evals * block)), 1)
    return block, evals * per_epoch * block


def _restore_path(args: argparse.Namespace, checkpoint: Path) -> str | None:
    """Which checkpoint Brax should load weights from, or None to start random.

    Brax reloads the normaliser and both networks here, but reinitialises the
    optimizer and restarts ``env_steps`` at zero, so ``--init-from`` is a weight
    transplant rather than a continuation -- which is what seeding a league corner
    from another corner's net needs. A run that already has its own weights keeps
    them, because resetting a corner to the seed every iteration would make the
    league loop train the same few steps for ever.
    """
    if args.resume:
        own = _latest_step(checkpoint)
        if own is None:
            raise SystemExit(f"--resume: no orbax step directory under {checkpoint}")
        return str(own.resolve())
    if args.init_from is None:
        return None
    own = _latest_step(checkpoint)
    if own:
        print(f"{checkpoint} already has weights; continuing it instead of "
              f"transplanting {args.init_from}", flush=True)
        return str(own.resolve())
    seed = _latest_step(args.init_from) or _latest_step(args.init_from / "checkpoint")
    if seed is None:
        raise SystemExit(f"--init-from {args.init_from} holds no orbax step directory")
    # orbax rejects a relative directory outright -- "Checkpoint path should be
    # absolute" -- and every caller passes a run directory a person typed, so
    # resolve here rather than at each call site.
    return str(seed.resolve())


def main(argv=None) -> int:
    args = _parse_args(argv)
    _jax_compat_for_brax()

    print(f"devices: {jax.devices()}", flush=True)
    env = BoxingEnv(
        n_frames=args.n_frames,
        spawn_distance=args.spawn_distance,
        target_range=args.target_range,
        action_scale=args.action_scale,
    )
    if args.mode == "frozen":
        opponent = policy_module.make_opponent(args.opponent, env)
        wrapped = selfplay.wrap_frozen(
            env, opponent, episode_length=args.episode_length,
            learner_side=0 if args.learner_side == "red" else 1,
        )
        # One learner per physics world: the other corner is not being trained.
        learners = args.worlds
    else:
        wrapped = selfplay.wrap(
            env, episode_length=args.episode_length, observation_dim=env.observation_size
        )
        learners = args.worlds * NUM_AGENTS
    print(f"mode={args.mode} learners={learners} worlds={args.worlds}", flush=True)

    if (args.batch_size * args.num_minibatches) % learners:
        raise SystemExit(
            "--batch-size * --num-minibatches must be a multiple of the learner "
            f"count ({learners})"
        )

    block, horizon = eval_schedule(args)
    print(f"one checkpoint = {block:,} learner steps; {args.num_evals} checkpoints "
          f"will run ~{horizon:,} learner steps (asked for {args.total_steps:,})",
          flush=True)
    if horizon > 1.25 * args.total_steps:
        print(f"WARNING: --total-steps is being ignored. Lower --num-evals "
              f"(or --batch-size) to get near {args.total_steps:,}; the shortest "
              f"run at this batch geometry is {block:,} steps.", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    # orbax rejects relative paths outright, so resolve before handing it over.
    checkpoint = (args.output / "checkpoint").resolve()
    progress_log = args.output / "progress.jsonl"
    clock = {"start": time.time(), "step": 0, "tick": time.time()}

    def progress(step: int, metrics) -> None:
        now = time.time()
        interval = now - clock["tick"]
        delta = step - clock["step"]
        clock["step"], clock["tick"] = step, now
        flat = {
            key: float(np.asarray(value))
            for key, value in metrics.items()
            if np.ndim(value) == 0 and np.isfinite(np.asarray(value, dtype=float)).all()
        }
        record = {"step": step, "walltime": now - clock["start"], **flat}
        if interval > 0 and delta > 0:
            # Frozen mode trains one learner per physics world, self-play two, so
            # dividing by a fixed NUM_AGENTS would understate frozen throughput 2x
            # -- and these are the rates the mode comparison is read from.
            record["learner_steps_per_second"] = delta / interval
            record["world_steps_per_second"] = delta * args.worlds / learners / interval
        with progress_log.open("a") as sink:
            sink.write(json.dumps(record) + "\n")
        headline = "  ".join(
            f"{key.split('/')[-1]}={flat[key]:.3f}"
            for key in sorted(flat)
            if "episode" in key or key.endswith("sps") or "return" in key
        )
        sps = record.get("learner_steps_per_second")
        rate = f"{sps:,.0f} learner-steps/s" if sps else "warming up"
        print(f"[{step:>12,}] {rate:>22}  {headline}", flush=True)

    make_inference_fn, params, metrics = ppo.train(
        environment=wrapped,
        wrap_env=False,
        num_envs=learners,
        episode_length=args.episode_length,
        action_repeat=1,
        num_timesteps=args.total_steps,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        reward_scaling=args.reward_scaling,
        normalize_observations=args.normalize_observations,
        bootstrap_on_timeout=True,
        log_training_metrics=True,
        seed=args.seed,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(args.policy_layers),
            value_hidden_layer_sizes=tuple(args.value_layers),
        ),
        use_pmap_on_reset=False,
        # Brax only calls progress_fn and writes checkpoints when num_evals > 0,
        # so the count is what makes a long run observable; run_evals separately
        # decides whether the in-loop evaluator actually rolls a policy out.
        num_evals=max(args.num_evals, 1),
        num_eval_envs=args.num_eval_envs,
        run_evals=args.run_evals,
        progress_fn=progress,
        save_checkpoint_path=str(checkpoint),
        # Brax hands this straight to orbax's metadata reader, which cannot point
        # at a checkpoint root -- it raises "No structure could be identified for
        # the checkpoint" -- so name the newest step directory instead. Resuming
        # wins over --init-from: a corner that already has weights should continue
        # with its own rather than be reset to the seed.
        restore_checkpoint_path=_restore_path(args, checkpoint),
    )
    del make_inference_fn, params, metrics

    print(f"checkpoint at {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
