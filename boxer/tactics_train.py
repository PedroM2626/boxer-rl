"""Self-play PPO for boxing tactics over the frozen sparring motor.

Generation 0 trains red against the scripted ring brain. Each later
generation trains against a pool: frozen copies of every earlier generation,
plus the scripted brain in a fraction of bouts. Training only against the
latest generation produced a policy that beat it in batched physics but lost
to it natively (see docs/tactics.md); the pool and native-style starts
target that gap. The learner warm-starts from its own latest weights; Adam
is fresh each generation (Brax does not expose it).
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import jax_compat, load_policy_params, save_policy, utc
from boxer.tactics_env import (ACTION_SIZE, OBS_SIZE, TacticsEnv, policy_sizes,
                               tactics_network_factory, widen_legacy)
from boxer.tactics_gate import gate_generation


def generation_dir(output, generation):
    return output / f"gen-{generation}"


def train_generation(args, generation, motor, learner, pool):
    env = TacticsEnv(motor, pool or None, scripted_fraction=args.scripted_fraction if pool else 1.0,
                     worlds_hint=args.worlds, native_reset_fraction=args.native_reset_fraction,
                     scoring=args.scoring, fall_penalty=args.fall_penalty)
    root = generation_dir(args.output, generation)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = (root / "progress.jsonl").open("a")

    def progress(step, metrics):
        record = dict(generation=generation, step=int(step), elapsed=time.monotonic() - started,
                      updated_utc=utc(), **{k: float(np.asarray(v)) for k, v in metrics.items()
                                            if k.startswith(("episode/", "training/sps", "training/entropy"))})
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    def checkpoint(step, make_policy, params):
        del make_policy
        leaves = jax.tree.leaves(jax.device_get(params))
        if not all(np.isfinite(x).all() for x in leaves):
            raise FloatingPointError("non-finite tactics policy")
        save_policy(root / "checkpoint", params, dict(stage="tactics", generation=generation,
                    stage_steps=int(step), opponents=["scripted"] + [f"pool-{g}" for g in range(len(pool))],
                    native_reset_fraction=args.native_reset_fraction, scoring=args.scoring,
                    randomize=args.randomize, fall_penalty=args.fall_penalty,
                    pool_sources=[str(p) for p in args.pool], initialized_from=str(args.init_from),
                    scripted_fraction=env.scripted_fraction, updated_utc=utc()))

    _, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps_per_generation, num_envs=args.worlds,
        episode_length=env.episode_steps,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        learning_rate=args.learning_rate, entropy_cost=args.entropy_cost, discounting=.97,
        gae_lambda=.95, unroll_length=20, batch_size=args.worlds // 2, num_minibatches=8,
        num_updates_per_batch=4, normalize_observations=True, max_grad_norm=1.0,
        network_factory=tactics_network_factory(), seed=args.seed + generation,
        num_evals=args.checkpoints + 1, num_eval_envs=64, run_evals=False,
        restore_params=learner, policy_params_fn=checkpoint, progress_fn=progress,
        randomization_fn=env.randomizer(args.randomize) if args.randomize else None)
    log.close()
    atomic_json(root / "done.json", dict(generation=generation, steps=args.steps_per_generation,
                                         seconds=time.monotonic() - started, updated_utc=utc()))
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--output", type=Path, default=Path("runs/tactics"))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--steps-per-generation", type=int, default=20_000_000)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--scripted-fraction", type=float, default=.3)
    parser.add_argument("--native-reset-fraction", type=float, default=.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scoring", choices=("punch", "native"), default="punch",
                        help="training hit rule; 'native' mirrors the native ImpactScorecard")
    parser.add_argument("--randomize", type=float, default=0.0,
                        help="domain randomization strength (TacticsEnv.randomizer); 0 = nominal physics")
    parser.add_argument("--fall-penalty", type=float, default=10.0,
                        help="reward lost when the learner falls")
    parser.add_argument("--native-gate", type=int, default=0,
                        help="natively evaluate this many final checkpoints per generation and "
                             "continue from the best (boxer.tactics_gate); 0 trusts the last one")
    parser.add_argument("--champion", type=Path, default=Path("checkpoints/tactics"),
                        help="opponent for the native gate")
    parser.add_argument("--init-from", type=Path, help="tactics checkpoint to warm-start generation 0")
    parser.add_argument("--pool", type=Path, nargs="*", default=[],
                        help="earlier tactics checkpoints that join every generation's opponent pool")
    args = parser.parse_args(argv)
    jax_compat()
    motor, _ = load_policy_params(args.motor)
    learner = load_policy_params(args.init_from)[0] if args.init_from else None
    if learner is not None and policy_sizes(learner) != (OBS_SIZE, ACTION_SIZE):
        learner = widen_legacy(learner)
    pool = [load_policy_params(path)[0] for path in args.pool]
    for generation in range(args.generations):
        root = generation_dir(args.output, generation)
        if (root / "done.json").exists():
            print(f"generation {generation} already complete; resuming after it", flush=True)
        else:
            train_generation(args, generation, motor, learner, pool)
        selected = root / "checkpoint"
        if args.native_gate:
            selected = gate_generation(root, args.champion, args.native_gate)
        learner, _ = load_policy_params(selected)
        pool.append(learner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
