"""Measures how fast the duel actually runs on this machine.

Reports control steps per second -- the unit an RL run spends -- alongside the
raw physics substep rate, since the two differ by the ``--n-frames`` decimation.
Set ``JAX_PLATFORMS=cuda`` to measure the GPU.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import jax
from jax import numpy as jp

from boxer import selfplay
from boxer.env import NUM_AGENTS, BoxingEnv


def measure(env: BoxingEnv, worlds: int, episode_length: int, steps: int) -> Dict[str, float]:
    learners = worlds * NUM_AGENTS
    wrapped = selfplay.wrap(
        env, episode_length=episode_length, observation_dim=env.observation_size
    )
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    action = jp.zeros((learners, env.action_size))

    def rollout(state):
        body = lambda state, _: (step(state, action), None)
        return jax.lax.scan(body, state, None, length=steps)[0]

    rollout = jax.jit(rollout)
    state = reset(jax.random.split(jax.random.PRNGKey(0), learners))

    started = time.time()
    jax.block_until_ready(rollout(state))
    compile_seconds = time.time() - started

    started = time.time()
    jax.block_until_ready(rollout(state))
    elapsed = time.time() - started

    world_steps = worlds * steps
    return {
        "worlds": worlds,
        "learners": learners,
        "compile_s": compile_seconds,
        "control_steps_per_second": world_steps / elapsed,
        "learner_steps_per_second": world_steps * NUM_AGENTS / elapsed,
        "physics_substeps_per_second": world_steps * env._n_frames / elapsed,
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50, help="control steps per measurement")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    print(f"backend devices: {jax.devices()}", flush=True)
    env = BoxingEnv(n_frames=args.n_frames)
    print(
        f"control dt = {float(env.dt) * 1000:.0f} ms "
        f"({args.n_frames} substeps of {float(env.sys.opt.timestep) * 1000:.0f} ms)\n",
        flush=True,
    )
    header = (
        f"{'worlds':>7} {'learners':>9} {'compile s':>10} "
        f"{'control/s':>11} {'learner/s':>11} {'substep/s':>11}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for worlds in args.worlds:
        result = measure(env, worlds, args.episode_length, args.steps)
        print(
            f"{result['worlds']:>7} {result['learners']:>9} {result['compile_s']:>10.1f} "
            f"{result['control_steps_per_second']:>11,.0f} "
            f"{result['learner_steps_per_second']:>11,.0f} "
            f"{result['physics_substeps_per_second']:>11,.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
