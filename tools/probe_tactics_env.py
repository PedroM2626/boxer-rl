"""Smoke-test and time the batched tactics env with scripted tactics on both sides."""
import argparse
import time

import jax
import jax.numpy as jp
import numpy as np

from boxer.curriculum_train import load_policy_params
from boxer.tactics_env import TacticsEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--randomize", type=float, default=0.0,
                        help="domain randomization strength (0 = nominal physics)")
    args = parser.parse_args()
    motor, _ = load_policy_params("checkpoints/spar")
    env = TacticsEnv(motor, worlds_hint=args.worlds)
    if args.randomize:
        from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
        keys = jax.random.split(jax.random.PRNGKey(99), args.worlds)
        batched = BraxDomainRandomizationVmapWrapper(
            env, lambda model: env.randomizer(args.randomize)(model, keys))
        reset, step = jax.jit(batched.reset), jax.jit(batched.step)
    else:
        reset = jax.jit(jax.vmap(env.reset))
        step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def scripted(state, keys):
        actions, _ = jax.vmap(env._scripted)(state.obs, keys, state.info["scripted_wait"])
        return actions

    state = reset(jax.random.split(jax.random.PRNGKey(0), args.worlds))
    alive = np.ones(args.worlds, bool)
    totals = {k: np.zeros(args.worlds) for k in ("hits", "hits_taken", "clinch", "punches")}
    ended_at = np.full(args.worlds, args.steps)
    started = None
    for t in range(args.steps):
        keys = jax.random.split(jax.random.PRNGKey(t + 1), args.worlds)
        state = step(state, scripted(state, keys))
        if t == 0:
            jax.block_until_ready(state.obs)
            started = time.perf_counter()
        done = np.asarray(state.done) > 0
        for k in totals:
            totals[k] += np.asarray(state.metrics[k]) * alive
        fell = (np.asarray(state.metrics["fell"]) + np.asarray(state.metrics["opponent_fell"])) > 0
        ended_at[alive & done & fell] = t
        alive &= ~done
        if not alive.any():
            break
    jax.block_until_ready(state.obs)
    elapsed = time.perf_counter() - started
    print(f"{args.worlds} worlds x {t} steps in {elapsed:.1f}s: "
          f"{args.worlds * t / elapsed:,.0f} world-steps/s")
    print("finite obs:", bool(np.isfinite(np.asarray(state.obs)).all()))
    print("bouts with a fall:", int((ended_at < args.steps).sum()), "/", args.worlds,
          "median fall step:", float(np.median(ended_at[ended_at < args.steps])) if (ended_at < args.steps).any() else None)
    for k, v in totals.items():
        print(f"{k}: mean {v.mean():.2f} per bout")


if __name__ == "__main__":
    main()
