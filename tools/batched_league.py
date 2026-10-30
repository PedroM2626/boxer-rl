"""Head-to-head of two tactics policies in the batched env (training-proxy hits).

Compares with native ``spar_match`` results to tell a real skill difference
from a batched/native mismatch. ``-`` as a policy means the scripted brain.
"""
import argparse
import json

import jax
import numpy as np

from boxer.curriculum_train import load_policy_params
from boxer.tactics_env import TacticsEnv, tactics_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("red")
    parser.add_argument("blue")
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--stochastic", action="store_true", help="sample both policies")
    args = parser.parse_args()
    motor, _ = load_policy_params("checkpoints/spar")
    blue = None if args.blue == "-" else load_policy_params(args.blue)[0]
    env = TacticsEnv(motor, None if blue is None else [blue], scripted_fraction=0.0, worlds_hint=args.worlds,
                     opponent_deterministic=not args.stochastic)
    red = None if args.red == "-" else jax.vmap(
        tactics_policy(load_policy_params(args.red)[0], not args.stochastic))
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

    @jax.jit
    def act(state, keys):
        if red is None:
            return jax.vmap(env._scripted)(state.obs, keys, state.info["scripted_wait"])[0]
        return red(state.obs, keys)[0]

    state = reset(jax.random.split(jax.random.PRNGKey(7), args.worlds))
    alive = np.ones(args.worlds, bool)
    total = {k: np.zeros(args.worlds) for k in ("hits", "hits_taken", "fell", "opponent_fell", "clinch")}
    for t in range(env.episode_steps):
        state = step(state, act(state, jax.random.split(jax.random.PRNGKey(t), args.worlds)))
        for k in total:
            total[k] += np.asarray(state.metrics[k]) * alive
        alive &= np.asarray(state.done) == 0
        if not alive.any():
            break
    red_win = (total["opponent_fell"] > 0) | ((total["fell"] == 0) & (total["hits"] > total["hits_taken"]))
    blue_win = (total["fell"] > 0) | ((total["opponent_fell"] == 0) & (total["hits_taken"] > total["hits"]))
    print(json.dumps(dict(red=args.red, blue=args.blue, stochastic=args.stochastic, bouts=args.worlds,
                          red_wins=int(red_win.sum()), blue_wins=int(blue_win.sum()),
                          red_hits=float(total["hits"].mean()), blue_hits=float(total["hits_taken"].mean()),
                          red_falls=int((total["fell"] > 0).sum()),
                          blue_falls=int((total["opponent_fell"] > 0).sum()),
                          clinch_seconds=float(total["clinch"].mean() * env.dt))))


if __name__ == "__main__":
    main()
