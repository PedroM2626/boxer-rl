"""Runs fixed matchups and reports who wins.

Self-play reward is not a monotonic progress signal: a policy that gets better
at boxing can score *less* because its opponent is also better. Win rate against
a frozen opponent is the measurement that does not move under you, so this module
exists to make "is it learning to box?" answerable.

Each round is one physics world containing both boxers. Sides are optionally
exchanged and the two halves pooled, because a shared policy can favour one
corner for reasons that have nothing to do with skill.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer import policy as policy_module
from boxer.env import BoxingEnv


def _step_pair(env_step, policy_a, policy_b, state, key):
    """One control step with both corners driven by fixed policies.

    ``env_step`` is the vmapped env step: the policies get a real batch of
    observations, while each physics world stays independent.
    """
    obs = state.obs                                   # (worlds, 2, D)
    clock = state.pipeline_state.time                 # (worlds,)
    key_a, key_b, step_key = jax.random.split(key, 3)
    action_a = policy_a(obs[:, 0], clock, key_a)      # (worlds, A)
    action_b = policy_b(obs[:, 1], clock, key_b)
    action = jp.stack([action_a, action_b], axis=1)   # (worlds, 2, A)
    return env_step(state, action), step_key


def outcomes(
    done: np.ndarray,
    down_red: np.ndarray,
    down_blue: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduces a logged round to per-world results.

    ``done`` and the two ``down_*`` arrays are ``(steps, worlds)``. A round's
    result is fixed by the step it ends on: the ``down_*`` metrics are per-step
    indicators, not accumulators, so reading them anywhere else is meaningless.

    Returns ``(a_win, b_win, draw, finished, last_step)`` as vectors over worlds.
    """
    ended = done > 0.5
    finished = ended.any(axis=0)
    first = np.where(ended, np.arange(length)[:, None], length).min(axis=0)
    last_step = np.where(finished, first, length - 1)
    rows = np.arange(done.shape[1])

    red_down = down_red[last_step, rows] > 0.5
    blue_down = down_blue[last_step, rows] > 0.5
    return (
        finished & blue_down & ~red_down,
        finished & red_down & ~blue_down,
        finished & red_down & blue_down,
        finished,
        last_step,
    )


def play(
    env: BoxingEnv,
    policy_a,
    policy_b,
    rounds: int = 64,
    length: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Plays ``rounds`` parallel rounds, A as red and B as blue.

    Returns per-round means. ``a_win + b_win + draw + undecided`` is 1.
    """
    step = jax.jit(functools.partial(
        _step_pair, jax.vmap(env.step), policy_a, policy_b
    ))
    reset = jax.jit(jax.vmap(env.reset))
    states = reset(jax.random.split(jax.random.PRNGKey(seed), rounds))

    def body(state, key):
        state, key = step(state, key)
        m = state.metrics
        return state, (state.done, m["down_red"], m["down_blue"],
                       m["hits_landed"], m["head_shots"], m["damage"],
                       m["hits_red"], m["hits_blue"], m["heads_red"], m["heads_blue"])

    _, logged = jax.lax.scan(body, states, jax.random.split(jax.random.PRNGKey(seed + 1), length))
    (done, down_red, down_blue, hits, heads, damage, hits_red, hits_blue,
     heads_red, heads_blue) = (np.asarray(x) for x in logged)

    a_win, b_win, draw, finished, last_step = outcomes(done, down_red, down_blue, length)
    survived = np.where(finished, last_step + 1, length)

    # A diverged physics state turns one step's contact count into NaN, and the
    # sum over the round then poisons the mean for every world -- the arena used
    # to print "a_hits_per_round: nan" and nothing else. Keep the rounds that
    # stayed finite and report how many were thrown away, so a missing number is
    # visible instead of looking like a zero.
    totals = {
        "hits": hits.sum(axis=0), "heads": heads.sum(axis=0), "damage": damage.sum(axis=0),
        "hits_red": hits_red.sum(axis=0), "hits_blue": hits_blue.sum(axis=0),
        "heads_red": heads_red.sum(axis=0), "heads_blue": heads_blue.sum(axis=0),
    }
    finite = np.isfinite(np.stack(list(totals.values()))).all(axis=0)
    finite &= np.isfinite(done.sum(axis=0)) & np.isfinite(down_red.sum(axis=0))
    finite &= np.isfinite(down_blue.sum(axis=0))

    a_win, b_win, draw = a_win[finite], b_win[finite], draw[finite]
    survived = survived[finite]
    totals = {key: values[finite] for key, values in totals.items()}

    def mean(values) -> float:
        kept = np.asarray(values)
        return float(kept.mean()) if kept.size else float("nan")

    return {
        "a_win": mean(a_win),
        "b_win": mean(b_win),
        "draw": mean(draw),
        "undecided": float(1.0 - mean(a_win) - mean(b_win) - mean(draw)),
        "round_seconds": mean(survived) * float(env.dt),
        "hits_per_round": mean(totals["hits"]),
        "heads_per_round": mean(totals["heads"]),
        "damage_per_round": mean(totals["damage"]),
        # A is always red inside a single roll, so the pooled means above say how
        # much boxing happened but not by whom -- and "won because he punched"
        # versus "won because he fell on top of the other one" is the whole
        # question this arena exists to answer.
        "a_hits_per_round": mean(totals["hits_red"]),
        "b_hits_per_round": mean(totals["hits_blue"]),
        "a_heads_per_round": mean(totals["heads_red"]),
        "b_heads_per_round": mean(totals["heads_blue"]),
        "dropped_rounds": float(1.0 - finite.mean()),
    }


def matchup(
    env: BoxingEnv,
    spec_a: str,
    spec_b: str,
    rounds: int = 64,
    length: int = 500,
    seed: int = 0,
    swap_sides: bool = True,
) -> dict[str, float]:
    """Plays two opponents, optionally with the corners exchanged.

    ``swap_sides`` matters more than it looks: with a shared or near-shared policy
    a systematic corner advantage would otherwise be read as skill.
    """
    a = policy_module.make_opponent(spec_a, env)
    b = policy_module.make_opponent(spec_b, env)
    forward = play(env, a, b, rounds, length, seed)
    if not swap_sides:
        return forward

    # Roles exchanged: now A is blue, so every per-corner count flips corners
    # before the halves are pooled.
    reverse = play(env, b, a, rounds, length, seed + 10_000)
    pooled = {}
    for key, value in forward.items():
        if key.startswith("a_"):
            other = "b_" + key[2:]
        elif key.startswith("b_"):
            other = "a_" + key[2:]
        else:
            other = key
        pooled[key] = 0.5 * (value + reverse[other])
    return pooled


def _describe(spec: str) -> str:
    path = Path(spec)
    return spec if path.name in ("passive", "jabber") else str(path)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("policy_a", help="checkpoint dir, 'passive' or 'jabber'")
    parser.add_argument("policy_b", nargs="?", default="passive",
                        help="checkpoint dir, 'passive' or 'jabber'")
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--length", type=int, default=500, help="max control steps per round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--no-swap", action="store_true", help="do not exchange corners")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    env = BoxingEnv(n_frames=args.n_frames)
    print(f"devices: {jax.devices()}", flush=True)
    print(f"{_describe(args.policy_a)}  vs  {_describe(args.policy_b)}  "
          f"({args.rounds} rounds, corners {'swapped' if not args.no_swap else 'fixed'})\n",
          flush=True)

    result = matchup(
        env, args.policy_a, args.policy_b,
        rounds=args.rounds, length=args.length, seed=args.seed, swap_sides=not args.no_swap,
    )
    for key, value in result.items():
        print(f"  {key:>20}: {value:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
