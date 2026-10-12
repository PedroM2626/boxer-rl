"""Where does a counted punch leak through the gates?

The A/B said the two gates cut 243 counted punches a round to 181, which is still
33 a second, so at least one of them is not binding. This plays one checkpoint
against its own recorded environment and prints, per control step, what each gate
saw: whether the boxer was armed, how far the closest fist actually was from the
closest target, how much approach speed it lost, and whether a punch was counted.

    python -m tools.counter_trace <checkpoint> [--steps 300]
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import jax
import numpy as np

from boxer import arena, policy as policy_module
from boxer.env import resolve_env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--opponent", default="passive")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--worlds", type=int, default=16)
    args = ap.parse_args(argv)

    env = resolve_env((str(args.checkpoint), args.opponent), 5, 0.25, visual=False)
    cfg = env.config()
    print(f"env: rearm_steps={cfg['rearm_steps']} anchor={cfg['anchor'] or '-'} "
          f"guard_pose={cfg['guard_pose']}")

    learner = policy_module.load_policy(policy_module.resolve_checkpoint(args.checkpoint))
    opponent = policy_module.make_opponent(args.opponent, env)
    step = jax.jit(functools.partial(arena._step_pair, jax.vmap(env.step), learner, opponent))
    reset = jax.jit(jax.vmap(env.reset))
    state = reset(jax.random.split(jax.random.PRNGKey(0), args.worlds))

    def body(state, key):
        state, key = step(state, key)
        m, info = state.metrics, state.info
        return state, (m["hits_red"], m["contacts"], info["steps_off"],
                       info["last_contact"])

    _, logged = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), args.steps))
    hits, contacts, steps_off, last_contact = (np.asarray(x) for x in logged)

    scored = hits > 0.0
    print(f"\nsteps where a punch was counted: {scored.mean() * 100:.1f}% of "
          f"{args.steps} x {args.worlds} steps  "
          f"({hits.sum() / (args.steps * env.dt * args.worlds):.1f} punches/s)")
    print(f"steps off the opponent entering a counted step: mean "
          f"{steps_off[scored].mean():.1f}, min {steps_off[scored].min():.0f} "
          f"(the gate asks for {cfg['rearm_steps']})")
    print(f"contact present on the step BEFORE a counted one: "
          f"{(last_contact[scored] > 0).mean() * 100:.1f}%")
    print(f"contact present on a counted step: {(contacts[scored] > 0).mean() * 100:.1f}%")
    print(f"share of all steps spent off the opponent: {(steps_off > 0).mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
