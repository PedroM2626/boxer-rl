"""Which of the four gate factors is zero when a trained arm scores nothing?

The fifth arm's falsifier ended 0.000 hits against the anchored statue with 379
contacts a round, and "why zero" has two candidate answers that call for opposite
next moves: the fists never reach a scoring surface (a morphology/targeting
question -- the head premium is measured to be out of envelope, and the torso may
be too), or they do reach one and the counter refuses them (a gate question, and
the gate is ours to fix). Guessing between them is how a sixth wrong arm gets
trained.

This replays the arena matchup exactly -- learner on red, passive zeros on blue,
the env rebuilt from the run's own recorded config -- and reads one factor per
control step, each from the state the env just returned rather than by re-running
physics:

* ``raw``    red's strike geoms on a *scoring* target in any substep. Read off
             ``info["steps_off"]`` resetting to 0: the env already computes
             ``scored`` (the pre-gate hit sum over the five substeps) and resets
             the rearm counter on it, so post-step ``steps_off == 0`` is exactly
             "scored > 0 this step". No duplicate physics pass, no last-substep
             sampling -- a 4 ms contact in one substep out of five still shows.
* ``commit`` ``env._committed`` recomputed on the state ``env.step`` returned,
             which is the same pipeline its own gate read (the anchor replace
             touches qpos/qvel, not the sensordata the frame reads, and red's
             corner is untouched by it either way).
* ``edge``   presence rose this step: post ``last_contact > 0`` and pre ``<= 0``.
* ``armed``  pre ``steps_off >= rearm_steps``.

Plus naming, not counting: which contact slots sit inside the margin on the
returned state (the last substep's buffer), resolved through the env's own model
so the ids mean what the gate means, and how close each fist actually got to the
head and the torso.

    JAX_PLATFORMS=cuda,cpu python tools/hit_trace.py \
        [--run runs/frozen_paid_200m] [--rounds 16] [--length 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import policy as policy_module
from boxer import scene
from boxer.env import build_from_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="runs/frozen_paid_200m",
                        help="run directory whose config and checkpoint are traced")
    parser.add_argument("--rounds", type=int, default=16,
                        help="parallel rounds (worlds) to trace")
    parser.add_argument("--length", type=int, default=500,
                        help="max control steps per round, as in the arena")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = build_from_run(args.run)
    config = env.config()
    learner = policy_module.make_opponent(
        str(Path(args.run) / "checkpoint"), env, deterministic=True
    )
    statue = policy_module.passive(env.action_size)

    print(f"devices: {jax.devices()}", flush=True)
    print(f"env: action_scale={config['action_scale']} guard_pose={config['guard_pose']} "
          f"anchor={config['anchor'] or '-'} rearm_steps={config['rearm_steps']} "
          f"target_range={config['target_range']} spawn={config['spawn_distance']}",
          flush=True)
    print(f"tracing {Path(args.run).name} on red vs passive on blue: "
          f"{args.rounds} worlds x {args.length} steps\n", flush=True)

    step_env = jax.jit(jax.vmap(env.step))
    reset = jax.jit(jax.vmap(env.reset))
    frame_red = jax.jit(jax.vmap(lambda ps: env._frame(ps, scene.RED)))
    frame_blue = jax.jit(jax.vmap(lambda ps: env._frame(ps, scene.BLUE)))
    commit_of = jax.jit(jax.vmap(env._committed))

    def body(state, key):
        key_a, key_b, step_key = jax.random.split(key, 3)
        clock = state.pipeline_state.time
        action_a = learner(state.obs[:, 0], clock, key_a)
        action_b = statue(state.obs[:, 1], clock, key_b)
        action = jp.stack([action_a, action_b], axis=1)

        pre_presence = state.info["last_contact"][:, 0]
        armed = (state.info["steps_off"][:, 0] >= env._rearm_steps).astype(jp.float32)

        state = step_env(state, action)

        presence = state.info["last_contact"][:, 0]
        edge = ((presence > 0.0) & (pre_presence <= 0.0)).astype(jp.float32)
        raw = (state.info["steps_off"][:, 0] == 0.0).astype(jp.float32)

        red, blue = frame_red(state.pipeline_state), frame_blue(state.pipeline_state)
        commit = commit_of(red, blue)
        fists = jp.stack([red["left_palm"], red["right_palm"]])
        targets = jp.stack([blue["head"], blue["torso"]])
        gap = jp.sqrt(jp.sum((fists[:, None] - targets[None]) ** 2, axis=-1))

        active = state.pipeline_state.contact.dist < state.pipeline_state.contact.includemargin

        return state, (
            raw, commit, edge, armed, state.metrics["hits_red"], presence,
            state.info["steps_off"][:, 0], state.done, state.metrics["distance"],
            gap, active,
        )

    states = reset(jax.random.split(jax.random.PRNGKey(args.seed), args.rounds))
    _, logged = jax.lax.scan(
        jax.jit(body), states, jax.random.split(jax.random.PRNGKey(args.seed + 1), args.length)
    )
    (raw, commit, edge, armed, hit, presence, steps_off, done, distance, gap,
     active) = (np.asarray(x) for x in logged)

    # A step is counted while its round has not ended before it (the ending step
    # itself still happened), and only while every factor stayed finite: one
    # diverged world used to turn an arena mean into nan and hide every other
    # number behind it.
    ended = done > 0.5
    before_end = np.cumsum(ended, axis=0) - ended
    mask = before_end == 0
    finite = np.isfinite(np.stack([raw, commit, edge, armed, hit])).all(axis=0)
    mask &= finite
    total = int(mask.sum())
    dropped = int((~finite).sum())

    print(f"=== the four factors, pooled over {total} counted steps "
          f"({dropped} non-finite dropped) ===")
    print(f"  presence (any touch):        {100.0 * (presence > 0)[mask].mean():6.2f}% of steps")
    print(f"  raw (scoring contact):       {100.0 * (raw > 0.5)[mask].mean():6.2f}% of steps")
    print(f"  commit > 0.5:                {100.0 * (commit > 0.5)[mask].mean():6.2f}% of steps")
    print(f"  rising edge:                 {100.0 * (edge > 0.5)[mask].mean():6.2f}% of steps")
    print(f"  armed:                       {100.0 * (armed > 0.5)[mask].mean():6.2f}% of steps")
    print(f"  hits_red (the product):      {100.0 * (hit > 0.5)[mask].mean():6.2f}% of steps\n")

    touch = mask & (presence > 0.5)
    raw_steps = mask & (raw > 0.5)
    print(f"=== among the {int(touch.sum())} steps touching the opponent ===")
    print(f"  on a scoring surface (raw):  {100.0 * (raw > 0.5)[touch].mean() if touch.any() else 0:6.2f}%")
    if raw_steps.any():
        print(f"  commit on raw steps:         mean {commit[raw_steps].mean():.3f} "
              f"p90 {np.percentile(commit[raw_steps], 90):.3f} max {commit[raw_steps].max():.3f}")
        print(f"  edge on raw steps:           {100.0 * (edge > 0.5)[raw_steps].mean():6.2f}%")
        print(f"  armed on raw steps:          {100.0 * (armed > 0.5)[raw_steps].mean():6.2f}%")
    print()

    # First factor that is zero, in the order the gate multiplies them. On a step
    # with no touch at all, raw is zero by absence and that is the honest reading.
    chain = np.where(raw <= 0.5, 0, np.where(commit <= 0.5, 1,
                      np.where(edge <= 0.5, 2, np.where(armed <= 0.5, 3, 4))))
    names = ("raw==0 (never on a scoring surface)", "commit==0 (gate rejects the motion)",
             "edge==0 (already touching)", "armed==0 (rearm window shut)", "all four passed")
    print("=== binding zero, first in gate order ===")
    for code, name in enumerate(names):
        count = int((chain[mask] == code).sum())
        print(f"  {name}: {count}")
    print()

    hits_per_world = hit.sum(axis=0)
    print(f"=== per-world sanity (rearm_steps={env._rearm_steps}) ===")
    print(f"  hits per world: mean {hits_per_world.mean():.2f} max {hits_per_world.max():.0f}")
    print(f"  steps_off max:   {steps_off.max():.0f} -- it grows by 1 per step without a "
          f"scored contact, so it never resetting reads exactly {args.length} + "
          f"{env._rearm_steps} here, and anything below that means it did reset")
    print(f"  rounds ended early: {int(ended.any(axis=0).sum())} of {args.rounds}")
    print(f"  distance mean:     {distance[mask].mean():.3f} m\n")

    hands = ("left fist", "right fist")
    parts = ("head", "torso")
    print("=== how close each fist came (m, over counted steps) ===")
    for h in range(2):
        for t in range(2):
            values = gap[:, h, t, :]
            print(f"  {hands[h]} -> {parts[t]}: min {values[mask].min():.3f} "
                  f"mean {values[mask].mean():.3f}")
    print()

    # The slot table is a naming aid, not a counter: it reads the buffer of the
    # returned state, which is the last of the five substeps. Prolonged contact
    # -- 379 contacts a round says most of it is -- is named correctly; a 4 ms
    # graze can be missed by the sampling, which is what `raw` above is for.
    counts = (active & mask[..., None]).sum(axis=(0, 1))
    order = np.argsort(-counts)
    model = env._model

    def name(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<{gid}>"

    slot_pairs = np.asarray(env.pipeline_init(
        env._stance_qpos, jp.zeros(env.sys.nv), None, env._stance_ctrl
    ).contact.geom)
    weight = np.asarray(env._slot["red"]["weight"])
    scoring = np.asarray(env._slot["red"]["scoring"])
    head = np.asarray(env._slot["red"]["head"])

    print("=== geoms inside the contact margin (last substep, by slot) ===")
    shown = 0
    for slot in order:
        if counts[slot] <= 0 or shown >= 14:
            break
        a, b = name(slot_pairs[slot, 0]), name(slot_pairs[slot, 1])
        print(f"  {counts[slot]:7d} steps  {a} <-> {b}"
              f"  [red weight {weight[slot]:.2f} scoring {scoring[slot]:.0f} head {head[slot]:.0f}]")
        shown += 1
    if shown == 0:
        print("  (no slot was ever inside the margin on a counted step)")

    verdict = "reach" if not raw_steps.any() else "counter"
    print(f"\nverdict: the binding zero is {verdict} -- "
          + ("the fists never put red's strike geoms on a scoring surface, so no gate "
             "change can pay this policy for a punch it is not throwing."
             if verdict == "reach" else
             "scoring contact happens, so the gate (commit/edge/armed) is what zeroes it; "
             "decompose the named factor before touching anything else."))


if __name__ == "__main__":
    main()
