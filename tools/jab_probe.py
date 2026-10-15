"""Why a scripted jab does or does not score, step by step.

`test_a_moving_fist_that_connects_scores` teleports the jabbing corner 0.75 m
closer and asks for at least one counted punch in 40 control steps. Two changes
can each take that to zero on their own, and the assertion cannot tell them
apart:

* the rearm window grew from 8 to 20 steps, so a jab needs 0.4 s off the
  opponent before it can score again;
* arms are worth nothing now, so a jab that lands on a guard is contact without
  a hit -- which is the intended behaviour, but it means "connects" has to say
  what it connected with.

This prints, per step, the touch and hit terms and which contact slots were
inside the margin by geom name, for the teleport and window in play.

    JAX_PLATFORMS=cuda,cpu python tools/jab_probe.py [--rearm 8 20] [--gap 0.75 0.50]
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

from boxer import policy as policy_module, scene
from boxer.env import BoxingEnv


def run(rearm: int, gap: float, steps: int) -> None:
    level = BoxingEnv(rearm_steps=rearm)
    model = level._model

    def name(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<unnamed {gid}>"

    jab = policy_module.jabber(
        level.action_size,
        pitch_adr=level.actuator_index("red/left_shoulder_pitch_joint"),
        elbow_adr=level.actuator_index("red/left_elbow_joint"),
    )
    state = level.reset(jax.random.PRNGKey(6))
    pipeline = state.pipeline_state
    blue_free = level._sides[scene.BLUE]["free_adr"]
    state = state.replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[blue_free].add(-gap))
    )

    def body(state, _):
        clock = jp.atleast_1d(state.pipeline_state.time)[:1]
        action = jp.stack([
            jp.zeros(level.action_size),
            jab(state.obs[1][None], clock, jax.random.PRNGKey(0))[0],
        ])
        state = level.step(state, action)
        inside = (state.pipeline_state.contact.dist
                  < state.pipeline_state.contact.includemargin)
        return state, (state.metrics["contacts"], state.metrics["hits_landed"],
                       state.info["steps_off"], inside,
                       state.pipeline_state.contact.geom)

    _, logged = jax.lax.scan(body, state, None, length=steps)
    contacts, hits, steps_off, inside, geoms = (np.asarray(x) for x in logged)

    print(f"== rearm_steps={rearm}  teleport={gap} m  ({steps} steps)")
    print(f"   contacts>0 on {int((contacts > 0).sum())} steps, "
          f"hits>0 on {int((hits > 0).sum())} steps")
    print(f"   steps_off (jabber is agent 1): "
          f"{' '.join(f'{v:.0f}' for v in steps_off[:, 1][:24])}")

    pairs: dict[tuple[str, str], int] = {}
    for step in range(steps):
        for slot in np.flatnonzero(inside[step]):
            a, b = name(geoms[step, slot, 0]), name(geoms[step, slot, 1])
            if "/" in a and "/" in b and a.split("/")[0] != b.split("/")[0]:
                key = tuple(sorted((a, b)))
                pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        print("   no cross-corner contact inside the margin on any step")
    for pair, count in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"   {count:3d} steps  {pair[0]} <-> {pair[1]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--rearm", type=int, nargs="+", default=[8, 20])
    ap.add_argument("--gap", type=float, nargs="+", default=[0.75, 0.50])
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args(argv)
    for rearm in args.rearm:
        for gap in args.gap:
            run(rearm, gap, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
