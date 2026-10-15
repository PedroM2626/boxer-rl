"""Does the punch gate still accept a punch that is actually thrown?

Tightening the gate to reject a forward topple is only safe if it still says yes
to a real jab, and "it passes the unit test" is a weak guarantee when the unit
test moves the fist in a straight line at a stationary target. This measures it
the way the arena does: drive one corner with the scripted metronome jab, hold the
other still, keep them in contact range, and report how many of the contacts that
actually happened were accepted as punches.

    python -m tools.gate_calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxer import policy, scene
from boxer.env import NUM_AGENTS, BoxingEnv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spawn-distance", type=float, default=1.15)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)

    env = BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0,
                    spawn_distance=args.spawn_distance)
    jab = policy.jabber(
        env.action_size,
        pitch_adr=env.actuator_index("red/left_shoulder_pitch_joint"),
        elbow_adr=env.actuator_index("red/left_elbow_joint"),
    )
    state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)

    contacts = hits = heads = 0.0
    best_gate = 0.0
    overlap = 0.0
    for _ in range(args.steps):
        action = jp.stack([
            jab(state.obs[0][None], jp.array([float(state.pipeline_state.time)]),
                jax.random.PRNGKey(0))[0],
            jp.zeros(env.action_size),
        ])
        # The gate has to be read before the step: it describes the fist as it
        # travels, and a contact is only a punch if the gate was open on the way
        # into the target.
        me = env._frame(state.pipeline_state, scene.RED)
        foe = env._frame(state.pipeline_state, scene.BLUE)
        gate = float(env._committed(me, foe))
        best_gate = max(best_gate, gate)
        state = step(state, action)
        m = {k: float(np.asarray(v)) for k, v in state.metrics.items()}
        contacts += m["contacts"]
        hits += m["hits_red"]
        heads += m["heads_red"]
        overlap += m["contacts"] * (gate > 0.5)

    print(f"{args.steps} control steps at {args.spawn_distance} m, red jabbing at a still opponent")
    print(f"  contacts              {contacts:8.1f}")
    print(f"  punches (red)         {hits:8.1f}")
    print(f"  head shots            {heads:8.1f}")
    print(f"  accepted              {hits / contacts if contacts else 0.0:8.3f}")
    print(f"  peak gate while open  {best_gate:8.3f}   <- the gate does accept the jab")
    print(f"  contacts thrown into  {overlap:8.1f}   <- gate open on the step before")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
