"""Starting states taken from the end of a referee's count, for training.

A boxer that got up after a knockdown resumes from a stance its boxing policy
never started from: after the get-up policy has held it up through the
mandatory count, joints and arms where the get-up left them. ``collect``
builds a bank of such states in native MuJoCo. Each starts one robot lying
(as ``runs/getup_native_probe.py``) next to a standing opponent in guard on
its motor, runs the get-up policy through the count, and keeps the pair's
state once the referee has set the boxers apart (``PairedFoundation.separate``),
which is how ``spar_match --getup --count-separate`` resumes a bout.
``native_finetune --start-bank`` starts a share of its training bouts from it.

    python -m boxer.getup_starts --output runs/getup_starts.npz --states 400
"""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ("qpos", "qvel", "ctrl", "last_action", "phase", "frequency")


def lay_down(pair, corner, rng):
    """Put ``corner`` on the floor in a random lying pose (the get-up stage's starts)."""
    import mujoco
    m = pair.maps[corner]
    q = m['root']
    tilt, heading, yaw = rng.uniform(1.3, 1.85), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
    axis = np.array([np.cos(heading), np.sin(heading), 0.])
    a = np.r_[np.cos(tilt / 2), np.sin(tilt / 2) * axis]
    b = np.r_[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, a, b)
    pair.data.qpos[q + 2] = max(.8 * np.cos(tilt) + .05, .3)
    pair.data.qpos[q + 3:q + 7] = quat
    pair.data.qpos[m['q']] = pair.pose + rng.uniform(-.25, .25, 29)
    pair.data.qvel[m['rootv']:m['rootv'] + 6] = 0
    pair.data.qvel[m['v']] = 0
    mujoco.mj_forward(pair.model, pair.data)


def snapshot(pair):
    return dict(qpos=pair.data.qpos.copy(), qvel=pair.data.qvel.copy(), ctrl=pair.data.ctrl.copy(),
                last_action=pair.last_action.copy(), phase=pair.phase.copy(),
                frequency=pair.frequency.copy())


def restore(pair, state):
    """Load a banked state into a reset pair (same model layout, floor colliders or not)."""
    import mujoco
    pair.data.qpos[:] = state["qpos"]
    pair.data.qvel[:] = state["qvel"]
    pair.data.ctrl[:] = state["ctrl"]
    pair.last_action = np.array(state["last_action"])
    pair.phase = np.array(state["phase"])
    pair.frequency = np.array(state["frequency"])
    pair.done = False
    mujoco.mj_forward(pair.model, pair.data)


def collect_one(pair, motor, getup, rng, count_seconds=8., separation=2.):
    """One count from a lying start; the pair's state as boxing resumes, or None
    if the fallen boxer was not up and steady at the end of the count or the
    waiting one fell."""
    from boxer.curriculum_env import LEFT_ARM
    from boxer.spar_match import COUNT_STEADY, PunchArm, motor_actions
    pair.reset(seed=int(rng.integers(1 << 30)))
    pair.stop_on_fall = False
    corner = int(rng.integers(2))
    lay_down(pair, corner, rng)
    guard = PunchArm(pair.pose, pair.action_scale, pair.dt)
    steady = 0
    for _ in range(int(round(count_seconds / pair.dt))):
        obs = pair.observation(np.zeros((2, 3)))
        action = motor_actions(motor, obs)
        action[corner] = motor_actions(getup, obs)[corner]
        action[1 - corner, LEFT_ARM:] = guard.actions(False)
        pair.step(action)
        steady = steady + 1 if pair.standing()[corner] else 0
        if pair.fallen()[1 - corner] and not pair.standing()[1 - corner]:
            return None
    pair.stop_on_fall = True
    if steady * pair.dt < COUNT_STEADY:
        return None
    pair.separate(separation)
    state = snapshot(pair)
    state["risen"] = corner
    return state


def load_bank(path):
    data = np.load(path)
    return [{k: data[k][i] for k in (*FIELDS, "risen")} for i in range(len(data["risen"]))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--getup", type=Path, default=Path("checkpoints/getup"))
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar-native"))
    parser.add_argument("--output", type=Path, default=Path("runs/getup_starts.npz"))
    parser.add_argument("--states", type=int, default=400)
    parser.add_argument("--count-seconds", type=float, default=8.)
    parser.add_argument("--separation", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    import jax
    from boxer.curriculum_env import UnassistedG1
    from boxer.curriculum_train import inference, load_policy_params
    from boxer.paired_foundation import PairedFoundation
    foundation = UnassistedG1('spar')
    getup = jax.jit(inference(foundation, load_policy_params(args.getup)[0]))
    motor = jax.jit(inference(foundation, load_policy_params(args.motor)[0]))
    pair = PairedFoundation(foundation, ring=True, floor_body=True)
    rng = np.random.default_rng(args.seed)
    states, tries = [], 0
    while len(states) < args.states:
        tries += 1
        state = collect_one(pair, motor, getup, rng, args.count_seconds, args.separation)
        if state is not None:
            states.append(state)
        if tries % 20 == 0:
            print(json.dumps(dict(tries=tries, kept=len(states))), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k: np.stack([s[k] for s in states]) for k in (*FIELDS, "risen")})
    print(json.dumps(dict(output=str(args.output), states=len(states), tries=tries)), flush=True)


if __name__ == "__main__":
    main()
