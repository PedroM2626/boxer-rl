"""Fit hook and high-guard arm poses for the G1 by forward kinematics.

Poses are (shoulder pitch, roll, yaw, elbow) for the left arm; the right arm
mirrors roll and yaw, like ``curriculum_env.PUNCH_*``. Fist positions are
reported in the torso frame (x forward, y left, z up).
"""
import argparse

import mujoco
import numpy as np
from scipy.optimize import minimize

from boxer.curriculum_env import PUNCH_BODY, PUNCH_HEAD, UnassistedG1

JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")


class Arm:
    def __init__(self):
        env = UnassistedG1("spar")
        self.model = env.mj_model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = np.asarray(env._init_q)
        self.torso = self.model.body("torso_link").id
        self.adr, self.range = {}, {}
        for hand in ("left", "right"):
            ids = [self.model.joint(f"{hand}_{j}_joint").id for j in JOINTS]
            self.adr[hand] = [int(self.model.jnt_qposadr[i]) for i in ids]
            self.range[hand] = self.model.jnt_range[ids]
        self.fist = {h: self.model.geom(f"{h}_hand_collision").id for h in ("left", "right")}
        self.elbow_body = {h: self.model.body(f"{h}_elbow_link").id for h in ("left", "right")}

    def place(self, pose, hand="left"):
        sign = 1 if hand == "left" else -1
        q = np.asarray(pose) * [1, sign, sign, 1]
        self.data.qpos[self.adr[hand]] = q
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.xmat[self.torso].reshape(3, 3)
        origin = self.data.xpos[self.torso]
        fist = rot.T @ (self.data.geom_xpos[self.fist[hand]] - origin)
        elbow = rot.T @ (self.data.xpos[self.elbow_body[hand]] - origin)
        return fist, elbow


def fit(arm, target, start, elbow_target=None, weight=4.0):
    low, high = arm.range["left"][:, 0], arm.range["left"][:, 1]

    def cost(q):
        fist, elbow = arm.place(q)
        c = np.sum((fist - target) ** 2)
        if elbow_target is not None:
            c += weight * np.sum((elbow - elbow_target) ** 2) * .25
        return c
    best = min((minimize(cost, np.clip(s, low, high), bounds=list(zip(low, high)), method="L-BFGS-B")
                for s in start), key=lambda r: r.fun)
    return best.x, arm.place(best.x), best.fun


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    arm = Arm()
    np.set_printoptions(precision=3, suppress=True)
    print("left arm ranges", arm.range["left"])
    guard = np.asarray(arm.data.qpos[arm.adr["left"]])
    for name, pose in (("guard", guard), ("straight body", PUNCH_BODY), ("straight head", PUNCH_HEAD)):
        fist, elbow = arm.place(pose)
        print(f"{name:14s} pose {np.asarray(pose)} fist {fist} elbow {elbow}")
    _, head = arm.place(PUNCH_HEAD)[0], None
    head_fist = arm.place(PUNCH_HEAD)[0]
    body_fist = arm.place(PUNCH_BODY)[0]
    guard_fist = arm.place(guard)[0]
    starts = [guard, PUNCH_HEAD, PUNCH_BODY, np.array([-1.3, .9, .3, .2]), np.array([-1.5, .5, -.5, 0.]),
              np.array([-.8, 1.2, 0., .6])]
    results = {}
    # Hook: the fist arrives across the centre line from the side, elbow held
    # wide and level with the fist, shorter reach than the straight punch.
    for name, fist_ref in (("hook head", head_fist), ("hook body", body_fist)):
        target = np.array([fist_ref[0] - .06, -.02, fist_ref[2]])
        elbow_target = np.array([target[0] - .12, .26, target[2]])
        results[name] = fit(arm, target, starts, elbow_target)
    # Wind-up: fist pulled out to the side at strike height before the swing.
    for name, fist_ref in (("windup head", head_fist), ("windup body", body_fist)):
        target = np.array([guard_fist[0] + .02, .30, fist_ref[2]])
        results[name] = fit(arm, target, starts)
    # High guard: both fists in front of the face, forearms vertical.
    target = np.array([guard_fist[0] + .02, .07, head_fist[2] - .02])
    results["high guard"] = fit(arm, target, starts, np.array([target[0] - .05, .12, target[2] - .2]))
    for name, (q, (fist, elbow), err) in results.items():
        print(f"{name:12s} pose {q} fist {fist} elbow {elbow} residual {err:.4f}")

    # Long hook (v2). The short hook above lands 0-6 % natively: it ends 0.32 m
    # out, at the centre of the face where a high guard sits. This one keeps the
    # straight's reach and sweeps in horizontally from wide and far forward,
    # past the opponent's guard plane (their fists ~0.27 m before their torso).
    long_hook = {}
    for name, fist_ref in (("long hook head", head_fist), ("long hook body", body_fist)):
        target = np.array([fist_ref[0] - .02, .03, fist_ref[2]])
        long_hook[name] = fit(arm, target, starts, np.array([target[0] - .2, .22, target[2]]), weight=1.)
    for name, fist_ref in (("long windup head", head_fist), ("long windup body", body_fist)):
        target = np.array([fist_ref[0] - .06, .34, fist_ref[2]])
        long_hook[name] = fit(arm, target, starts)
    for name, (q, (fist, elbow), err) in long_hook.items():
        print(f"{name:16s} pose {q} fist {fist} elbow {elbow} residual {err:.4f}")

    # Level arc. Interpolating joints from wind-up to strike lifts the head hook
    # to z 0.535 m. Instead, solve IK at waypoints on a horizontal arc around
    # the shoulder, each warm-started from (and kept near) the previous one.
    for aim in ("head", "body"):
        path = arc_path(arm, long_hook[f"long windup {aim}"][0], long_hook[f"long hook {aim}"][0])
        print(f"LONG_HOOK_PATH_{aim.upper()} = np.array([")
        for q in path:
            print("    [" + ", ".join(f"{v:.4f}" for v in q) + "],")
        print("])")
        zs = [arm.place(q)[0][2] for q in path]
        print(f"# fist z along the arc: {np.round(zs, 3).tolist()}")


def arc_path(arm, start, end, waypoints=6, smooth=.02):
    """Joint poses whose fists lie on a level arc around the shoulder."""
    rot = arm.data.xmat[arm.torso].reshape(3, 3)
    shoulder = rot.T @ (arm.data.xpos[arm.model.body("left_shoulder_pitch_link").id]
                        - arm.data.xpos[arm.torso])
    a, b = arm.place(start)[0], arm.place(end)[0]
    angle = lambda p: np.arctan2(p[1] - shoulder[1], p[0] - shoulder[0])
    radius = lambda p: np.hypot(p[1] - shoulder[1], p[0] - shoulder[0])
    low, high = arm.range["left"][:, 0], arm.range["left"][:, 1]
    path, previous = [np.asarray(start)], np.asarray(start)
    for t in np.linspace(0, 1, waypoints)[1:]:
        theta = angle(a) + t * (angle(b) - angle(a))
        r = radius(a) + t * (radius(b) - radius(a))
        target = np.array([shoulder[0] + r * np.cos(theta), shoulder[1] + r * np.sin(theta),
                           a[2] + t * (b[2] - a[2])])
        cost = lambda q: np.sum((arm.place(q)[0] - target) ** 2) + smooth * np.sum((q - previous) ** 2)
        previous = minimize(cost, previous, bounds=list(zip(low, high)), method="L-BFGS-B").x
        path.append(previous)
    return np.array(path)


if __name__ == "__main__":
    main()
