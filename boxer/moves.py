"""Arm moves shared by the spar generator, batched tactics and native play.

Poses are left-arm (shoulder pitch, roll, yaw, elbow); the right arm mirrors
roll and yaw. Hook and high-guard poses were fitted by forward kinematics
(``tools/fit_moves.py``); fist positions below are in the torso frame.

* straight: guard -> strike in 0.18 s, back in 0.26 s (fist reaches 0.41-0.44 m).
* hook: guard -> wind-up (fist 0.34 m out to the side, far forward) -> a level
  arc through IK waypoints to just past the centre line (fist 0.37-0.41 m),
  0.26 s out and 0.28 s back.
* high guard: both fists in front of the face (0.43 m up against the normal
  guard's 0.26 m), held while not punching. Fists and forearms collide
  with incoming punches; only head and torso contacts score.

Every function takes ``xp`` (``numpy`` or ``jax.numpy``) so all callers share
one definition.
"""
from __future__ import annotations

import numpy as np

STRAIGHT, HOOK = 0, 1
STRAIGHT_BODY = np.array([-1.18, .04, .10, 1.20])
STRAIGHT_HEAD = np.array([-1.94, .065, .20, 1.51])
# Hook (v2, active): straight-like reach (fist 0.37-0.41 m), swung on a level
# arc around the shoulder from a wide, far-forward wind-up (fist 0.35-0.38 m
# out, 0.34 m to the side) to just past the centre line. Waypoints come from
# IK along the arc (tools/fit_moves.py): fist height stays within 5 mm.
HOOK_PATH_HEAD = np.array([
    [-1.4984, 0.5590, 0.8216, 0.8652],
    [-1.5297, 0.4382, 0.7645, 0.8802],
    [-1.5615, 0.3039, 0.7028, 0.8862],
    [-1.5897, 0.1684, 0.6416, 0.8834],
    [-1.6146, 0.0338, 0.5806, 0.8730],
    [-1.6372, -0.0982, 0.5189, 0.8561],
])
HOOK_PATH_BODY = np.array([
    [-0.8071, 0.6152, 0.2590, 0.9402],
    [-0.8563, 0.5005, 0.2002, 0.9131],
    [-0.8995, 0.3735, 0.1354, 0.8868],
    [-0.9319, 0.2462, 0.0698, 0.8638],
    [-0.9543, 0.1199, 0.0038, 0.8439],
    [-0.9674, -0.0049, -0.0630, 0.8272],
])
WINDUP_HEAD, HOOK_HEAD = HOOK_PATH_HEAD[0], HOOK_PATH_HEAD[-1]
WINDUP_BODY, HOOK_BODY = HOOK_PATH_BODY[0], HOOK_PATH_BODY[-1]
# Short hook (v1): fist 0.32-0.36 m out at the face centre, joint-space
# swing. It landed 0-6 % natively (tools/move_effectiveness.py) and the
# policy never used it. Motors before 2026-10-31 evening were trained on it.
SHORT_HOOK_BODY = np.array([-1.191, .565, -1.17, .388])
SHORT_HOOK_HEAD = np.array([-2.09, .817, -1.541, .357])
SHORT_WINDUP_BODY = np.array([.026, .702, .296, -.12])
SHORT_WINDUP_HEAD = np.array([-1.19, .354, .738, .077])
HIGH_GUARD = np.array([-1.162, .087, .008, -.074])
TIMING = {STRAIGHT: (.18, .26), HOOK: (.26, .28)}      # extension, retraction seconds
WINDUP_FRACTION = .4                                   # share of a hook's extension


def mirror(xp, pose, hand):
    """Left-arm pose -> pose for ``hand`` (0 left, 1 right)."""
    sign = xp.where(hand == 0, 1.0, -1.0)
    return pose * xp.stack([xp.ones_like(sign), sign, sign, xp.ones_like(sign)])


def durations(xp, move, dt):
    """(extension, retraction) in control steps for ``move``."""
    extend = xp.where(move == HOOK, TIMING[HOOK][0], TIMING[STRAIGHT][0]) / dt
    retract = xp.where(move == HOOK, TIMING[HOOK][1], TIMING[STRAIGHT][1]) / dt
    return extend, retract


def strike_pose(xp, move, aim, jitter=0.):
    """Left-arm strike endpoint for ``aim`` in [0 body, 1 head]."""
    straight = STRAIGHT_BODY + aim * (STRAIGHT_HEAD - STRAIGHT_BODY)
    hook = HOOK_BODY + aim * (HOOK_HEAD - HOOK_BODY)
    return xp.where(move == HOOK, hook, straight) + jitter


def punch_pose(xp, move, aim, target, age, rest, dt):
    """Left-arm pose ``age`` steps into a punch toward ``target`` (strike pose).

    ``rest`` is the pose the arm leaves from and returns to (normal or high
    guard). Straights keep the original timing: fraction (age + 1) / extension
    out, then 1 - (age - extension) / retraction back.
    """
    extend, retract = durations(xp, move, dt)
    u = xp.clip((age + 1) / extend, 0., 1.)
    path = HOOK_PATH_BODY + aim * (HOOK_PATH_HEAD - HOOK_PATH_BODY)
    # The strike endpoint is ``target`` (it carries the launch jitter).
    path = xp.concatenate([path[:-1], target[None]], axis=0)
    first = xp.clip(u / WINDUP_FRACTION, 0., 1.)
    second = xp.clip((u - WINDUP_FRACTION) / (1 - WINDUP_FRACTION), 0., 1.)
    position = second * (len(HOOK_PATH_HEAD) - 1)
    index = xp.clip(xp.floor(position), 0, len(HOOK_PATH_HEAD) - 2).astype(int)
    swing = path[index] + (position - index) * (path[index + 1] - path[index])
    hook_out = xp.where(u < WINDUP_FRACTION, rest + first * (path[0] - rest), swing)
    straight_out = rest + u * (target - rest)
    out = xp.where(move == HOOK, hook_out, straight_out)
    back = xp.clip(1 - (age - extend) / retract, 0., 1.)
    home = rest + back * (target - rest)
    return xp.where(age < extend, out, home)


def finished(xp, move, age, dt):
    extend, retract = durations(xp, move, dt)
    return age >= extend + retract


def rest_pose(xp, guard_pose, high):
    """Arm pose between punches: the normal guard or the high guard."""
    return xp.where(high, HIGH_GUARD, guard_pose)
