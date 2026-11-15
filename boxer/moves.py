"""Arm moves shared by the spar generator, batched tactics and native play.

Poses are left-arm (shoulder pitch, roll, yaw, elbow); the right arm mirrors
roll and yaw. Hook and high-guard poses were fitted by forward kinematics
(``tools/fit_moves.py``); fist positions below are in the torso frame.

* straight: guard -> strike in 0.18 s, back in 0.26 s (fist reaches 0.41-0.44 m).
* hook: guard -> wind-up (fist 0.34 m out to the side, far forward) -> a level
  arc through IK waypoints to just past the centre line (fist 0.37-0.41 m),
  0.26 s out and 0.28 s back.
* uppercut: guard -> dip (fist low and close, 35 % of the extension) -> a rise
  through IK waypoints, forearm upright, to the centre line under the chin or
  in the middle of the torso (fist 0.34-0.37 m out), 0.22 s out and 0.26 s back.
* high guard: both fists in front of the face (0.43 m up against the normal
  guard's 0.26 m), held while not punching. Fists and forearms collide
  with incoming punches; only head and torso contacts score.

Every function takes ``xp`` (``numpy`` or ``jax.numpy``) so all callers share
one definition.
"""
from __future__ import annotations

import numpy as np

STRAIGHT, HOOK, UPPERCUT = 0, 1, 2
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
# Uppercut (2026-11-15): from a dip (fist 0.26-0.28 m out, low) the fist rises
# almost vertically, forearm upright, to end on the centre line: under the
# chin (0.34 m out, z 0.42) or mid-torso (0.37 m, z 0.22), where a high
# guard's elbows leave a gap. IK waypoints from ``tools/fit_moves.py --uppercut``.
UPPERCUT_PATH_HEAD = np.array([
    [-0.3198, -0.1175, -0.1817, -0.4197],
    [-0.4213, -0.0047, -0.2355, -0.5273],
    [-0.6565, 0.0057, -0.2038, -0.4147],
    [-0.8885, -0.0001, -0.1515, -0.2156],
    [-1.1171, 0.0006, -0.0918, 0.0578],
    [-1.3620, 0.0136, -0.0300, 0.4232],
])
UPPERCUT_PATH_BODY = np.array([
    [-0.0368, -0.0467, -0.2352, -0.0233],
    [0.1070, 0.1150, -0.3867, -0.3698],
    [-0.0410, 0.1093, -0.4060, -0.3596],
    [-0.2548, 0.0564, -0.3711, -0.2158],
    [-0.5188, 0.0022, -0.2972, 0.0486],
    [-0.8437, -0.0307, -0.1984, 0.4601],
])
UPPERCUT_HEAD, UPPERCUT_BODY = UPPERCUT_PATH_HEAD[-1], UPPERCUT_PATH_BODY[-1]
TIMING = {STRAIGHT: (.18, .26), HOOK: (.26, .28), UPPERCUT: (.22, .26)}   # extension, retraction seconds
WINDUP_FRACTION = .4                                   # share of a hook's extension
DIP_FRACTION = .35                                     # share of an uppercut's extension


def mirror(xp, pose, hand):
    """Left-arm pose -> pose for ``hand`` (0 left, 1 right)."""
    sign = xp.where(hand == 0, 1.0, -1.0)
    return pose * xp.stack([xp.ones_like(sign), sign, sign, xp.ones_like(sign)])


def durations(xp, move, dt):
    """(extension, retraction) in control steps for ``move``."""
    pick = lambda k: xp.where(move == HOOK, TIMING[HOOK][k],
                              xp.where(move == UPPERCUT, TIMING[UPPERCUT][k], TIMING[STRAIGHT][k]))
    extend, retract = pick(0) / dt, pick(1) / dt
    return extend, retract


def strike_pose(xp, move, aim, jitter=0.):
    """Left-arm strike endpoint for ``aim`` in [0 body, 1 head]."""
    straight = STRAIGHT_BODY + aim * (STRAIGHT_HEAD - STRAIGHT_BODY)
    hook = HOOK_BODY + aim * (HOOK_HEAD - HOOK_BODY)
    uppercut = UPPERCUT_BODY + aim * (UPPERCUT_HEAD - UPPERCUT_BODY)
    return xp.where(move == HOOK, hook, xp.where(move == UPPERCUT, uppercut, straight)) + jitter


def punch_pose(xp, move, aim, target, age, rest, dt):
    """Left-arm pose ``age`` steps into a punch toward ``target`` (strike pose).

    ``rest`` is the pose the arm leaves from and returns to (normal or high
    guard). Straights keep the original timing: fraction (age + 1) / extension
    out, then 1 - (age - extension) / retraction back. Hooks and uppercuts go
    to the first waypoint of their path (wind-up or dip), then along it.
    """
    extend, retract = durations(xp, move, dt)
    u = xp.clip((age + 1) / extend, 0., 1.)
    upper = move == UPPERCUT
    path = xp.where(upper, UPPERCUT_PATH_BODY + aim * (UPPERCUT_PATH_HEAD - UPPERCUT_PATH_BODY),
                    HOOK_PATH_BODY + aim * (HOOK_PATH_HEAD - HOOK_PATH_BODY))
    # The strike endpoint is ``target`` (it carries the launch jitter).
    path = xp.concatenate([path[:-1], target[None]], axis=0)
    lead = xp.where(upper, DIP_FRACTION, WINDUP_FRACTION)
    first = xp.clip(u / lead, 0., 1.)
    second = xp.clip((u - lead) / (1 - lead), 0., 1.)
    position = second * (len(HOOK_PATH_HEAD) - 1)
    index = xp.clip(xp.floor(position), 0, len(HOOK_PATH_HEAD) - 2).astype(int)
    swing = path[index] + (position - index) * (path[index + 1] - path[index])
    curved_out = xp.where(u < lead, rest + first * (path[0] - rest), swing)
    straight_out = rest + u * (target - rest)
    out = xp.where(move == STRAIGHT, straight_out, curved_out)
    back = xp.clip(1 - (age - extend) / retract, 0., 1.)
    home = rest + back * (target - rest)
    return xp.where(age < extend, out, home)


def finished(xp, move, age, dt):
    extend, retract = durations(xp, move, dt)
    return age >= extend + retract


def rest_pose(xp, guard_pose, high):
    """Arm pose between punches: the normal guard or the high guard."""
    return xp.where(high, HIGH_GUARD, guard_pose)
