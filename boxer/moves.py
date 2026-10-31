"""Arm moves shared by the spar generator, batched tactics and native play.

Poses are left-arm (shoulder pitch, roll, yaw, elbow); the right arm mirrors
roll and yaw. Hook and high-guard poses were fitted by forward kinematics
(``tools/fit_moves.py``); fist positions below are in the torso frame.

* straight: guard -> strike in 0.18 s, back in 0.26 s (fist reaches 0.41-0.44 m).
* hook: guard -> wind-up (fist 0.30 m out to the side) -> strike across the
  centre line with the elbow wide and level (fist 0.32-0.36 m), 0.26 s out
  and 0.28 s back.
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
HOOK_BODY = np.array([-1.191, .565, -1.17, .388])
HOOK_HEAD = np.array([-2.09, .817, -1.541, .357])
WINDUP_BODY = np.array([.026, .702, .296, -.12])
WINDUP_HEAD = np.array([-1.19, .354, .738, .077])
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
    windup = WINDUP_BODY + aim * (WINDUP_HEAD - WINDUP_BODY)
    first = xp.clip(u / WINDUP_FRACTION, 0., 1.)
    second = xp.clip((u - WINDUP_FRACTION) / (1 - WINDUP_FRACTION), 0., 1.)
    hook_out = xp.where(u < WINDUP_FRACTION, rest + first * (windup - rest),
                        windup + second * (target - windup))
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
