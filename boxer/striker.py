"""One learned striker: footwork and both arms' joint targets from one policy.

The champion stack splits a boxer into tactics (when, with which hand, how
high and which scripted move to throw, ``boxer.tactics_env``), scripted arm
paths (``boxer.moves``) and an arm skill that bends them
(``boxer.arm_skill``). The striker replaces all three with one policy. Every
control step it chooses the footwork (advance and circle; turning toward the
opponent stays automatic, as in ``tactics_env.decode``) and the shoulder and
elbow targets of both arms. There is no punch request, no move list and no
fixed path: a punch is whatever arm motion the policy produces, and it
scores under the same native impact scorecard as any other. The leg motor
still keeps balance.

Arm targets live in a box around every scripted pose (guard, high guard,
straights, hook paths) widened by ``MARGIN``, and move at most ``RATE`` rad
per control step, a little faster than the fastest scripted swing. The right
arm uses the left arm's convention (``moves.mirror``), as in the arm skill.

A striker starts by imitating the champion stack (``boxer.striker_train``)
and is then trained natively (``native_finetune --striker``).
"""
from __future__ import annotations

import functools

import mujoco
import numpy as np

from boxer import moves
from boxer.curriculum_env import LEFT_ARM
from boxer.scene import GUARD_ARMS_BOXING
from boxer.spar_match import PunchArm

STRIKER_OBS_SIZE = 74
STRIKER_ACTION_SIZE = 10          # advance, circle, left arm 4, right arm 4
RATE = .3                         # rad per control step (scripted swings: up to ~0.22)
MARGIN = .35                      # rad beyond the scripted poses (the arm skill's range)
GUARD = np.array(GUARD_ARMS_BOXING["left"][:4])
_POSES = np.vstack([GUARD, moves.HIGH_GUARD, moves.STRAIGHT_BODY, moves.STRAIGHT_HEAD,
                    moves.HOOK_PATH_HEAD, moves.HOOK_PATH_BODY])
POSE_LOW, POSE_HIGH = _POSES.min(0) - MARGIN, _POSES.max(0) + MARGIN
_MID, _HALF = (POSE_HIGH + POSE_LOW) / 2, (POSE_HIGH - POSE_LOW) / 2
# Distance from the guard that counts as a punch under way (rad, joint space):
# about a third of the way to a body straight.
ACTIVE = .35 * np.linalg.norm(moves.STRAIGHT_BODY - GUARD)


def to_unit(pose):
    """Left-convention arm pose(s) -> policy units in [-1, 1]."""
    return np.clip((np.asarray(pose) - _MID) / _HALF, -1, 1)


def from_unit(unit):
    return _MID + _HALF * np.clip(unit, -1, 1)


class StrikerArm(PunchArm):
    """Both arms of a striker corner, driven pose by pose.

    Keeps ``PunchArm``'s interface so brains, arm skills and bouts handle it
    like any arm: it is always ``ready`` (no scripted punch is ever latched)
    and ``features`` describes its motion in the tactics observation's terms.
    """

    def __init__(self, pose, action_scale, dt):
        super().__init__(pose, action_scale, dt)
        self.guards = np.stack([moves.mirror(np, self.pose[LEFT_ARM + 7 * s:LEFT_ARM + 7 * s + 4], s)
                                for s in range(2)])
        self.arm_pose = self.guards.copy()
        self.launched = False

    def command(self, unit):
        """Moves both arm targets toward ``unit`` (8 policy outputs), rate-limited."""
        was = self.features()[1]
        target = from_unit(np.asarray(unit, np.float64).reshape(2, 4))
        self.arm_pose = self.arm_pose + np.clip(target - self.arm_pose, -RATE, RATE)
        self.launched = bool(self.features()[1] and not was)

    def follow(self, arm_actions):
        """Sets the arm targets to 14 executed arm actions (imitation data)."""
        for s in range(2):
            start = LEFT_ARM + 7 * s
            self.arm_pose[s] = moves.mirror(np, self.pose[start:start + 4]
                                            + self.scale * np.asarray(arm_actions[7 * s:7 * s + 4]), s)

    def features(self):
        """(extension, active, hand sign, aim, hook, high guard) like ``PunchArm.features``."""
        reach = np.linalg.norm(self.arm_pose - self.guards, axis=1)
        side = int(np.argmax(reach))
        active = reach[side] > ACTIVE
        high = bool((np.linalg.norm(self.arm_pose - moves.HIGH_GUARD, axis=1) < .3).all())
        return np.array([min(reach[side] / (3 * ACTIVE), 1.) if active else 0., float(active),
                         (2. * side - 1) if active else 0., .5 if active else 0., 0., float(high)])

    def actions(self, high=None):
        arms = np.zeros(14)
        for s in range(2):
            start = LEFT_ARM + 7 * s
            arms[7 * s:7 * s + 4] = (moves.mirror(np, self.arm_pose[s], s) - self.pose[start:start + 4]) / self.scale
        return arms


def striker_observe(pair, corner, arms, clock):
    """Observation for ``corner`` (74 floats).

    The tactics observation (32, ``tactics_env.observe``), then per arm in the
    left arm's convention: joint angles 4, velocities 4 (x0.1), fist relative
    to the opponent's head 3 and torso 3, fist velocity relative to the torso
    3 (all in the torso frame); last, both arms' current targets in policy
    units (8).
    """
    from boxer.spar_match import _body
    from boxer.tactics_env import observe
    bodies = [_body(pair, k) for k in range(2)]
    tactics = observe(np, bodies[corner], bodies[1 - corner], arms[corner].features(),
                      arms[1 - corner].features(), clock)
    model, data = pair.model, pair.data
    site = pair._torso_sites[corner]
    rot = data.site_xmat[site].reshape(3, 3)
    origin = data.site_xpos[site]
    torso_v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, torso_v, 0)
    head, torso = pair._aim_geoms[1 - corner]
    rows = [tactics]
    for hand in range(2):
        flip = np.array([1., -1., 1.]) if hand else np.ones(3)
        sign = np.array([1., -1., -1., 1.]) if hand else np.ones(4)
        joints = slice(LEFT_ARM + 7 * hand, LEFT_ARM + 7 * hand + 4)
        fist = int(pair.fists[corner][hand])
        position = data.geom_xpos[fist]
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, fist, v, 0)
        velocity = rot.T @ (v[3:] - torso_v[3:] - np.cross(torso_v[:3], position - origin))
        rows += [sign * data.qpos[pair.maps[corner]['q'][joints]],
                 sign * data.qvel[pair.maps[corner]['v'][joints]] * .1,
                 flip * (rot.T @ (position - data.geom_xpos[head])),
                 flip * (rot.T @ (position - data.geom_xpos[torso])), flip * velocity]
    rows.append(to_unit(arms[corner].arm_pose).ravel())
    return np.concatenate(rows).astype(np.float32)


def striker_command(action, distance, local):
    """Footwork command from the striker's first two outputs (advance, circle)."""
    from boxer.tactics_env import decode
    tactics = np.array([-1., -1., 0., action[0], action[1], -1., -1.])
    return decode(tactics, distance, local)[0]


def striker_network_factory():
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=(256, 256),
                             value_hidden_layer_sizes=(256, 256))


def striker_net():
    from brax.training.acme import running_statistics
    net = striker_network_factory()(STRIKER_OBS_SIZE, STRIKER_ACTION_SIZE,
                                    preprocess_observations_fn=running_statistics.normalize)
    return net, net.parametric_action_distribution


def is_striker(params):
    from boxer.tactics_env import policy_sizes
    return policy_sizes(params) == (STRIKER_OBS_SIZE, STRIKER_ACTION_SIZE)


class StrikerBrain:
    """A trained striker playing one corner of ``spar_match`` (inference)."""

    def __init__(self, params, deterministic=True):
        import jax
        from brax.training.agents.ppo import networks as ppo_networks
        policy = ppo_networks.make_inference_fn(striker_net()[0])(params, deterministic=deterministic)

        def act(obs, key):
            key, sub = jax.random.split(key)
            return policy(obs, sub)[0], key
        self.act = jax.jit(act)
        self.key = jax.random.PRNGKey(0)

    def new_arm(self, pair):
        return StrikerArm(pair.pose, pair.action_scale, pair.dt)

    def decide_pair(self, pair, i, arms, clock):
        from boxer.spar_match import _body
        obs = striker_observe(pair, i, arms, clock)
        action, self.key = self.act(obs, self.key)
        action = np.asarray(action, np.float64)
        bodies = [_body(pair, k) for k in range(2)]
        delta = bodies[1 - i]['pos'] - bodies[i]['pos']
        command = striker_command(action, np.linalg.norm(delta), bodies[i]['rot'].T @ delta)
        arms[i].command(action[2:])
        return command, None, False
