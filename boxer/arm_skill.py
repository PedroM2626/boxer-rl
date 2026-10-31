"""Learned punch corrections: an arm-skill policy on top of the scripted moves.

The scripted moves (``boxer.moves``) swing the arm along a fixed joint-space
path to a fixed pose, wherever the opponent actually is. The arm skill sees
the punching fist relative to the target it was asked to hit (the opponent's
torso or head, chosen by the punch's aim) and adds a residual of up to
``RESIDUAL_SCALE`` rad to the punching arm's shoulder and elbow targets on
every control step of the punch. The trajectory is therefore learned and
corrected in real time, while staying close to what the leg motor was
trained to balance through.

One policy serves both hands: observations and residuals of the right arm
are mirrored into the left arm's frame (``moves.mirror``). Tactics still
choose when, with which hand, how high and which move; the motor still
keeps balance.
"""
from __future__ import annotations

import mujoco
import numpy as np

from boxer import moves
from boxer.curriculum_env import LEFT_ARM

ARM_OBS_SIZE = 19
ARM_ACTION_SIZE = 4                  # shoulder pitch, roll, yaw, elbow
RESIDUAL_SCALE = .35                 # rad


def arm_observe(pair, corner, arm):
    """Observation for ``corner``'s punching arm, in its torso frame, mirrored to the left.

    (fist - aimed target point) 3, fist velocity relative to the torso 3,
    target velocity relative to the torso 3, arm joint angles 4 and
    velocities 4, punch progress (extension fraction, overall fraction) 2.
    The aimed point is the opponent's torso (aim 0) to head (aim 1) centre.
    """
    model, data = pair.model, pair.data
    hand = arm.hand
    site = pair._torso_sites[corner]
    rot = data.site_xmat[site].reshape(3, 3)
    origin = data.site_xpos[site]
    torso_v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, torso_v, 0)

    def velocity(geom, point):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, v, 0)
        return rot.T @ (v[3:] - torso_v[3:] - np.cross(torso_v[:3], point - origin))

    fist = int(pair.fists[corner][hand])
    head, torso = pair._aim_geoms[1 - corner]
    target = (1 - arm.aim) * data.geom_xpos[torso] + arm.aim * data.geom_xpos[head]
    flip = np.array([1., -1., 1.]) if hand == 1 else np.ones(3)
    q_adr = pair.maps[corner]['q'][LEFT_ARM + 7 * hand:LEFT_ARM + 7 * hand + 4]
    v_adr = pair.maps[corner]['v'][LEFT_ARM + 7 * hand:LEFT_ARM + 7 * hand + 4]
    joint_sign = np.array([1., -1., -1., 1.]) if hand == 1 else np.ones(4)
    extend, retract = moves.durations(np, arm.move, arm.dt)
    return np.concatenate([
        flip * (rot.T @ (data.geom_xpos[fist] - target)),
        flip * velocity(fist, data.geom_xpos[fist]),
        flip * velocity(torso if arm.aim < .5 else head, target),
        joint_sign * data.qpos[q_adr], joint_sign * data.qvel[v_adr] * .1,
        [min(max(arm.age, 0) / extend, 1.), max(arm.age, 0) / (extend + retract)],
    ]).astype(np.float32)


def arm_network_factory():
    """Brax PPO networks for the arm skill (19 -> 4, tanh-squashed Gaussian)."""
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=(128, 128),
                             value_hidden_layer_sizes=(128, 128))


def arm_net():
    from brax.training.acme import running_statistics
    net = arm_network_factory()(ARM_OBS_SIZE, ARM_ACTION_SIZE,
                                preprocess_observations_fn=running_statistics.normalize)
    return net, net.parametric_action_distribution


def initial_arm_params(key, scale_bias=-1.5):
    """Fresh arm-skill params whose residual starts at zero.

    The policy's last layer starts at zero, so every mean residual is 0 and
    the scripted punch is unchanged; the action scale starts at
    softplus(``scale_bias``) + 0.001 (0.2), i.e. about 0.07 rad of exploration.
    """
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    net, _ = arm_net()
    k1, k2 = jax.random.split(key)
    policy, value = net.policy_network.init(k1), net.value_network.init(k2)
    layers = policy["params"]
    last = sorted(layers, key=lambda name: int(name.split("_")[1]))[-1]
    layers[last]["kernel"] = jp.zeros_like(layers[last]["kernel"])
    layers[last]["bias"] = jp.concatenate([jp.zeros(ARM_ACTION_SIZE),
                                           jp.full(ARM_ACTION_SIZE, scale_bias)])
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((ARM_OBS_SIZE,), jp.float32))
    return norm, policy, value


class ArmSkill:
    """A trained arm-skill policy steering one corner's punches (inference only)."""

    def __init__(self, params, deterministic=True):
        import jax
        from brax.training.agents.ppo import networks as ppo_networks
        net, _ = arm_net()
        policy = ppo_networks.make_inference_fn(net)(params, deterministic=deterministic)
        self.act = jax.jit(lambda obs, key: policy(obs, key)[0])
        self.key = jax.random.PRNGKey(0)

    def steer(self, pair, corner, arm):
        """Sets ``arm.residual`` for this step (zero when not punching)."""
        if arm.ready:
            arm.residual = np.zeros(ARM_ACTION_SIZE)
            return
        obs = arm_observe(pair, corner, arm)
        arm.residual = RESIDUAL_SCALE * np.asarray(self.act(obs, self.key), np.float64)
