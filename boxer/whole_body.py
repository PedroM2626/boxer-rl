"""One policy for the whole boxer: legs, waist and both arms, end to end.

The stacks so far are hierarchies: a brain (tactics or striker) sends a
footwork command to a separately trained leg motor and sets the arms, and the
motor turns the command into leg and waist targets. Each part trained with
the others frozen, and changing one broke another (docs/striker.md). The
whole-body policy has no command in between: every control step it sets the
leg and waist joint targets itself (15, in the motor's action units) and both
arms' shoulder and elbow targets (8, in the striker's units: the pose box
around every scripted pose, moved at most ``striker.RATE`` per step, wrists
held as in the striker).

It observes what both former layers saw: the striker observation (74: the
opponent, the ring, both arms' motion) and the motor's proprioception
without the command (100: pelvis velocity, gyro, gravity, joint positions and
velocities, last actions, gait clock). A whole-body policy starts by
imitating the striker on its legs (``boxer.whole_body_train``) and is then
trained natively (``native_finetune --whole-body``).
"""
from __future__ import annotations

import functools

import numpy as np

LEGS = 15                             # leg and waist actuators (LEFT_ARM = 15)
WHOLE_BODY_OBS_SIZE = 74 + 100
WHOLE_BODY_ACTION_SIZE = LEGS + 8
COMMAND = slice(9, 12)                # the footwork command inside a motor observation


def body_observation(pair, corner):
    """The motor observation of ``corner`` without its command (100)."""
    row = pair.observation(np.zeros((2, 3)))[corner]
    return np.concatenate([row[:COMMAND.start], row[COMMAND.stop:]])


def whole_body_observe(pair, corner, arms, clock):
    from boxer.striker import striker_observe
    return np.concatenate([striker_observe(pair, corner, arms, clock),
                           body_observation(pair, corner)]).astype(np.float32)


def apply_arms(arm, action):
    """Moves a ``StrikerArm`` toward the policy's 8 arm outputs; returns the 15 leg outputs."""
    action = np.asarray(action, np.float64)
    arm.command(action[LEGS:])
    return action[:LEGS]


def whole_body_network_factory():
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=(512, 256, 256),
                             value_hidden_layer_sizes=(512, 256, 256))


def whole_body_net():
    from brax.training.acme import running_statistics
    net = whole_body_network_factory()(WHOLE_BODY_OBS_SIZE, WHOLE_BODY_ACTION_SIZE,
                                       preprocess_observations_fn=running_statistics.normalize)
    return net, net.parametric_action_distribution


def is_whole_body(params):
    from boxer.tactics_env import policy_sizes
    return policy_sizes(params) == (WHOLE_BODY_OBS_SIZE, WHOLE_BODY_ACTION_SIZE)


class WholeBodyBrain:
    """A trained whole-body policy playing one corner of ``spar_match`` (inference).

    ``decide_pair`` moves the arms and keeps the step's leg and waist targets
    in ``legs``; the bout puts them over the motor's row for this corner. The
    command it returns is zero: no motor drives this corner.
    """

    def __init__(self, params, deterministic=True):
        import jax
        from brax.training.agents.ppo import networks as ppo_networks
        policy = ppo_networks.make_inference_fn(whole_body_net()[0])(params, deterministic=deterministic)

        def act(obs, key):
            key, sub = jax.random.split(key)
            return policy(obs, sub)[0], key
        self.act = jax.jit(act)
        self.key = jax.random.PRNGKey(0)
        self.legs = np.zeros(LEGS)

    def new_arm(self, pair):
        from boxer.striker import StrikerArm
        return StrikerArm(pair.pose, pair.action_scale, pair.dt)

    def decide_pair(self, pair, i, arms, clock):
        action, self.key = self.act(whole_body_observe(pair, i, arms, clock), self.key)
        self.legs = apply_arms(arms[i], action)
        return np.zeros(3), None, False
