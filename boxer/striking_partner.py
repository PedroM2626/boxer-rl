"""Single-learner striking practice against a physically free neutral partner.

Neutral means zero high-level residuals, not a stationary or supported body.
The original frozen motor still balances and approaches with all 29 joints.
"""
import numpy as np

from boxer.striking_env import StrikingEnv


class PartnerStrikingEnv:
    observation_size = 146
    action_size = 17
    learners_per_world = 1
    opponent_mode = 'neutral_motor'

    def __init__(self, *args, **kwargs):
        self.env = StrikingEnv(*args, **kwargs)
        self.learner_corner = 0

    @property
    def motor(self):
        return self.env.motor

    @motor.setter
    def motor(self, value):
        self.env.motor = value

    def reset(self, seed=0, swap=False):
        self.learner_corner = int(seed % 2)
        obs = self.env.reset(seed, swap)
        return obs[self.learner_corner:self.learner_corner+1]

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (1, 17):
            raise ValueError('one learner action row required')
        paired = np.zeros((2, 17), dtype=actions.dtype)
        paired[self.learner_corner] = actions[0]
        obs, reward, done, outcome = self.env.step(paired)
        outcome['learner_corner'] = self.learner_corner
        i = self.learner_corner
        return obs[i:i+1], reward[i:i+1], done, outcome
