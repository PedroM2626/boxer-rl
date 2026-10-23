"""Opponent-aware arm residuals over a frozen, unassisted full-body actor."""
from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from boxer import scene
from boxer.curriculum_train import inference
from boxer.paired_foundation import PairedFoundation


class StrikingEnv:
    action_size = 17  # local command offsets and fourteen arm joint residuals
    observation_size = 146
    residual_scale = .8  # radians; filtered at the 20 ms control interval

    def __init__(self, foundation, params, motor=None, pair=None):
        self.pair = pair or PairedFoundation(foundation)
        self.motor = motor or jax.jit(inference(foundation, params))
        self.arm_indices = np.array([i for i in range(29)
            if any(name in foundation.mj_model.actuator(i).name for name in
                   ('shoulder', 'elbow', 'wrist'))])
        if len(self.arm_indices) != 14:
            raise ValueError('fourteen arm joints required; residuals cannot control legs')
        self.reset()

    def reset(self, seed=0, swap=False):
        self.pair.reset(seed=seed, swap=swap)
        self.previous = np.zeros((2, 17))
        self.residual = np.zeros((2, 14))
        self.age = 0
        self.previous_distance = self.hand_distances()
        return self.observation()

    def commands(self, actions=None):
        command = self.pair.approach_commands(stop_distance=.58)
        if actions is not None:
            command += actions[:, :3] * [.15, .15, .25]
        return np.clip(command, [-.35, -.25, -.5], [.55, .25, .5])

    def hand_distances(self):
        data, model = self.pair.data, self.pair.model
        distances = []
        for i, side in enumerate(scene.SIDES):
            targets = [data.geom_xpos[g] for g in self.pair.targets[1-i]]
            distances.append(min(np.linalg.norm(data.geom_xpos[g] - target)
                for g in self.pair.fists[i] for target in targets))
        return np.asarray(distances)

    def observation(self):
        p = self.pair
        base = p.observation(self.commands())
        rows = []
        for i, m in enumerate(p.maps):
            other = p.maps[1-i]
            origin = p.data.qpos[m['root']:m['root']+3]
            rotation = p.data.site(m['side']+'imu_in_pelvis').xmat.reshape(3,3)
            def local(position):
                return rotation.T @ (position - origin)
            relative = [local(p.data.qpos[other['root']:other['root']+3])]
            for target in ('head_collision', 'torso_collision'):
                relative.append(local(p.data.geom_xpos[p.model.geom(other['side']+target).id]))
            for corner in (i,1-i):
                relative.extend(local(p.data.geom_xpos[g]) for g in p.fists[corner])
            velocity = (p.data.sensor(other['side']+'global_linvel_pelvis').data
                        - p.data.sensor(m['side']+'global_linvel_pelvis').data)
            relative.append(rotation.T @ velocity)
            upright = [p.data.sensor(z['side']+'upvector_torso').data[2] for z in (m,other)]
            rows.append(np.concatenate([base[i], *relative, upright, self.previous[i]]))
        result = np.asarray(rows, np.float32)
        if result.shape != (2, self.observation_size) or not np.isfinite(result).all():
            raise FloatingPointError('invalid opponent-aware observation')
        return result

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (2,17) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.00001):
            raise ValueError('finite bounded (2, 17) actions required')
        commands = self.commands(actions)
        obs = self.pair.observation(commands)
        motor, _ = self.motor({'state':jp.array(obs),'privileged_state':jp.zeros((2,216))},
                               jax.random.PRNGKey(0))
        motor = np.asarray(motor).copy()
        self.residual = .8 * self.residual + .2 * actions[:,3:]
        motor[:,self.arm_indices] += self.residual_scale / self.pair.action_scale * self.residual
        outcome = self.pair.step(motor)
        self.previous = actions.copy()
        self.age += 1
        distances = self.hand_distances()
        progress = np.clip(self.previous_distance - distances, -.1, .1)
        self.previous_distance = distances
        roots = [self.pair.data.qpos[m['root']:m['root']+2] for m in self.pair.maps]
        distance = float(np.linalg.norm(roots[1]-roots[0]))
        both_up = not outcome['done']
        facing = []
        for i,m in enumerate(self.pair.maps):
            forward=self.pair.data.site(m['side']+'imu_in_pelvis').xmat.reshape(3,3)[:2,0]
            facing.append(float(forward @ (roots[1-i]-roots[i]) / max(distance,1e-6)))
        reward = (2.*outcome['scored'] - 2.*outcome['scored'][::-1]
                  - 5.*outcome['fallen'].astype(float))
        if both_up:
            reward += .5*progress + self.pair.dt*(
                .1*np.asarray(facing) + .15*np.exp(-((distance-.6)/.2)**2)
                - .03*np.mean(self.residual**2,axis=1))
        timed_out = self.age >= 500
        outcome.update(distance=distance, timed_out=timed_out,
                       applied_actions=motor, arm_residual=self.residual.copy())
        return self.observation(), reward.astype(np.float32), outcome['done'] or timed_out, outcome
