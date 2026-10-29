"""Batched NumPy observations with the unchanged paired policy input ordering.

These functions consume engine snapshots. They do not advance dynamics, decide
strike validity, or manufacture contact history.
"""
import numpy as np


class PairedFeatures:
    def __init__(self, pair, engagement_distance=.58):
        if not np.isfinite(engagement_distance) or engagement_distance <= 0:
            raise ValueError('positive engagement distance required')
        self.engagement_distance = float(engagement_distance)
        self.model = pair.model
        self.maps = pair.maps
        self.pose = pair.pose
        self.fists = pair.fists
        self.imu = [self.model.site(m['side']+'imu_in_pelvis').id for m in self.maps]
        self.targets = [[self.model.geom(m['side']+name).id for name in
                        ('head_collision', 'torso_collision')] for m in self.maps]

    def sensor(self, snapshot, side, name):
        sensor = self.model.sensor(side+name)
        start, width = int(sensor.adr[0]), int(sensor.dim[0])
        return snapshot['sensordata'][:, start:start+width]

    def commands(self, snapshot, offsets=None):
        roots = [snapshot['qpos'][:, m['root']:m['root']+2] for m in self.maps]
        delta = roots[1]-roots[0]
        distance = np.linalg.norm(delta, axis=-1)
        speed = np.clip((distance-self.engagement_distance)*.8, 0., .45)
        result = []
        for corner, site in enumerate(self.imu):
            rotation = snapshot['site_xmat'][:, site].reshape(-1, 3, 3)
            local = np.einsum('nji,nj->ni', rotation[:, :2, :2], delta if corner == 0 else -delta)
            result.append(np.column_stack((speed[:, None]*local/np.maximum(distance[:, None], 1e-6),
                                          np.clip(np.arctan2(local[:, 1], local[:, 0]), -.5, .5))))
        result = np.stack(result, axis=1)
        if offsets is not None:
            result += offsets*np.array([.15, .15, .25])
        return np.clip(result, [-.35, -.25, -.5], [.55, .25, .5])

    def motor_observation(self, snapshot, commands, last_action, phase):
        rows = []
        for corner, mapping in enumerate(self.maps):
            side = mapping['side']
            rotation = snapshot['site_xmat'][:, self.imu[corner]].reshape(-1, 3, 3)
            gravity = -rotation[:, 2, :]
            rows.append(np.concatenate((self.sensor(snapshot, side, 'local_linvel_pelvis'),
                self.sensor(snapshot, side, 'gyro_pelvis'), gravity, commands[:, corner],
                snapshot['qpos'][:, mapping['q']]-self.pose, snapshot['qvel'][:, mapping['v']],
                last_action[:, corner], np.cos(phase[:, corner]), np.sin(phase[:, corner])), axis=-1))
        return np.stack(rows, axis=1).astype(np.float32)

    def striking_observation(self, snapshot, last_action, phase, previous, contact_history):
        base = self.motor_observation(snapshot, self.commands(snapshot), last_action, phase)
        rows = []
        for corner, mapping in enumerate(self.maps):
            other = self.maps[1-corner]
            origin = snapshot['qpos'][:, mapping['root']:mapping['root']+3]
            rotation = snapshot['site_xmat'][:, self.imu[corner]].reshape(-1, 3, 3)
            def local(position):
                return np.einsum('nji,nj->ni', rotation, position-origin)
            relative = [local(snapshot['qpos'][:, other['root']:other['root']+3])]
            relative += [local(snapshot['geom_xpos'][:, g]) for g in self.targets[1-corner]]
            relative += [local(snapshot['geom_xpos'][:, g]) for c in (corner, 1-corner) for g in self.fists[c]]
            velocity = (self.sensor(snapshot, other['side'], 'global_linvel_pelvis')
                        - self.sensor(snapshot, mapping['side'], 'global_linvel_pelvis'))
            relative.append(np.einsum('nji,nj->ni', rotation, velocity))
            upright = np.column_stack([self.sensor(snapshot, m['side'], 'upvector_torso')[:, 2]
                                       for m in (mapping, other)])
            rows.append(np.concatenate([base[:, corner], *relative, upright, previous[:, corner],
                                        contact_history[:, corner]], axis=-1))
        result = np.stack(rows, axis=1).astype(np.float32)
        if result.shape[1:] != (2, 148) or not np.isfinite(result).all():
            raise FloatingPointError('invalid batched striking observation')
        return result


def snapshot(data, gpu=False):
    keys = ('qpos', 'qvel', 'sensordata', 'site_xmat', 'geom_xpos')
    return {key: getattr(data, key).numpy() if gpu else np.asarray(getattr(data, key))[None].copy()
            for key in keys}
