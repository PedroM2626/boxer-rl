"""Batched native-substep measurements; no physics or acceptance-rule changes.

All kinematic arithmetic uses float64, as the native readback oracle does.
Contact force inputs are local-frame normal forces from the actual solver.
"""
import numpy as np


class PairedBatchMetrics:
    def __init__(self, pair):
        self.model, self.maps = pair.model, pair.maps
        self.fists = pair.fists
        self.targets = [np.array(sorted(t), dtype=int) for t in pair.targets]
        self.dt = float(pair.model.opt.timestep)
        self.torso_sites = [self.model.site(m['side']+'imu_in_torso').id for m in self.maps]
        self.torso_bodies = self.model.site_bodyid[self.torso_sites]
        self.geom_bodies = self.model.geom_bodyid
        self.geom_roots = self.model.body_rootid[self.geom_bodies]
        self.torso_roots = self.model.body_rootid[self.torso_bodies]
        self.elbows = np.array([[self.model.jnt_qposadr[self.model.joint(m['side']+h+'_elbow_joint').id]
            for h in ('left', 'right')] for m in self.maps])
        self.up = [int(self.model.sensor(m['side']+'upvector_torso').adr[0])+2 for m in self.maps]
        self.leg_pairs = []
        for corner, mapping in enumerate(self.maps):
            for a, b in (('left_foot', 'right_foot'), ('left_foot', 'right_shin'), ('right_foot', 'left_shin')):
                self.leg_pairs.append((corner, *sorted((self.model.geom(mapping['side']+a).id,
                                                        self.model.geom(mapping['side']+b).id))))

    def before(self, state):
        position = np.asarray(state['geom_xpos'], np.float64)
        cvel = np.asarray(state['cvel'], np.float64)
        center = np.asarray(state['subtree_com'], np.float64)
        body = cvel[:, self.geom_bodies]
        velocity = body[..., 3:]+np.cross(body[..., :3], position-center[:, self.geom_roots])
        torso_position = np.asarray(state['site_xpos'], np.float64)[:, self.torso_sites]
        motion = cvel[:, self.torso_bodies].copy()
        motion[..., 3:] += np.cross(motion[..., :3], torso_position-center[:, self.torso_roots])
        rotation = np.asarray(state['site_xmat'], np.float64)[:, self.torso_sites].reshape(-1, 2, 3, 3)
        fist_position = position[:, self.fists]
        offset = fist_position-torso_position[:, :, None]
        relative = velocity[:, self.fists]-motion[:, :, None, 3:]-np.cross(motion[:, :, None, :3], offset)
        local_position = np.einsum('ncji,nchj->nchi', rotation, offset)
        local_velocity = np.einsum('ncji,nchj->nchi', rotation, relative)
        distance, toward = [], []
        for corner in (0, 1):
            delta = position[:, self.targets[1-corner]][:, None]-fist_position[:, corner, :, None]
            lengths = np.linalg.norm(delta, axis=-1)
            chosen = np.argmin(lengths, axis=-1)
            nearest = np.take_along_axis(delta, chosen[..., None, None], axis=2)[:, :, 0]
            distance.append(np.min(lengths, axis=-1))
            toward.append(np.sum(relative[:, corner]*nearest, axis=-1)/np.maximum(np.linalg.norm(nearest, axis=-1), 1e-12))
        return dict(position=position, velocity=velocity, relative_world_velocity=relative,
            local_position=local_position, local_velocity=local_velocity,
            elbow=np.asarray(state['qpos'], np.float64)[:, self.elbows],
            toward_nearest_target=np.stack(toward, axis=1), target_center_distance=np.stack(distance, axis=1),
            time_before=np.asarray(state['time'], np.float64))

    def contacts(self, before, world, geoms, forces):
        worlds = len(before['position'])
        world, geoms, forces = np.asarray(world), np.asarray(geoms), np.asarray(forces, np.float64)
        if (world.ndim != 1 or geoms.shape != (len(world), 2) or forces.shape != world.shape
                or not np.issubdtype(world.dtype, np.integer) or not np.issubdtype(geoms.dtype, np.integer)
                or np.any(world < 0) or np.any(world >= worlds) or np.any(geoms < 0)
                or np.any(geoms >= self.model.ngeom) or not np.isfinite(forces).all() or np.any(forces < 0)):
            raise ValueError('finite nonnegative normal forces and valid contact world/geom indices required')
        present = np.zeros((worlds, 2, 2), bool)
        impulse, closing, advancing = (np.zeros((worlds, 2, 2)) for _ in range(3))
        for corner in (0, 1):
            for hand, fist in enumerate(self.fists[corner]):
                first, second = geoms[:, 0] == fist, geoms[:, 1] == fist
                target = np.where(first, geoms[:, 1], geoms[:, 0])
                rows = np.flatnonzero((first | second) & np.isin(target, self.targets[1-corner]))
                w, t = world[rows], target[rows]
                present[w, corner, hand] = True
                np.add.at(impulse[:, corner, hand], w, forces[rows]*self.dt)
                direction = before['position'][w, t]-before['position'][w, fist]
                direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
                relative_target = before['velocity'][w, fist]-before['velocity'][w, t]
                np.maximum.at(closing[:, corner, hand], w, np.sum(relative_target*direction, axis=-1))
                np.maximum.at(advancing[:, corner, hand], w,
                    np.sum(before['relative_world_velocity'][w, corner, hand]*direction, axis=-1))
        return dict(present=present, impulse=impulse, contact_closing=closing, contact_advancing=advancing)

    def fallen(self, state, world, geoms):
        qpos, qvel, sensors = (np.asarray(state[k]) for k in ('qpos', 'qvel', 'sensordata'))
        finite = np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1)
        flags = np.column_stack([~finite | (qpos[:, m['root']+2] < .55) | (sensors[:, self.up[c]] < .6)
                                 for c, m in enumerate(self.maps)])
        world, geoms = np.asarray(world, dtype=int), np.sort(np.asarray(geoms, dtype=int), axis=-1)
        for corner, first, second in self.leg_pairs:
            mask = (geoms[:, 0] == first) & (geoms[:, 1] == second)
            flags[world[mask], corner] = True
        return flags


def metric_snapshot(data, gpu=False):
    keys = ('qpos', 'qvel', 'sensordata', 'geom_xpos', 'site_xmat', 'site_xpos', 'cvel', 'subtree_com')
    result = {k: np.asarray(getattr(data, k).numpy(), np.float64) if gpu
              else np.asarray(getattr(data, k), np.float64)[None].copy() for k in keys}
    result['time'] = np.asarray(data.time.numpy(), np.float64) if gpu else np.asarray([data.time], np.float64)
    return result
