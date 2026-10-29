"""Experimental CUDA dynamics for the current paired model.

This is a physics primitive, not a certified environment or a PPO replacement.
Native scratch arrays are deliberately not transferred (see the efm_fluid audit).
Solver limit flags remain observable; all other overflow flags are fatal.
"""
from __future__ import annotations

import mujoco_warp as mjwarp
import numpy as np
import warp as wp


class PairedWarpPhysics:
    def __init__(self, model, worlds=1, contacts_per_world=256, constraints_per_world=2048):
        if worlds < 1 or contacts_per_world < 1 or constraints_per_world < 1:
            raise ValueError('positive world and capacity counts required')
        if model.neq or np.any(model.body_gravcomp):
            raise ValueError('unassisted model required')
        self.native_model = model
        self.worlds = worlds
        self.step_graph = None
        self.forward_graph = None
        self.device = wp.get_device('cuda:0')
        with wp.ScopedDevice(self.device):
            self.model = mjwarp.put_model(model)
            self.data = mjwarp.make_data(model, nworld=worlds,
                nconmax=contacts_per_world, njmax=constraints_per_world)

    def load(self, qpos, qvel, ctrl, time=None):
        """Load a complete batch into fresh solver state; never copy native scratch."""
        model = self.native_model
        values = [(self.data.qpos, qpos, model.nq), (self.data.qvel, qvel, model.nv),
                  (self.data.ctrl, ctrl, model.nu)]
        checked = []
        for target, value, width in values:
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (self.worlds, width) or not np.isfinite(value).all():
                raise ValueError('finite physical state with exact batch shape required')
            checked.append((target, value))
        time = np.zeros(self.worlds, np.float32) if time is None else np.asarray(time, np.float32)
        if time.shape != (self.worlds,) or not np.isfinite(time).all():
            raise ValueError('finite per-world time required')
        # Reset all data, including warm starts, forces and contact history.
        with wp.ScopedDevice(self.device):
            mjwarp.reset_data(self.model, self.data)
            for target, value in checked:
                target.assign(value)
            self.data.time.assign(time)
            if self.forward_graph is None:
                mjwarp.forward(self.model, self.data)
            else:
                wp.capture_launch(self.forward_graph)

    def step(self, ctrl=None):
        if ctrl is not None:
            ctrl = np.asarray(ctrl, np.float32)
            if ctrl.shape != (self.worlds, self.native_model.nu) or not np.isfinite(ctrl).all():
                raise ValueError('finite complete actuator targets required')
            self.data.ctrl.assign(ctrl)
        with wp.ScopedDevice(self.device):
            if self.step_graph is None:
                mjwarp.step(self.model, self.data)
            else:
                wp.capture_launch(self.step_graph)

    def capture_step(self):
        """Compile after an explicitly accounted warm-up step, before reset/load."""
        with wp.ScopedCapture(device=self.device) as capture:
            mjwarp.step(self.model, self.data)
        self.step_graph = capture.graph

    def capture_forward(self):
        """Compile the initialized reset-forward path without physical stepping."""
        with wp.ScopedCapture(device=self.device) as capture:
            mjwarp.forward(self.model, self.data)
        self.forward_graph = capture.graph

    def health(self, readback=None, synchronized=False):
        """Synchronize and reject physical corruption or dropped constraints."""
        if not synchronized: wp.synchronize_device(self.device)
        data = self.data if readback is None else readback
        flags = data.overflow.numpy()
        solver_mask = int(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
        if np.any(flags & ~solver_mask):
            raise FloatingPointError(f'Warp capacity/physics overflow: {flags.tolist()}')
        for key in ('qpos', 'qvel', 'qacc', 'sensordata'):
            if not np.isfinite(getattr(data, key).numpy()).all():
                raise FloatingPointError(f'nonfinite Warp {key}')
        return dict(overflow_flags=flags.tolist(), solver_limit_worlds=int(np.count_nonzero(flags & solver_mask)),
                    contacts=int(data.nacon.numpy()[0]), contact_capacity=int(self.data.naconmax),
                    max_constraints=int(data.nefc.numpy().max()))

    def prepare_packed_readback(self):
        """Allocate reusable contact extraction and explicit pinned state views."""
        from boxer.warp_readback import SelectedWarpReadback
        capacity = int(self.data.naconmax)
        with wp.ScopedDevice(self.device):
            self._packed_force = wp.zeros(capacity, dtype=wp.spatial_vector)
            self._packed_contact_ids = wp.array(np.arange(capacity, dtype=np.int32))
        self._packed = SelectedWarpReadback(self.data,
            ('qpos', 'qvel', 'sensordata', 'geom_xpos', 'site_xmat', 'site_xpos',
             'cvel', 'subtree_com', 'time', 'qacc', 'overflow', 'nacon', 'nefc',
             'qfrc_applied', 'xfrc_applied', 'contact.worldid', 'contact.geom'),
            extras=dict(normal_force=self._packed_force))

    def packed_readback(self):
        """Extract actual solver forces then read all required arrays in one sync.

        Upstream contact_force_kernel guards contactid >= nacon. Stale unused
        force entries are never read; capacity/finite checks still run every step.
        """
        with wp.ScopedDevice(self.device):
            mjwarp.contact_force(self.model, self.data, self._packed_contact_ids, False, self._packed_force)
        return self._packed.refresh()

    def normal_forces(self, readback=None):
        data = self.data if readback is None else readback
        count = int(data.nacon.numpy()[0])
        if count > self.data.naconmax:
            raise FloatingPointError('contact capacity exceeded')
        if count == 0:
            return np.empty(0, np.int32), np.empty((0, 2), np.int32), np.empty(0, np.float32)
        if readback is not None:
            return (data.contact.worldid.numpy()[:count], data.contact.geom.numpy()[:count],
                    np.maximum(0., data.normal_force.numpy()[:count, 0]))
        with wp.ScopedDevice(self.device):
            force = wp.zeros(count, dtype=wp.spatial_vector)
            mjwarp.contact_force(self.model, self.data,
                wp.array(np.arange(count, dtype=np.int32)), False, force)
        return (self.data.contact.worldid.numpy()[:count],
                self.data.contact.geom.numpy()[:count], np.maximum(0., force.numpy()[:, 0]))
