"""Slow, auditable CUDA adapter using the original native measurement code.

GPU dynamics are copied back without advancing native physics. This reference
adapter prioritizes identical scoring/trace code, not batched training speed.
It is diagnostic and cannot issue a native promotion gate.
"""
import mujoco_warp as mjwarp
import numpy as np

from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_physics import PairedWarpPhysics
from boxer.warp_readback import WarpReadback


class WarpPairedReference(PairedFoundation):
    def __init__(self, foundation, *, iterations=None, ls_iterations=None, account=None):
        self.physics = None
        self.gpu_substeps = 0
        self.solver_flag_observations = 0
        self.solver_flag_bits_seen = 0
        self.gpu_forward_calls = 0
        self.native_reset_forward_calls = 0
        self.gpu_account = account
        super().__init__(foundation)
        if iterations is not None:
            self.model.opt.iterations = iterations
        if ls_iterations is not None:
            self.model.opt.ls_iterations = ls_iterations
        self.physics = PairedWarpPhysics(self.model)
        self.readback = WarpReadback(self.physics.data)
        # Suppress repetitive solver text, never flags or capacity warnings.
        self.physics.model.opt.warn_overflow &= ~(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
        self.gpu_forward_calls += 1; self._account()
        self.physics.load(self.data.qpos[None], self.data.qvel[None], self.data.ctrl[None])
        self.gpu_substeps += 1
        self._account()
        self.physics.step(); self.physics.health(); self.physics.capture_step()
        self.reset()

    def reset(self, *args, **kwargs):
        self.native_reset_forward_calls += 1; self._account()
        result = super().reset(*args, **kwargs)
        if self.physics is not None:
            self.gpu_forward_calls += 1; self._account()
            self.physics.load(self.data.qpos[None], self.data.qvel[None], self.data.ctrl[None], [self.data.time])
            self._read_state()
            result = self.observation(np.zeros((2, 3)))
        return result

    def _account(self):
        if self.gpu_account is not None:
            self.gpu_account(dict(gpu_world_substeps=self.gpu_substeps,
                gpu_forward_calls=self.gpu_forward_calls,
                native_reset_forward_calls=self.native_reset_forward_calls,
                solver_limit_world_checks=self.solver_flag_observations,
                overflow_bits_seen=self.solver_flag_bits_seen))

    def _read_state(self):
        health = self.physics.health()
        for flags in health['overflow_flags']: self.solver_flag_bits_seen |= flags
        self.solver_flag_observations += health['solver_limit_worlds']
        self.readback.export(self.data, self.model)
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise FloatingPointError('unexpected external assistance')

    def _physics_step(self):
        self.gpu_substeps += 1
        self._account()
        self.physics.step(self.data.ctrl[None])
        self._read_state()
