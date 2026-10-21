"""Hierarchical boxing with physical impacts and explicit balance assistance.

PPO chooses a lead/rear punch, head/body aim, advance, circle and guard. A motor
controller executes complete extension/retraction cycles using the G1's actual
position actuators. A bounded spring/damper supports the floating pelvis; roots
are never reset, welded or teleported during a round. This is ASSISTED boxing,
not a learned walking or balance policy. The original WarpBoxing stays unchanged.
"""
from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import numpy as np

from boxer import scene
from boxer.warp_env import WarpBoxing, _quat_mul


class SupportedBoxing(WarpBoxing):
    """Low-level physical controller used only by SkillsBoxing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Forearms block physically, but only a fist can score in this mode.
        for side in scene.SIDES:
            for name in scene.FOREARM_GEOMS:
                self._striker[self._model.geom(side + name).id] = False
            for name in scene.BODY_TARGET_GEOMS:
                self._target_w[self._model.geom(side + name).id] = 0

    def reset(self, seed=0):
        obs = super().reset(seed)
        self.support_target = np.stack([
            self._qpos_sync[:, self._sides[s]["free_adr"]:
                                self._sides[s]["free_adr"] + 3]
            for s in scene.SIDES], axis=1).copy()
        return obs

    def _randomize(self, rng, rows):
        obs = super()._randomize(rng, rows)
        if hasattr(self, "support_target"):
            for i, side in enumerate(scene.SIDES):
                a = self._sides[side]["free_adr"]
                self.support_target[rows, i] = self._qpos_sync[rows, a:a + 3]
        return obs

    def _support(self):
        q, v = self._d.qpos.numpy(), self._d.qvel.numpy()
        forces = np.zeros((self.nworld, self._model.nv), np.float32)
        for i, side in enumerate(scene.SIDES):
            a = self._sides[side]["free_adr"]
            jid = self._model.joint(side + "floating_base_joint").id
            b = int(self._model.jnt_dofadr[jid])
            position, orientation = q[:, a:a + 3], q[:, a + 3:a + 7]
            linear = (self.support_target[:, i] - position) * [1200, 1200, 3000]
            linear -= v[:, b:b + 3] * [120, 120, 180]
            forces[:, b:b + 3] = np.clip(linear, -500, 500)
            other = self._sides[scene.SIDES[1 - i]]["free_adr"]
            delta = q[:, other:other + 2] - position[:, :2]
            yaw = np.arctan2(delta[:, 1], delta[:, 0])
            # Error quaternion current^-1 * desired, expressed in the free
            # joint's local angular coordinates (MuJoCo qvel convention).
            desired = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                                np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)
            error = _quat_mul(orientation * [1, -1, -1, -1], desired)
            error *= np.where(error[:, :1] < 0, -1, 1)
            torque = 360 * error[:, 1:] - 35 * v[:, b + 3:b + 6]
            forces[:, b + 3:b + 6] = np.clip(torque, -120, 120)
        self._d.qfrc_applied.assign(forces)

    def _advance(self):
        hit = np.zeros((self.nworld, 2))
        before = self._d.sensordata.numpy()
        for _ in range(self.n_frames):
            self._support()
            mjwarp.step(self._m, self._d)
            sensors = self._d.sensordata.numpy()
            # Frame qpos/qvel are not used by the scorecard's motion gate.
            hit = np.maximum(hit, self._scorecard(sensors, before))
            before = sensors
        return hit


class SkillsBoxing:
    """Six bounded decisions per boxer; a shared policy controls both corners.

    Actions: lead, rear, aim (-1 body / +1 head), advance, circle, guard.
    A punch is latched at launch: subsequent policy samples cannot turn its
    retraction into another strike. Stunned corners cannot launch new punches.
    """
    action_size = 6
    cycle_steps = 38

    def __init__(self, nworld=256, n_frames=5, spawn_distance=0.62,
                 round_steps=600, damage_cap=6.0, stun_steps=10,
                 spawn_jitter=0.025, yaw_jitter=0.08, pose_noise=0.005,
                 action_scale=1.0, guard_pose="boxing", imitation_path=None,
                 w_imit=0.3):
        if action_scale != 1.0 or guard_pose != "boxing":
            raise ValueError("skills require action_scale=1 and the boxing guard")
        if imitation_path is not None:
            raise ValueError("joint imitation is only supported by the direct environment")
        self.physics = SupportedBoxing(
            nworld=nworld, n_frames=n_frames, spawn_distance=spawn_distance,
            guard_pose="boxing", action_scale=1.0, round_steps=round_steps,
            damage_cap=damage_cap, stun_steps=stun_steps,
            spawn_jitter=spawn_jitter, yaw_jitter=yaw_jitter, pose_noise=pose_noise,
            target_range=0.54, punch_reach=0.43, clinch_range=0.30,
            rearm_steps=12, retract_radius=0.12, w_attempt=0.03,
            w_upright=0.05, w_alive=0.0, w_range=0.3)
        self.nworld, self.n_frames = nworld, n_frames
        self.control_dt = n_frames * self.physics._model.opt.timestep
        self.cycle_steps = max(1, int(np.ceil(.76 / self.control_dt)))
        self.obs_size = self.physics.obs_size + 9
        self.age = np.full((nworld, 2), -1, np.int32)
        self.hand = np.zeros((nworld, 2), np.int32)
        self.aim = np.zeros((nworld, 2), np.float32)
        self.last_action = np.zeros((nworld, 2, 6), np.float32)
        self._columns = {}
        for side in scene.SIDES:
            names = [mujoco.mj_id2name(self.physics._model,
                     mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))[len(side):]
                     for a in self.physics._sides[side]["actuators"]]
            self._columns[side] = {name: i for i, name in enumerate(names)}

    def __getattr__(self, name):
        return getattr(self.physics, name)

    def _observe(self, obs):
        active = (self.age >= 0).astype(np.float32)
        return np.concatenate([obs, (np.maximum(self.age, 0) /
                               self.cycle_steps)[..., None],
                               (active * (2 * self.hand - 1))[..., None],
                               self.aim[..., None], self.last_action], axis=-1)

    def reset(self, seed=0):
        self.age.fill(-1)
        self.hand.fill(0)
        self.aim.fill(0)
        self.last_action.fill(0)
        return self._observe(self.physics.reset(seed))

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.nworld, 2, self.action_size):
            raise ValueError(f"expected {(self.nworld, 2, self.action_size)}, "
                             f"got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        actions = np.clip(actions, -1, 1)
        p = self.physics
        choice = np.argmax(actions[..., :2], axis=-1)
        launch = ((self.age < 0) & (actions[..., :2].max(axis=-1) > 0.15)
                  & (p._stun == 0) & (actions[..., 5] < 0.65))
        self.hand[launch], self.aim[launch] = choice[launch], actions[..., 2][launch]
        self.age[launch] = 0
        active = self.age >= 0
        # Extend for 0.18 s, retract for 0.26 s, then let the motors settle
        # at guard before this corner is allowed to launch the next punch.
        extend = max(1, int(np.ceil(.18 / self.control_dt)))
        retract = max(1, int(np.ceil(.26 / self.control_dt)))
        extension = np.where(self.age < extend, (self.age + 1) / extend,
                              np.maximum(0, 1 - (self.age - extend) / retract))
        extension = np.clip(extension, 0, 1) * active
        low = np.zeros((self.nworld, 2, p.action_size), np.float32)
        for i, side in enumerate(scene.SIDES):
            cols = self._columns[side]
            for hand_index, (hand, sign) in enumerate((("left", 1), ("right", -1))):
                # FK-checked targets, with straight wrists: body and head share
                # the same physical fist geom and contact-force scorecard.
                head = (self.aim[:, i] + 1) / 2
                target = np.stack([-1.18 - .76 * head,
                                   sign * (.04 + .025 * head),
                                   sign * (.10 + .10 * head),
                                   1.20 + .31 * head], axis=-1)
                fraction = extension[:, i] * (self.hand[:, i] == hand_index)
                guard = np.asarray(scene.GUARD_ARMS_BOXING[hand][:4])
                for k, joint in enumerate(scene.ARM_JOINTS[:4]):
                    col = cols[f"{hand}_{joint}_joint"]
                    low[:, i, col] = fraction * (target[:, k] - guard[k])
            # The support target moves slowly; the bodies still respond to
            # impacts through the solver. Assistance is bounded, not a pin.
            a = p._sides[side]["free_adr"]
            b = p._sides[scene.SIDES[1 - i]]["free_adr"]
            delta = p._qpos_sync[:, b:b + 2] - p._qpos_sync[:, a:a + 2]
            distance = np.linalg.norm(delta, axis=1)
            direction = delta / (distance[:, None] + 1e-6)
            forward = actions[:, i, 3] * 0.10
            forward = np.where((distance < .49) & (forward > 0), 0, forward)
            lateral = actions[:, i, 4] * 0.07
            tangent = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
            p.support_target[:, i, :2] += (forward[:, None] * direction
                                          + lateral[:, None] * tangent) * self.control_dt
            p.support_target[:, i, :2] = np.clip(p.support_target[:, i, :2], -1.3, 1.3)
        obs, reward, done, parts = p.step(low)
        # A small positive-sum clean-strike term supplies a sparring curriculum:
        # trading real punches can improve both agents before either can win.
        # The larger scored-minus-taken term and knockout remain competitive.
        reward += 4.0 * parts["scored_by"]
        self.age[active] += 1
        self.age[self.age >= self.cycle_steps] = -1
        self.last_action[:] = actions
        self.age[done] = -1
        self.hand[done] = 0
        self.aim[done] = 0
        self.last_action[done] = 0
        parts["launched"] = launch.copy()
        return self._observe(obs), reward, done, parts


def scripted_actions(env, step):
    """Reproducible sparring baseline, explicitly a controller, not a policy."""
    action = np.zeros((env.nworld, 2, env.action_size), np.float32)
    for i in (0, 1):
        turn = (step + env.cycle_steps // 2 * i) // env.cycle_steps
        action[:, i, turn % 2] = 0.9
        action[:, i, 2] = 1.0 if turn % 3 else -1.0
        p = env.physics
        a, b = [p._sides[s]["free_adr"] for s in scene.SIDES]
        distance = np.linalg.norm(p._qpos_sync[:, a:a + 2] -
                                  p._qpos_sync[:, b:b + 2], axis=1)
        action[:, i, 3] = np.clip((distance - .53) * 12, -1, 1)
    return action
