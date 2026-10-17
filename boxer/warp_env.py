"""The two-boxer scene as a batched mujoco_warp environment (numpy-native).

This is the port of ``boxer/env.py``'s physics loop off MJX, decided after the
2026-10-17 checks: warp's sensordata matches host mj_forward to 1e-7, 2048
worlds step at 4,275 control steps/s in 1.95 GB, and the force-threshold
scorecard fires on a scripted jab (``tools/warp_jab_force.py``). The MJX env
could not read contact force at all, which capped every scorecard we could
write on it.

Differences from the MJX env, each tied to a measurement:

* the distance bands are fixed. ``tools/reach_sweep.py`` measured 0.00% of the
  arm's joint box scoring beyond 0.65 m, while the old reward paid its
  ``in_range`` peak at 0.70 m and started the clinch penalty at 0.68 m -- the
  boxer was paid to stand where no punch exists and punished for entering the
  only band where one does. Here ``target_range`` peaks inside the scoring
  band and the clinch only starts below it.
* a hit is a force-threshold contact event on a weighted target surface, not a
  margin crossing: the published recipe's scorecard, now that the engine
  exposes force.

v0 keeps the observation vector's structure but reduces the reward to the
measured core (upright, in_range, facing, hit events, clinch, energy, ring);
the guard-hold and strike-shaping channels of the MJX env are not ported yet.
Runs only in the warp venv (mujoco 3.16 + mujoco-warp + mujoco_menagerie);
``boxer/scene.py`` is jax-free so it imports unchanged.
"""
from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene

WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_FORWARD = np.array([1.0, 0.0, 0.0])

# Observation entries that do not scale with the joint count; mirrors
# boxer/env.py's _OBS_FIXED: 3 up-axis + 3 torso velocity + 3 gyro + 6 guard
# error + 6 palm velocities, then 3 opponent torso + 3 opponent head + 6
# opponent palms + 3 opponent tilt + 3 ring centre + 2 distance/facing.
OBS_FIXED = 41

# What a landed contact is worth, by what it landed on: env.py's TARGET_WEIGHTS.
# Arms are zero so blocking stays a physical event that scores nothing.
TARGET_WEIGHTS: dict = {"head_collision": 1.0, "torso_collision": 0.5}
for _geom in scene.UPPER_TARGET_GEOMS:
    TARGET_WEIGHTS.setdefault(_geom, 0.0)
for _geom in scene.BODY_TARGET_GEOMS:
    TARGET_WEIGHTS.setdefault(_geom, 0.35)
del _geom


def quat_rotate(q, v):
    """Rotate v by quaternion q (w, x, y, z), batched on the leading axis."""
    w, u = q[..., :1], q[..., 1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_inv_rotate(q, v):
    conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_rotate(conj, v)


def _quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate([
        aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
        aw * bv + bw * av + np.cross(av, bv)], axis=-1)


def _yaw_quat(angle):
    return np.stack([np.cos(angle / 2.0), np.zeros_like(angle),
                     np.zeros_like(angle), np.sin(angle / 2.0)], axis=-1)


class WarpBoxing:
    """N independent boxing rounds stepping in parallel on one warp Data."""

    def __init__(self, nworld: int = 1024, n_frames: int = 5,
                 spawn_distance: float = 1.3, guard_pose: str = "natural",
                 action_scale: float = 0.25, spawn_jitter: float = 0.12,
                 yaw_jitter: float = 0.3, pose_noise: float = 0.02,
                 target_range: float = 0.55, punch_reach: float = 0.45,
                 clinch_range: float = 0.35, force_threshold: float = 50.0,
                 rearm_steps: int = 20, round_steps: int = 500,
                 down_up_z: float = 0.15, down_head_z: float = 0.55,
                 w_upright: float = 0.8, w_range: float = 0.3,
                 w_facing: float = 0.2, w_hit: float = 10.0,
                 w_clinch: float = 0.5, w_energy: float = 0.05,
                 w_ring: float = 1.0):
        self.nworld, self.n_frames = nworld, n_frames
        self._spawn_distance = spawn_distance
        self._action_scale = action_scale
        self._spawn_jitter, self._yaw_jitter = spawn_jitter, yaw_jitter
        self._pose_noise = pose_noise
        self._target_range, self._punch_reach = target_range, punch_reach
        self._clinch_range, self._force_threshold = clinch_range, force_threshold
        self._rearm_steps, self._round_steps = rearm_steps, round_steps
        self._down_up_z, self._down_head_z = down_up_z, down_head_z
        self._w = dict(upright=w_upright, range=w_range, facing=w_facing,
                       hit=w_hit, clinch=w_clinch, energy=w_energy, ring=w_ring)

        self._model = scene.build_model(spawn_distance=spawn_distance, visual=False)
        self._qpos0, self._ctrl0 = scene.stance(self._model, guard_pose)
        self._host = mujoco.MjData(self._model)
        self._host.qpos[:] = self._qpos0
        mujoco.mj_forward(self._model, self._host)

        self._sensor_adr, self._sensor_dim = {}, {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = int(self._model.sensor_adr[i])
            self._sensor_dim[name] = int(self._model.sensor_dim[i])

        self._sides = {}
        for side in scene.SIDES:
            actuators = [a for a in range(self._model.nu)
                         if mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                         .startswith(side)]
            joints = [int(self._model.actuator_trnid[a, 0]) for a in actuators]
            free = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT,
                                     side + "floating_base_joint")
            self._sides[side] = dict(
                actuators=np.array(actuators),
                qpos_adr=np.array([int(self._model.jnt_qposadr[j]) for j in joints]),
                dof_adr=np.array([int(self._model.jnt_dofadr[j]) for j in joints]),
                free_adr=int(self._model.jnt_qposadr[free]))
        red = self._sides[scene.RED]
        if not (len(red["actuators"]) == len(red["qpos_adr"]) == len(red["dof_adr"])):
            raise AssertionError("actuators and joints are not one-to-one")
        self._obs_size = OBS_FIXED + 3 * len(red["actuators"])

        # Per-agent constants for the observation: guard palm targets and the
        # stance joint vector, both read off the posed host model.
        self._guard, self._stance_joints = {}, {}
        for i, side in enumerate(scene.SIDES):
            frame = self._host_frame(side)
            quat = frame["quat"]
            hands = np.stack([
                quat_inv_rotate(quat, frame["left_palm"] - frame["torso"]),
                quat_inv_rotate(quat, frame["right_palm"] - frame["torso"])], axis=0)
            self._guard[i] = hands
            self._stance_joints[i] = self._qpos0[self._sides[side]["qpos_adr"]]

        self._red_ids = [g for g in range(self._model.ngeom) if (mujoco.mj_id2name(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(scene.RED)]
        self._blue_ids = [g for g in range(self._model.ngeom) if (mujoco.mj_id2name(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(scene.BLUE)]
        # Scorecard weights by target surface, indexed by geom id.
        self._weight = self._target_weights(scene.BLUE)
        self._red_weight = self._target_weights(scene.RED)

        self._m = mjwarp.put_model(self._model)
        # nconmax/njmax must be sized by hand: put_data derives defaults from
        # the host MjData's contact state, which is empty before any forward
        # pass, and the solver then silently drops constraints on overflow
        # (measured as 0 N on real contacts -- see tools/warp_scene_check.sh).
        self._d = mjwarp.put_data(self._model, self._host, nworld=nworld,
                                  nconmax=384, njmax=2048)
        try:
            self._m.opt.warn_overflow = 0
        except AttributeError:
            pass

        self._steps = np.zeros(nworld, dtype=np.int64)
        self._steps_off = np.full((nworld, 2), rearm_steps, dtype=np.int64)
        self._last_action = np.zeros((nworld, 2, self.action_size))

    def _target_weights(self, side: str) -> np.ndarray:
        out = np.zeros(self._model.ngeom)
        for name, value in TARGET_WEIGHTS.items():
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM,
                                    side + name)
            if gid >= 0:
                out[gid] = value
        return out

    @property
    def action_size(self) -> int:
        return len(self._sides[scene.RED]["actuators"])

    @property
    def obs_size(self) -> int:
        return self._obs_size

    # -- state -----------------------------------------------------------

    def _host_frame(self, side: str) -> dict:
        def read(name):
            adr = self._sensor_adr[side + name]
            return self._host.sensordata[adr:adr + self._sensor_dim[side + name]]
        return dict(torso=read("torso_pos"), head=read("head_pos"),
                    left_palm=read("left_palm_pos"), right_palm=read("right_palm_pos"),
                    torso_vel=read("torso_linvel"),
                    left_palm_vel=read("left_palm_linvel"),
                    right_palm_vel=read("right_palm_linvel"),
                    quat=read("torso_quat"), gyro=read("imu_in_torso_gyro"))

    def _frame(self, side: str, sensors: np.ndarray) -> dict:
        def read(name):
            adr = self._sensor_adr[side + name]
            return sensors[:, adr:adr + self._sensor_dim[side + name]]
        index = self._sides[side]
        return dict(torso=read("torso_pos"), head=read("head_pos"),
                    left_palm=read("left_palm_pos"), right_palm=read("right_palm_pos"),
                    torso_vel=read("torso_linvel"),
                    left_palm_vel=read("left_palm_linvel"),
                    right_palm_vel=read("right_palm_linvel"),
                    quat=read("torso_quat"), gyro=read("imu_in_torso_gyro"),
                    qpos=self._qpos_sync[:, index["qpos_adr"]],
                    qvel=self._qvel_sync[:, index["dof_adr"]])

    def reset(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        qpos = np.tile(self._qpos0.astype(np.float32), (self.nworld, 1))
        for i, side in enumerate(scene.SIDES):
            free = self._sides[side]["free_adr"]
            sign = -1.0 if i == 0 else 1.0
            lateral = rng.normal(0.0, self._spawn_jitter, self.nworld)
            forward = rng.normal(0.0, self._spawn_jitter, self.nworld)
            qpos[:, free + 0] += sign * forward
            qpos[:, free + 1] += sign * lateral
            yaw = rng.uniform(-self._yaw_jitter, self._yaw_jitter, self.nworld)
            stance = qpos[:, free + 3:free + 7].copy()
            qpos[:, free + 3:free + 7] = _quat_mul(stance, _yaw_quat(yaw))
            qpos[:, self._sides[side]["qpos_adr"]] += rng.normal(
                0.0, self._pose_noise, (self.nworld, self.action_size))
        # env.py seeds the round with small random joint velocities too.
        qvel = rng.normal(0.0, self._pose_noise * 2.5,
                          (self.nworld, self._model.nv)).astype(np.float32)
        ctrl = np.tile(np.asarray(self._ctrl0, dtype=np.float32), (self.nworld, 1))
        self._d.qpos.assign(qpos)
        self._d.qvel.assign(qvel)
        self._d.ctrl.assign(ctrl)
        mjwarp.forward(self._m, self._d)  # sensors must reflect the jittered pose
        self._qpos_sync, self._qvel_sync = qpos, qvel
        self._steps[:] = 0
        self._steps_off[:] = self._rearm_steps  # both spawn apart: start armed
        self._last_action[:] = 0.0
        return self._observe(self._d.sensordata.numpy())

    def step(self, actions: np.ndarray):
        """One control step. ``actions``: (nworld, 2, action_size) in [-1, 1]."""
        ctrl = np.tile(np.asarray(self._ctrl0, dtype=np.float32), (self.nworld, 1))
        for i, side in enumerate(scene.SIDES):
            adr = self._sides[side]["actuators"]
            ctrl[:, adr] += self._action_scale * actions[:, i, :]
        self._d.ctrl.assign(ctrl)
        for _ in range(self.n_frames):
            mjwarp.step(self._m, self._d)
        self._last_action = np.asarray(actions, dtype=np.float32)

        sensors = self._d.sensordata.numpy()
        self._qpos_sync = self._d.qpos.numpy()
        self._qvel_sync = self._d.qvel.numpy()
        hit = self._scorecard()
        reward, done, metrics = self._reward(actions, hit, sensors)
        return self._observe(sensors), reward, done, metrics

    def _scorecard(self) -> np.ndarray:
        """(nworld, 2) hit values: best target weight struck this step.

        Active contacts occupy the packed prefix ``[0, nacon)`` of the buffer;
        ``efc_address >= 0`` marks rows the solver took. Whichever corner is
        the striker, the struck surface sits in column 1 under the cross mask.
        """
        value = np.zeros((self.nworld, 2))
        nacon = int(self._d.nacon.numpy()[0])
        if not nacon:
            return value
        rows = slice(0, nacon)
        geoms = self._d.contact.geom.numpy()[rows]
        worldid = self._d.contact.worldid.numpy()[rows]
        efc = self._d.contact.efc_address.numpy()[rows, 0]
        red_first = (np.isin(geoms[:, 0], self._red_ids)
                     & np.isin(geoms[:, 1], self._blue_ids))
        blue_first = (np.isin(geoms[:, 0], self._blue_ids)
                      & np.isin(geoms[:, 1], self._red_ids))
        cross = np.flatnonzero((red_first | blue_first) & (efc >= 0))
        if not len(cross):
            return value
        force = wp.zeros(len(cross), dtype=wp.spatial_vector)
        mjwarp.contact_force(self._m, self._d, wp.array(cross.astype(np.int32)),
                             True, force)
        strong = np.linalg.norm(force.numpy()[:, :3], axis=1) >= self._force_threshold
        targets = geoms[cross, 1]
        weights = np.where(red_first[cross], self._weight[targets],
                           self._red_weight[targets])
        agents = np.where(red_first[cross], 0, 1)
        np.maximum.at(value, (worldid[cross], agents), weights * strong)
        return value

    def _reward(self, actions, hit, sensors: np.ndarray) -> tuple:
        w = self._w
        red = self._frame(scene.RED, sensors)
        blue = self._frame(scene.BLUE, sensors)
        rewards, parts = np.zeros((self.nworld, 2)), {}
        for i, (me, foe) in enumerate(((red, blue), (blue, red))):
            quat = me["quat"]
            upright = quat_rotate(quat, WORLD_UP[None, :])[:, 2]
            down = (upright < self._down_up_z) | (me["head"][:, 2] < self._down_head_z)
            alive = np.where(down, 0.0, 1.0)
            offset = foe["torso"] - me["torso"]
            distance = np.linalg.norm(offset, axis=1)
            forward = quat_rotate(quat, WORLD_FORWARD[None, :])
            facing = np.clip(
                np.sum(forward * offset / (distance[:, None] + 1e-6), axis=1), 0.0, 1.0)
            in_range = np.exp(-((distance - self._target_range) / 0.35) ** 2)
            clinch = np.clip((self._punch_reach - distance)
                             / (self._punch_reach - self._clinch_range), 0.0, 1.0)
            centre = np.linalg.norm(me["torso"][:, :2], axis=1)
            ring = np.clip(centre - 0.75 * scene.RING_RADIUS, 0.0, 10.0)
            # env.py: a hit scores only when this boxer is armed, and damage
            # taken is paced by the FOE's rearm window, not one's own.
            armed = self._steps_off[:, i] >= self._rearm_steps
            foe_armed = self._steps_off[:, 1 - i] >= self._rearm_steps
            scored = np.where(armed, hit[:, i], 0.0)
            taken = np.where(foe_armed, hit[:, 1 - i], 0.0)
            rewards[:, i] = (w["upright"] * upright
                             + w["range"] * in_range * alive * upright
                             + w["facing"] * facing * alive
                             + w["hit"] * (scored - taken)
                             - w["clinch"] * clinch * alive
                             - w["energy"] * np.mean(actions[:, i, :] ** 2, axis=1)
                             - w["ring"] * ring)
            if i == 0:
                parts = dict(upright=upright, in_range=in_range, facing=facing,
                             distance=distance, clinch=clinch, hit=hit[:, 0],
                             scored=scored, taken=taken)
        # env.py:831 -- the rearm counter resets on any scoring-surface
        # contact, armed or not, so a held clinch scores once and must let go
        # to re-arm.
        self._steps_off = np.where(hit > 0.0, 0, self._steps_off + 1)
        self._steps += 1
        done = self._steps >= self._round_steps
        return rewards, done, parts

    def _observe(self, sensors: np.ndarray) -> np.ndarray:
        red = self._frame(scene.RED, sensors)
        blue = self._frame(scene.BLUE, sensors)
        obs = np.zeros((self.nworld, 2, self.obs_size), dtype=np.float32)
        for i, (me, foe) in enumerate(((red, blue), (blue, red))):
            quat = me["quat"]
            local = lambda v: quat_inv_rotate(quat, v)
            offset = foe["torso"] - me["torso"]
            distance = np.linalg.norm(offset, axis=1)
            forward = quat_rotate(quat, WORLD_FORWARD[None, :])
            hands = np.stack([local(me["left_palm"] - me["torso"]),
                              local(me["right_palm"] - me["torso"])], axis=1)
            to_centre = np.concatenate(
                [-me["torso"][:, :2], np.zeros((self.nworld, 1))], axis=1)
            obs[:, i, :] = np.concatenate([
                quat_rotate(quat, WORLD_UP[None, :]),
                local(me["torso_vel"]),
                me["gyro"],
                (hands - self._guard[i]).reshape(self.nworld, -1),
                local(me["left_palm_vel"]),
                local(me["right_palm_vel"]),
                me["qpos"] - self._stance_joints[i],
                me["qvel"],
                self._last_action[:, i, :],
                local(offset),
                local(foe["head"] - me["torso"]),
                local(foe["left_palm"] - me["torso"]),
                local(foe["right_palm"] - me["torso"]),
                local(quat_rotate(foe["quat"], WORLD_UP[None, :])),
                local(to_centre),
                np.stack([distance, np.sum(
                    forward * offset / (distance[:, None] + 1e-6), axis=1)], axis=1),
            ], axis=1)
        return obs


if __name__ == "__main__":
    # Self-check: reset must produce fresh, finite observations; idle steps
    # must keep everything finite; a scripted jab must raise the
    # force-threshold hit channel. The jab check composes the scene at 0.45 m
    # -- the tools/warp_jab_force.py geometry, blue's arms down and red's
    # probe pose landing on the torso -- because at the training spawn of
    # 1.3 m nothing is in reach; that wall is measured, not a bug here.
    #   PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_env
    env = WarpBoxing(nworld=8, spawn_distance=0.45,
                     spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    obs = env.reset(0)
    assert obs.shape == (env.nworld, 2, env.obs_size), (
        f"obs shape {obs.shape} != (nworld, 2, obs_size={env.obs_size})")

    # Freshness: the sensors must describe the pose reset just wrote, checked
    # against a host forward of the same state.
    host = mujoco.MjData(env._model)
    host.qpos[:] = env._d.qpos.numpy()[0]
    host.qvel[:] = env._d.qvel.numpy()[0]
    host.ctrl[:] = env._ctrl0
    mujoco.mj_forward(env._model, host)
    worst = float(np.max(np.abs(host.sensordata - env._d.sensordata.numpy()[0])))
    print(f"reset: obs {obs.shape}, finite {bool(np.isfinite(obs).all())}, "
          f"distance {obs[:, 0, -2].mean():.3f} m, "
          f"sensor freshness |diff| {worst:.1e}")
    assert worst < 1e-4, "reset sensors are stale"

    zero = np.zeros((env.nworld, 2, env.action_size))
    for _ in range(3):
        obs, reward, done, parts = env.step(zero)
    print(f"idle: reward red {reward[:, 0].mean():.3f}, "
          f"distance {parts['distance'].mean():.3f} m, "
          f"in_range {parts['in_range'].mean():.3f}, "
          f"obs finite {bool(np.isfinite(obs).all())}")

    ctrl0 = np.asarray(env._ctrl0)
    red_ids = env._sides[scene.RED]["actuators"]
    blue_ids = env._sides[scene.BLUE]["actuators"]

    def column(ids, name):
        aid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return int(np.flatnonzero(ids == aid)[0])

    def amp(ids, name, ctrl_value):
        col = column(ids, name)
        return (ctrl_value - ctrl0[ids[col]]) / env._action_scale

    # The reach_sweep probe pose: shoulder pitch -1.5, elbow 0.0 puts a fist
    # on the torso at this distance; blue's arms go to their zero pose so no
    # forearm blocks the line.
    jab_pitch = amp(red_ids, "red/left_shoulder_pitch_joint", -1.5)
    jab_elbow = amp(red_ids, "red/left_elbow_joint", 0.0)
    action = np.zeros((env.nworld, 2, env.action_size))
    for half in ("left", "right"):
        for joint in scene.ARM_JOINTS:
            action[:, 1, column(blue_ids, f"blue/{half}_{joint}_joint")] = \
                amp(blue_ids, f"blue/{half}_{joint}_joint", 0.0)

    hits = 0.0
    for step in range(110):
        punch = max(0.0, np.sin(2.0 * np.pi * step / 30.0)) ** 2
        action[:, 0, column(red_ids, "red/left_shoulder_pitch_joint")] = jab_pitch * punch
        action[:, 0, column(red_ids, "red/left_elbow_joint")] = jab_elbow * punch
        obs, reward, done, parts = env.step(action)
        hits = max(hits, float(parts["hit"].max()))
    print(f"scripted jab over 110 steps: peak hit value {hits:.2f} "
          f"(force threshold {env._force_threshold:.0f} N)")
    assert np.isfinite(obs).all() and hits > 0.0, "self-check failed"
    print("self-check passed")
