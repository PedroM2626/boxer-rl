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
  exposes force. It is credited by ROLE -- a striker surface of one corner on a
  scoring surface of the other -- and gated on that fist closing on the
  opponent, because geometry order in the contact buffer says nothing about who
  threw the punch and a leaning guard clears 50 N on its own.
* the interaction terms (range, facing, clinch, imitation) need BOTH boxers
  standing. warp1 ended its rounds sitting on the floor propped on one arm,
  because the old down test (upright_z < 0.15, head below 0.55 m) called that
  alive: falling cost nothing and staying down collected ``in_range`` and facing
  all round. A torso past 45 deg or a head under 0.85 m is now down, and those
  terms stop for both corners the moment one of them is. ``upright`` itself stays
  ungated -- the first warp2 attempt gated it too, and the rollout then spent 87%
  of the round on the floor paying nobody anything, so the gradient died with the
  reward. See the comment in ``_reward``.
* ``in_range`` is narrow (``range_sigma`` 0.15, was brax's 0.35) and cheap
  (``w_range`` 0.15, was 0.3). At 0.86 m the old width still paid 55% of the peak
  -- 0.156/step of free income against a punch worth ~5 per event that happens a
  fraction of the time -- which is why warp1 froze at a distance where nothing
  can land.
* optionally, an imitation shaping term (``imitation_path``): the strike-phase
  arm poses of the retargeted jab (``tools/retarget_dense.py``; the JSON sits
  out of git with the clip), paid clock-free as the closest pose match and
  gated on forward fist speed so holding a pose pays nothing. Its tolerance is
  0.6 rad: warp1's 0.3 rad was unreachable from a random arm and the term
  measured exactly 0.0 through all 611 iterations.

* the fight has a state of its own: clean hits accumulate as damage (the weight of
  what landed -- head 1.0, torso 0.5, body 0.35), a hit also stuns its victim out of
  swinging for ``stun_steps``, and a corner at ``damage_cap`` is out, which closes the
  round and pays the survivor the zero-sum knockout. warp6's arithmetic is why: a swing
  that lands 37% of the time returns 10 x 0.49 x 0.37 = +1.81 against a 0.5 charge, so
  hitting is always profitable, and a punch that lands changed the score but not the
  fight. Damage is the state that makes an early punch compound; the stun is what makes
  being hit cost more than the points it gives away. Both corners' damage and the
  victim's stun are in the observation -- three entries, so ``OBS_FIXED`` is 44 and the
  vector is 131 wide -- because a state the policy cannot see is a state it cannot pace.

The reward carries the measured core (upright, alive, knockdown, in_range,
facing, hit events, clinch, energy, ring) and truncates per world the way
``boxer/env.py`` does: a round closes as soon as one boxer is down, out of the
ring, or further than ``MAX_SEPARATION`` from the other, and ``step`` respawns
those worlds and returns their next round's first observation. The knockdown is
zero-sum and therefore one-shot -- paid per step across the 320 floor steps an
untruncated round spends, it would reward a mutual collapse instead of a
knockout. The guard-hold and strike-shaping channels of the MJX env are still
not ported. Runs only in the warp venv (mujoco 3.16 + mujoco-warp +
mujoco_menagerie); ``boxer/scene.py`` is jax-free so it imports unchanged.
"""
from __future__ import annotations

import json

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene

WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_FORWARD = np.array([1.0, 0.0, 0.0])
# boxer/env.py:32. Past this the two are not in a fight anymore and the round is
# over whether or not anyone is on the floor.
MAX_SEPARATION = 2.4

# Observation entries that do not scale with the joint count; mirrors
# boxer/env.py's _OBS_FIXED: 3 up-axis + 3 torso velocity + 3 gyro + 6 guard
# error + 6 palm velocities, then 3 opponent torso + 3 opponent head + 6
# opponent palms + 3 opponent tilt + 3 ring centre + 2 distance/facing, plus the
# 3 fight-state entries this env adds: own damage, opponent damage, stunned.
OBS_FIXED = 44
# Where the distance entry sits, named because the self-check reads it out of the
# observation to prove a respawn really re-seeded the stance. The observation ends
# with ... ring centre (3), distance/facing (2), fight state (3), so distance is
# five from the end -- and a bare -2 silently stopped pointing at it the day the
# fight-state block went on, which is how this got caught.
OBS_DISTANCE = -5

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
                 spawn_distance: float = 0.75, guard_pose: str = "natural",
                 action_scale: float = 0.25, spawn_jitter: float = 0.12,
                 yaw_jitter: float = 0.3, pose_noise: float = 0.02,
                 target_range: float = 0.55, punch_reach: float = 0.45,
                 range_sigma: float = 0.15,
                 clinch_range: float = 0.35, force_threshold: float = 50.0,
                 rearm_steps: int = 20, round_steps: int = 500,
                 down_up_z: float = 0.70, down_head_z: float = 0.85,
                 punch_speed_min: float = 1.0,
                 damage_cap: float = 3.0, stun_steps: int = 10,
                 w_upright: float = 0.8, w_range: float = 0.15,
                 w_facing: float = 0.2, w_hit: float = 10.0,
                 w_clinch: float = 0.5, w_energy: float = 0.05,
                 w_ring: float = 1.0, w_alive: float = 0.2, w_ko: float = 3.0,
                 w_attempt: float = 0.5, retract_radius: float = 0.25,
                 imitation_path: str | None = None, w_imit: float = 0.3,
                 imit_sigma: float = 0.6, imit_speed_ref: float = 5.0):
        self.nworld, self.n_frames = nworld, n_frames
        self._spawn_distance = spawn_distance
        self._guard_pose = guard_pose
        self._action_scale = action_scale
        self._spawn_jitter, self._yaw_jitter = spawn_jitter, yaw_jitter
        self._pose_noise = pose_noise
        self._target_range, self._punch_reach = target_range, punch_reach
        self._range_sigma = range_sigma
        self._clinch_range, self._force_threshold = clinch_range, force_threshold
        self._rearm_steps, self._round_steps = rearm_steps, round_steps
        # A torso tilted past 45 deg or a head below 0.85 m is down. The old
        # thresholds (0.15 / 0.55) only caught a boxer flat on the floor, and
        # warp1's video showed why that is not a detail: both corners fell in
        # ~1 s and then sat propped on their arms collecting the dense terms.
        self._down_up_z, self._down_head_z = down_up_z, down_head_z
        self._punch_speed_min = punch_speed_min
        # Damage is the weight of every clean hit absorbed (head 1.0, torso 0.5,
        # body 0.35). The cap has to sit BELOW what a good round can deliver or the
        # knockout never fires: warp6's final rounds land 7.1 torso contacts, which
        # is 3.55 damage, so a cap of 5.0 -- the first number I wrote here -- made
        # the whole mechanism dormant and only the stun would have been in effect.
        # 3.0 is six torso shots or three to the head, reachable inside a long round.
        # The reason it exists at all is warp6's arithmetic: a swing that lands 37%
        # of the time returns +1.81 against a 0.5 charge, so hitting is always
        # profitable and the hit channel is symmetric and instantaneous -- a punch
        # changes the score, never the fight. Accumulated damage is the state that
        # makes an early punch compound into a later advantage, and the stun window
        # is what makes being hit cost more than the points it gives away.
        self._damage_cap, self._stun_steps = damage_cap, stun_steps
        self._imit_sigma, self._imit_speed_ref = imit_sigma, imit_speed_ref
        self._w = dict(upright=w_upright, range=w_range, facing=w_facing,
                       hit=w_hit, clinch=w_clinch, energy=w_energy, ring=w_ring,
                       alive=w_alive, ko=w_ko, attempt=w_attempt, imit=w_imit)
        # A fist this close to its guard spot counts as home, which is what
        # re-arms it (see _reward). Measured against the palm's own torso-frame
        # target, not against the torso site at body centre: env.py's retraction
        # gate used the latter and a fist resting on the opponent read as 0.16 m
        # retracted, which is why that gate was decoration.
        self._retract_radius = retract_radius

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
        # Scorecard value of being hit, indexed by target geom (both corners).
        self._target_w = np.zeros(self._model.ngeom)
        for side in scene.SIDES:
            for name, value in TARGET_WEIGHTS.items():
                gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM,
                                        side + name)
                if gid >= 0:
                    self._target_w[gid] = value
        # Who struck whom is a question about ROLES, not about the order the
        # collision pair happens to be stored in: tools/warp_hit_directions.py
        # measured a chest-to-chest press scoring 0.5 for red, and blue landing
        # a whole jab scoring 0.0, because red geoms keep the lower index and
        # the pair came out (red, blue) ~90% of the time. Only a fist or
        # forearm of one corner meeting a scoring surface of the other counts.
        self._striker = np.zeros(self._model.ngeom, dtype=bool)
        self._striker_palm = -np.ones(self._model.ngeom, dtype=np.int64)
        self._geom_side = -np.ones(self._model.ngeom, dtype=np.int64)
        for agent, side in enumerate(scene.SIDES):
            for geom in scene.HAND_GEOMS + scene.FOREARM_GEOMS:
                gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM,
                                        side + geom)
                if gid >= 0:
                    self._striker[gid] = True
                    # which of the two fists this surface belongs to, so the
                    # motion gate can read that palm's velocity
                    self._striker_palm[gid] = 1 if geom.startswith("right") else 0
            for g in (self._red_ids if side == scene.RED else self._blue_ids):
                self._geom_side[g] = agent

        # Optional imitation shaping: the strike-phase poses of the retargeted
        # jab (tools/retarget_dense.py), four right-arm joints per clip frame.
        self._imit = None
        self._imit_cols = None
        self._imit_arm = "right"
        self._imit_fps = 30.0
        if imitation_path is not None:
            with open(imitation_path) as handle:
                traj = json.load(handle)
            start, stop = traj["strike_window"]
            self._imit = np.asarray(traj["targets_radians"])[start:stop + 1]
            # the recorded arm decides which of the boxer's own arms is shaped
            self._imit_arm = traj.get("side", "right")
            self._imit_fps = float(traj.get("fps", 30))
            self._imit_cols = {side: self._arm_columns(side, self._imit_arm)
                               for side in scene.SIDES}

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
        # Per world and corner: has the fist come home since it last touched the
        # opponent, and was the last step already a committed swing (the attempt
        # is a rising edge, so one jab is one attempt and a vibrating fist is many).
        self._came_back = np.ones((nworld, 2), dtype=bool)
        self._was_committed = np.zeros((nworld, 2), dtype=bool)
        # Damage absorbed and steps of stun left, per world and corner.
        self._damage = np.zeros((nworld, 2), dtype=np.float32)
        self._stun = np.zeros((nworld, 2), dtype=np.int32)
        self._last_action = np.zeros((nworld, 2, self.action_size))
        # reset(seed) reseeds it; step() draws the per-world respawns from it.
        self._rng = np.random.default_rng(0)

    def _arm_columns(self, side: str, arm: str = "right") -> np.ndarray:
        """Columns of the recorded arm's four fitted joints inside me["qpos"]."""
        cols = []
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    side + arm + "_" + joint + "_joint")
            cols.append(int(np.flatnonzero(self._sides[side]["actuators"] == aid)[0]))
        return np.array(cols)

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
        self._rng = np.random.default_rng(seed)
        self._qpos_sync = np.tile(self._qpos0.astype(np.float32), (self.nworld, 1))
        self._qvel_sync = np.zeros((self.nworld, self._model.nv), np.float32)
        return self._randomize(self._rng, np.ones(self.nworld, dtype=bool))

    def _randomize(self, rng, rows: np.ndarray) -> np.ndarray:
        """Start a round in the selected worlds and return their fresh obs.

        Rows, not all: env.py closes a round the moment a boxer is down, out of
        the ring or out of reach of the other, and the worlds that are still
        fighting have to keep fighting. Writing the spawn into a mask is what
        makes that per-world instead of a global clock.
        """
        index = np.flatnonzero(rows)
        if not len(index):
            return np.zeros((0, 2, self.obs_size), dtype=np.float32)
        sub = np.tile(self._qpos0.astype(np.float32), (len(index), 1))
        for i, side in enumerate(scene.SIDES):
            free = self._sides[side]["free_adr"]
            sign = -1.0 if i == 0 else 1.0
            lateral = rng.normal(0.0, self._spawn_jitter, len(index))
            forward = rng.normal(0.0, self._spawn_jitter, len(index))
            sub[:, free + 0] += sign * forward
            sub[:, free + 1] += sign * lateral
            yaw = rng.uniform(-self._yaw_jitter, self._yaw_jitter, len(index))
            stance = sub[:, free + 3:free + 7].copy()
            sub[:, free + 3:free + 7] = _quat_mul(stance, _yaw_quat(yaw))
            sub[:, self._sides[side]["qpos_adr"]] += rng.normal(
                0.0, self._pose_noise, (len(index), self.action_size))
        self._qpos_sync[rows] = sub
        # env.py seeds the round with small random joint velocities too.
        self._qvel_sync[rows] = rng.normal(0.0, self._pose_noise * 2.5,
                                           (len(index), self._model.nv)).astype(np.float32)
        ctrl = np.tile(np.asarray(self._ctrl0, dtype=np.float32), (self.nworld, 1))
        self._d.qpos.assign(self._qpos_sync)
        self._d.qvel.assign(self._qvel_sync)
        self._d.ctrl.assign(ctrl)
        mjwarp.forward(self._m, self._d)  # sensors must reflect the jittered pose
        self._steps[rows] = 0
        self._steps_off[rows] = self._rearm_steps  # both spawn apart: start armed
        self._came_back[rows] = True   # they spawn in the guard, so the fist is home
        self._was_committed[rows] = False
        self._damage[rows] = 0.0
        self._stun[rows] = 0
        self._last_action[rows] = 0.0
        return self._observe(self._d.sensordata.numpy())[rows]

    def step(self, actions: np.ndarray):
        """One control step. ``actions``: (nworld, 2, action_size) in [-1, 1]."""
        ctrl = np.tile(np.asarray(self._ctrl0, dtype=np.float32), (self.nworld, 1))
        for i, side in enumerate(scene.SIDES):
            adr = self._sides[side]["actuators"]
            ctrl[:, adr] += self._action_scale * actions[:, i, :]
        self._d.ctrl.assign(ctrl)
        hit = self._advance()
        # np.array, not np.asarray: a jax-derived input may arrive read-only,
        # and reset zeroes this buffer in place.
        self._last_action = np.array(actions, dtype=np.float32)

        sensors = self._d.sensordata.numpy()
        self._qpos_sync = self._d.qpos.numpy()
        self._qvel_sync = self._d.qvel.numpy()
        if hit is None:
            hit = self._scorecard(sensors)
        reward, done, metrics = self._reward(actions, hit, sensors)
        obs = self._observe(sensors)
        if getattr(self, "capture_render_state", False):
            self.render_qpos = self._qpos_sync.copy()
            self.render_qvel = self._qvel_sync.copy()
        if bool(done.any()):
            # Autoreset, brax/PGAX semantics: the observation returned for a world
            # that just closed its round is the FIRST observation of the next one,
            # and done=1 is what tells the learner not to bootstrap across the
            # seam. Rounds now end per world, so the batch is no longer on a
            # shared clock.
            obs[done] = self._randomize(self._rng, done)
        return obs, reward, done, metrics

    def _advance(self):
        """Advance physics; subclasses may collect impacts at substep resolution."""
        for _ in range(self.n_frames):
            mjwarp.step(self._m, self._d)
        return None

    def _scorecard(self, sensors: np.ndarray,
                   before: np.ndarray | None = None) -> np.ndarray:
        """(nworld, 2) hit values: best target weight struck this step.

        Active contacts occupy the packed prefix ``[0, nacon)`` of the buffer;
        ``efc_address >= 0`` marks rows the solver took. Attribution is by
        role: the geom carrying a striker surface (fist/forearm) is the hitter
        and the other geom is the target, whichever column each landed in. A
        scored contact must also be moving: a guard forearm leaning on a chest
        crosses the force threshold under gravity alone
        (tools/warp_hit_directions.py), and env.py paid hits only under a
        commitment gate, so the striker's fist has to travel toward the
        opponent at ``punch_speed_min`` or better.
        """
        value = np.zeros((self.nworld, 2))
        nacon = int(self._d.nacon.numpy()[0])
        if not nacon:
            return value
        rows = slice(0, nacon)
        geoms = self._d.contact.geom.numpy()[rows]
        worldid = self._d.contact.worldid.numpy()[rows]
        efc = self._d.contact.efc_address.numpy()[rows, 0]
        striker_first = self._striker[geoms[:, 0]] & ~self._striker[geoms[:, 1]]
        striker_second = self._striker[geoms[:, 1]] & ~self._striker[geoms[:, 0]]
        cand = np.flatnonzero((striker_first | striker_second) & (efc >= 0))
        if not len(cand):
            return value
        first = striker_first[cand]
        strikers = np.where(first, geoms[cand, 0], geoms[cand, 1])
        targets = np.where(first, geoms[cand, 1], geoms[cand, 0])
        agents = self._geom_side[strikers]
        # a fist against its own corner's ribs is not a punch
        cross = self._geom_side[targets] != agents
        valid = np.flatnonzero(cross & (self._target_w[targets] > 0.0))
        if not len(valid):
            return value
        sel = cand[valid]
        force = wp.zeros(len(sel), dtype=wp.spatial_vector)
        mjwarp.contact_force(self._m, self._d, wp.array(sel.astype(np.int32)),
                             True, force)
        strong = np.linalg.norm(force.numpy()[:, :3], axis=1) >= self._force_threshold

        red = self._frame(scene.RED, sensors)
        blue = self._frame(scene.BLUE, sensors)
        palm = np.stack([[red["left_palm_vel"], red["right_palm_vel"]],
                         [blue["left_palm_vel"], blue["right_palm_vel"]]])
        torso = np.stack([red["torso"], blue["torso"]])
        who, where = agents[valid], worldid[sel]
        direction = torso[1 - who, where] - torso[who, where]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
        toward = np.sum(palm[who, self._striker_palm[strikers[valid]], where]
                        * direction, axis=1)
        if before is not None:
            # A real impact can stop the fist inside one physics step. Testing
            # only its post-impact speed rejects exactly those clean punches.
            previous = [self._frame(side, before) for side in scene.SIDES]
            velocity = np.stack([[f["left_palm_vel"], f["right_palm_vel"]]
                                 for f in previous])
            approach = np.sum(
                velocity[who, self._striker_palm[strikers[valid]], where] * direction,
                axis=1)
            toward = np.maximum(toward, approach)
        scored = strong & (toward >= self._punch_speed_min)
        np.maximum.at(value, (where, who),
                      self._target_w[targets[valid]] * scored)
        return value

    def _reward(self, actions, hit, sensors: np.ndarray) -> tuple:
        w = self._w
        red = self._frame(scene.RED, sensors)
        blue = self._frame(scene.BLUE, sensors)
        # Both corners' down state, needed before the per-agent loop so a boxer's
        # metrics can see the opponent's state (the video of warp1 iteration 375
        # shows the standing corner carrying the fallen one around, because the
        # range term only checked whether *I* was down).
        frames = (red, blue)
        downs = [(quat_rotate(f["quat"], WORLD_UP[None, :])[:, 2] < self._down_up_z)
                 | (f["head"][:, 2] < self._down_head_z) for f in frames]
        # env.py:805-809 -- the round also closes when a boxer is out of the ring
        # or the two are further apart than a fight.
        outside = [np.linalg.norm(f["torso"][:, :2], axis=1) > scene.RING_RADIUS
                   for f in frames]
        separation = np.linalg.norm(red["torso"] - blue["torso"], axis=1)
        rewards, parts = np.zeros((self.nworld, 2)), {}
        # Punch bookkeeping for BOTH corners, before the reward loop: damage taken
        # is paced by the opponent's window, so a corner's armed state has to exist
        # before either corner's reward is written.
        armed = np.zeros((self.nworld, 2), dtype=bool)
        attempts = np.zeros((self.nworld, 2))
        at_home = np.zeros((self.nworld, 2))
        for i, (me, foe) in enumerate(((red, blue), (blue, red))):
            quat = me["quat"]
            offset = foe["torso"] - me["torso"]
            distance = np.linalg.norm(offset, axis=1)
            direction = offset / (distance[:, None] + 1e-6)
            # The fist closing on the opponent, contact buffer or not. This is the
            # attempt -- the denominator RoboStriker's hit rate carries and a raw
            # event count does not, which is why an arm mill is free to score here
            # and expensive in the paper. Counted on the rising edge, so one jab is
            # one attempt and a fist vibrating across the threshold is many.
            closing = np.maximum(np.sum(me["left_palm_vel"] * direction, axis=1),
                                 np.sum(me["right_palm_vel"] * direction, axis=1))
            committed = closing >= self._punch_speed_min
            attempts[:, i] = committed & ~self._was_committed[:, i]
            # Hysteresis: a swing is one attempt, not one per step of its vibration.
            # Without the release level the closing speed flickers across the
            # threshold and the scripted jab measured 51 attempts in ~4 swings,
            # which would charge a real punch like a mill.
            still = closing >= 0.5 * self._punch_speed_min
            self._was_committed[:, i] = still & (committed | self._was_committed[:, i])
            # Has the fist come back to its own guard spot since it last touched the
            # opponent. Without this the rearm window only asks for 20 steps of no
            # contact, so a fist that flickers off and back onto a chest scores
            # again without ever returning -- which is the mill, unscored as such.
            palms = np.stack([quat_inv_rotate(quat, me["left_palm"] - me["torso"]),
                              quat_inv_rotate(quat, me["right_palm"] - me["torso"])],
                             axis=1)
            at_home[:, i] = np.linalg.norm(
                palms - self._guard[i][None, :, :], axis=2).max(axis=1)
            home = at_home[:, i] <= self._retract_radius
            touched = hit[:, i] > 0.0
            # Order matters: armed reads the flag as of the PREVIOUS step, and only
            # then does this step's contact clear it. Computed the other way round,
            # the step that touches marks the fist away from guard before asking
            # whether it may score, and no hit can ever be armed (measured: raw
            # contact 0.50, scored 0.00, with the gate open).
            armed[:, i] = ((self._steps_off[:, i] >= self._rearm_steps)
                           & self._came_back[:, i]
                           & (self._stun[:, i] == 0))
            self._came_back[:, i] = np.where(touched, False,
                                             np.where(home, True, self._came_back[:, i]))
        scored_by = np.where(armed, hit, 0.0)  # what each corner landed, armed or not
        # What each corner ABSORBED this step, from the other corner's armed punch.
        # Damage accumulates and a clean hit also stuns the victim out of its next few
        # swings -- the part that makes trading blows a loss instead of a wash. The
        # stun is read before it is written, for the same ordering reason as the
        # re-arm flag: the step that lands a hit must not be the step that loses the
        # ability to land it.
        absorbed = scored_by[:, ::-1].copy()
        self._damage += absorbed
        self._stun = np.where(absorbed > 0.0, self._stun_steps,
                              np.maximum(self._stun - 1, 0)).astype(np.int32)
        # Out on the feet or out on damage ends the fight the same way; `downs` stays
        # the physical floor test so the posture metric keeps its meaning.
        outs = [downs[i] | (self._damage[:, i] >= self._damage_cap) for i in (0, 1)]
        for i, (me, foe) in enumerate(((red, blue), (blue, red))):
            quat = me["quat"]
            upright = quat_rotate(quat, WORLD_UP[None, :])[:, 2]
            down = downs[i]
            out = outs[i]  # down, or out on accumulated damage
            # Two gates, two jobs. ``engaged`` (BOTH corners up) holds the
            # interaction terms: warp1 ended the round propped on one arm next to
            # a fallen foe because a per-agent gate still paid range and facing
            # there, so falling was free.
            #
            # ``upright`` itself is NOT gated -- it is about my own posture. The
            # first warp2 attempt gated it and paid for the mistake: measured with
            # tools/warp_probe_policy.py, the boxers spent 87% of the round on the
            # floor and both corners were up for only 11.8% of it, so for most of
            # every rollout all dense terms read zero. Reward flatlined at -0.005
            # (the energy residue), value_loss collapsed to 0.0003, entropy fell
            # 47 -> 8 and std pinned at its 0.3 floor -- a dead gradient, not a
            # converged policy. Standing had to pay again before falling could
            # lose.
            engaged = np.where(out | outs[1 - i], 0.0, 1.0)
            offset = foe["torso"] - me["torso"]
            distance = np.linalg.norm(offset, axis=1)
            forward = quat_rotate(quat, WORLD_FORWARD[None, :])
            facing = np.clip(
                np.sum(forward * offset / (distance[:, None] + 1e-6), axis=1), 0.0, 1.0)
            in_range = np.exp(-((distance - self._target_range) / self._range_sigma) ** 2)
            clinch = np.clip((self._punch_reach - distance)
                             / (self._punch_reach - self._clinch_range), 0.0, 1.0)
            centre = np.linalg.norm(me["torso"][:, :2], axis=1)
            ring = np.clip(centre - 0.75 * scene.RING_RADIUS, 0.0, 10.0)
            # env.py: a hit scores only when this boxer is armed, and damage
            # taken is paced by the FOE's rearm window, not one's own.
            scored = np.where(armed[:, i], hit[:, i], 0.0)
            taken = np.where(armed[:, 1 - i], hit[:, 1 - i], 0.0)
            # env.py:160/672 and 837-842 -- the two channels the v0 port dropped,
            # and the reason three warp arms treated a round as free time on the
            # floor. Being up pays on its own; the knockdown is zero-sum, so the
            # corner left standing collects w_ko and the one lying under it pays
            # the same. It is a ONE-SHOT payment only because the round closes on
            # this step: paid per step across 320 floor steps it would reward a
            # mutual collapse (both down cancels to zero) instead of a knockout,
            # which is the leak env.py:161-165 already recorded for this term.
            alive = np.where(out, 0.0, 1.0)
            ko = (alive * np.where(outs[1 - i], 1.0, 0.0)
                  - np.where(out, 1.0, 0.0) * np.where(outs[1 - i], 0.0, 1.0))
            rewards[:, i] = (w["upright"] * upright
                             + w["alive"] * alive
                             + w["ko"] * ko
                             + w["range"] * in_range * engaged
                             + w["facing"] * facing * engaged
                             + w["hit"] * (scored - taken)
                             - w["attempt"] * attempts[:, i]
                             - w["clinch"] * clinch * engaged
                             - w["energy"] * np.mean(actions[:, i, :] ** 2, axis=1)
                             - w["ring"] * ring)
            imit = np.zeros(self.nworld)
            if self._imit is not None:
                # Shape reward: how close this boxer's recorded arm sits to ANY
                # strike-phase pose of the retargeted jab -- no clock, the
                # boxer chooses when to pass through them. Gated by the fist
                # moving along the line to the opponent, so a static guard-
                # like pose near a target pays nothing (the old shaping term's
                # failure mode), and by both corners being up.
                cols = self._imit_cols[scene.RED if i == 0 else scene.BLUE]
                qarm = me["qpos"][:, cols]
                d2 = np.sum((qarm[:, None, :] - self._imit[None, :, :]) ** 2, axis=2)
                shape = np.exp(-d2 / self._imit_sigma ** 2).max(axis=1)
                speed = np.sum(me[self._imit_arm + "_palm_vel"] * offset
                               / (distance[:, None] + 1e-6), axis=1)
                gate = np.clip(speed / self._imit_speed_ref, 0.0, 1.0)
                imit = shape * gate * engaged
                rewards[:, i] += w["imit"] * imit
            if i == 0:
                parts = dict(upright=upright, in_range=in_range, facing=facing,
                             distance=distance, clinch=clinch, hit=hit[:, 0],
                             scored=scored, taken=taken, imit=imit,
                             # metric only: both corners' down flags, because the
                             # training log's reward and distance cannot tell
                             # "walked away" from "fell over and got gated off"
                             on_floor=np.stack(downs, axis=1).astype(np.float32),
                             alive=alive, ko=ko,
                             # both corners, because the punch channel is the one
                             # question the log cannot ask of one side alone: the
                             # hit value carries the target's weight (1.0 head, 0.5
                             # torso, 0.35 pelvis/hip/thigh), so the surface split,
                             # the attempts that produced it and the fist's distance
                             # from its own guard all come out per corner.
                             scored_by=scored_by, attempts=attempts,
                             at_home=at_home,
                             # metric only, no reward attached yet: how often red is
                             # inside punching distance of a DOWN blue -- the state
                             # the range term pays without any punch being thrown
                             carry=np.where(downs[1], (distance < self._punch_reach)
                                            & (in_range > 0.5), 0.0))
        # env.py:831 -- the rearm counter resets on any scoring-surface
        # contact, armed or not, so a held clinch scores once and must let go
        # to re-arm.
        self._steps_off = np.where(hit > 0.0, 0, self._steps_off + 1)
        self._steps += 1
        # env.py:809 -- a round closes for BOTH corners the moment one of them is
        # down, out of the ring, or further than MAX_SEPARATION from the other.
        # Without this the 480-step round keeps simulating a finished fight: warp3
        # measured 320 of its steps with both boxers on the floor, and a knockout
        # paid per step over those would reward a mutual collapse (both down
        # cancels the zero-sum term), so the channel needs the truncation to mean
        # what it means in env.py.
        timed_out = self._steps >= self._round_steps
        # Three different reasons, kept apart on purpose: a single flag that ORs
        # them cannot tell a fight ended on the floor from one that ended because
        # the corners wandered 2.4 m apart, and the whole point of truncating is to
        # make falling cost something -- so which one is closing the rounds is the
        # number that matters.
        fell = downs[0] | downs[1]
        # A fight can now also end the way a fight ends: one corner has absorbed
        # enough clean punches to be out, whoever its torso is doing.
        damage = self._damage >= self._damage_cap
        judged_out = damage[:, 0] | damage[:, 1]
        left_ring = outside[0] | outside[1]
        too_far = separation > MAX_SEPARATION
        knocked = fell | judged_out | left_ring | too_far
        done = timed_out | knocked
        parts.update(knockout=knocked, fell=fell, judged_out=judged_out,
                     left_ring=left_ring, too_far=too_far, timed_out=timed_out,
                     separation=separation,
                     # A copy, and it has to be one: ``step`` respawns the worlds
                     # that closed a round AFTER this dict is built, and `_randomize`
                     # zeroes the live ``self._damage`` in place. A reference would
                     # hand the caller the next round's state -- every round that
                     # ended on the damage cap reading zero damage, which is how the
                     # first arena matchup reported 0.000 knockout wins while
                     # ``end_ko`` said 63% of rounds ended on the cap.
                     damage=self._damage.copy(),
                     # per corner, for anything that has to decide who lost: `fell`
                     # and `judged_out` above are either-or, and an arena needs to
                     # know WHICH corner is out on the step the round ends.
                     out_by=np.stack(outs, axis=1).astype(np.float32),
                     stun=(self._stun > 0).astype(np.float32))
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
                # The state of the fight, which is the only way an early punch can
                # matter later: damage each corner has absorbed (normalised by the
                # cap, so 1.0 means "out"), and whether this corner is still stunned
                # out of swinging. Without these the policy cannot pace -- it can
                # only watch the score change, which is what warp6 did.
                np.stack([self._damage[:, i] / self._damage_cap,
                          self._damage[:, 1 - i] / self._damage_cap,
                          (self._stun[:, i] > 0).astype(np.float32)], axis=1),
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
          f"distance {obs[:, 0, OBS_DISTANCE].mean():.3f} m, "
          f"sensor freshness |diff| {worst:.1e}")
    assert worst < 1e-4, "reset sensors are stale"

    zero = np.zeros((env.nworld, 2, env.action_size))
    for _ in range(3):
        obs, reward, done, parts = env.step(zero)
    assert parts["on_floor"].shape == (env.nworld, 2), parts["on_floor"].shape
    print(f"idle: reward red {reward[:, 0].mean():.3f}, "
          f"distance {parts['distance'].mean():.3f} m, "
          f"in_range {parts['in_range'].mean():.3f}, "
          f"on_floor {parts['on_floor'].mean():.3f}, "
          f"obs finite {bool(np.isfinite(obs).all())}")

    # The two channels env.py has and this port dropped, checked by construction.
    # Roll blue's torso 90 deg about its own forward axis in the sensor buffer: the
    # step must read blue down, credit red the knockout, and close the round.
    def tilt(side):
        adr = env._sensor_adr[side + "torso_quat"]
        dim = env._sensor_dim[side + "torso_quat"]
        sensors = env._d.sensordata.numpy().copy()
        roll90 = np.tile(np.array([0.7071068, 0.7071068, 0.0, 0.0], np.float32),
                         (env.nworld, 1))  # 90 deg about the torso's own x axis
        sensors[:, adr:adr + dim] = _quat_mul(sensors[:, adr:adr + dim], roll90)
        return env._reward(zero, np.zeros((env.nworld, 2), np.float32), sensors)

    for side, who in ((scene.BLUE, 1), (scene.RED, 0)):
        reward_t, done_t, parts_t = tilt(side)
        assert parts_t["on_floor"][:, who].all(), f"{side} tilt not read as down"
        assert bool(done_t.all()), f"{side} knockdown did not close the round"
        ko = float(parts_t["ko"].mean())  # parts carries RED's side of the zero-sum
        expected = -1.0 if who == 0 else 1.0
        assert abs(ko - expected) < 1e-6, f"ko sign {ko:+.2f}, expected {expected:+.0f}"
        print(f"{side} torso rolled 90 deg: read down, knockout to red {ko:+.0f}, "
              f"round closed in {int(done_t.sum())}/{env.nworld} worlds")

    # Autoreset: force every world to terminate on this step and check that the
    # observation handed back is the NEXT round's first one, not the corpse of this.
    env._down_head_z = 5.0
    obs_r, reward_r, done_r, parts_r = env.step(zero)
    env._down_head_z = 0.85
    assert done_r.all() and not env._steps.any(), "the respawn did not reset the clock"
    fresh = float(obs_r[:, 0, OBS_DISTANCE].mean())
    print(f"autoreset: {int(done_r.sum())} worlds respawned inside step(), "
          f"distance back to {fresh:.3f} m (spawn {env._spawn_distance:.2f})")
    assert abs(fresh - env._spawn_distance) < 0.15, "respawn did not re-seed the stance"

    # Damage, stun and the knockout are the fight's own state, so they are tested
    # against the reward function rather than through physics: five clean hits of
    # head weight on blue must stun it and then close the round as a knockout.
    env.reset(0)
    sensors_now = env._d.sensordata.numpy()
    ko_done, ko_parts = None, None
    for _ in range(int(round(env._damage_cap))):
        # Re-arm both corners by hand: the first hit spends red's rearm window and
        # marks its fist away from guard, and this test is about damage, stun and
        # the knockout, not about the window those are separate assertions for.
        env._steps_off[:] = env._rearm_steps
        env._came_back[:] = True
        landed = np.zeros((env.nworld, 2), np.float32)
        landed[:, 0] = 1.0                       # red lands a clean head-value hit
        _r, ko_done, ko_parts = env._reward(zero, landed, sensors_now)
    assert np.allclose(env._damage[:, 1], env._damage_cap), env._damage[:, 1]
    assert bool(ko_parts["judged_out"].all()), "clean hits did not end the fight"
    assert bool(ko_done.all()), "a knockout did not close the round"
    assert ko_parts["stun"][:, 1].all(), "the victim was not stunned by a clean hit"
    # The damage the caller is told about must survive the respawn that ``step``
    # performs after this dict is built. The health test calls `_reward` directly,
    # so only the identity check pins it here -- the arena, which reads the dict
    # through `step`, is what found the violation: with a view of the live state,
    # every round that ended on the damage cap reported 0.00 damage and 0.000
    # knockout wins.
    assert ko_parts["damage"] is not env._damage, (
        "the reported damage is a view of the live fight state, which step() zeroes "
        "when it respawns the worlds that closed a round")
    assert ko_parts["damage"][:, 1].min() >= env._damage_cap
    print(f"health: {int(round(env._damage_cap))} clean hits put blue out (damage "
          f"{env._damage[:, 1].mean():.1f}/{env._damage_cap:.1f}), stun "
          f"{env._stun[:, 1].mean():.0f} steps, round closed as a knockout")

    # The jab check needs a spawn it can predict: the tilt and autoreset checks
    # above drew two more respawns from this env's rng, and with the default
    # 0.12 m / 0.3 rad jitter the scripted swing simply misses. Build a still
    # one instead -- same scorecard, no luck in it.
    jab_env = WarpBoxing(nworld=8, spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    jab_env.reset(0)
    ctrl0 = np.asarray(jab_env._ctrl0)
    red_ids = jab_env._sides[scene.RED]["actuators"]
    blue_ids = jab_env._sides[scene.BLUE]["actuators"]

    def column(ids, name):
        aid = mujoco.mj_name2id(jab_env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return int(np.flatnonzero(ids == aid)[0])

    def amp(ids, name, ctrl_value):
        col = column(ids, name)
        return (ctrl_value - ctrl0[ids[col]]) / jab_env._action_scale

    # The reach_sweep probe pose: shoulder pitch -1.5, elbow 0.0 puts a fist
    # on the torso at this distance; blue's arms go to their zero pose so no
    # forearm blocks the line.
    jab_pitch = amp(red_ids, "red/left_shoulder_pitch_joint", -1.5)
    jab_elbow = amp(red_ids, "red/left_elbow_joint", 0.0)
    action = np.zeros((jab_env.nworld, 2, jab_env.action_size))
    for half in ("left", "right"):
        for joint in scene.ARM_JOINTS:
            action[:, 1, column(blue_ids, f"blue/{half}_{joint}_joint")] = \
                amp(blue_ids, f"blue/{half}_{joint}_joint", 0.0)

    hits, jab_ends, jab_floor, jab_att = 0.0, 0, 0.0, 0.0
    for step in range(110):
        punch = max(0.0, np.sin(2.0 * np.pi * step / 30.0)) ** 2
        action[:, 0, column(red_ids, "red/left_shoulder_pitch_joint")] = jab_pitch * punch
        action[:, 0, column(red_ids, "red/left_elbow_joint")] = jab_elbow * punch
        obs, reward, done, parts = jab_env.step(action)
        hits = max(hits, float(parts["hit"].max()))
        jab_ends += int(done.any())
        jab_floor = max(jab_floor, float(parts["on_floor"].max()))
        jab_att += float(parts["attempts"][:, 0].sum())
    print(f"scripted jab over 110 steps: peak hit value {hits:.2f} "
          f"(force threshold {jab_env._force_threshold:.0f} N), rounds closed "
          f"{jab_ends}, on-floor seen {jab_floor:.0f}, attempts {jab_att:.0f}")
    assert np.isfinite(obs).all() and hits > 0.0, "self-check failed"
    assert jab_att > 0.0, "a full-speed swing registered no attempt"

    # The re-arm gate asks for the fist to come HOME, not merely for the contact to
    # break. Same gesture twice from the same spawn: armed, it must score; denied its
    # "came back" flag, the raw hit must still happen and score nothing.
    action[:, 0, column(red_ids, "red/left_shoulder_pitch_joint")] = jab_pitch
    action[:, 0, column(red_ids, "red/left_elbow_joint")] = jab_elbow

    def jab_window(deny):
        """The proven 110-step profile twice, from the same spawn: once armed, once
        with the fist denied its return to guard. The window has to be as long as the
        profile that scored -- a 30-step window sees no contact at all, which is itself
        worth knowing: the scripted jab lands late, not on the first swing."""
        jab_env.reset(0)
        scored_max = hit_max = 0.0
        for step in range(110):
            punch = max(0.0, np.sin(2.0 * np.pi * step / 30.0)) ** 2
            action[:, 0, column(red_ids, "red/left_shoulder_pitch_joint")] = \
                jab_pitch * punch
            action[:, 0, column(red_ids, "red/left_elbow_joint")] = jab_elbow * punch
            if deny:
                jab_env._came_back[:, 0] = False   # the fist never returned
            obs, reward, done, parts = jab_env.step(action)
            scored_max = max(scored_max, float(parts["scored_by"][:, 0].max()))
            hit_max = max(hit_max, float(parts["hit"].max()))
        return scored_max, hit_max

    scored_ok, hit_ref = jab_window(False)
    scored_denied, hit_denied = jab_window(True)
    print(f"re-arm gate: same jab scores {scored_ok:.2f} armed and "
          f"{scored_denied:.2f} denied, with raw contact {hit_ref:.2f} / "
          f"{hit_denied:.2f} under both (guard radius "
          f"{jab_env._retract_radius:.2f} m)")
    assert hit_ref > 0.0 and scored_ok > 0.0, "the scripted jab stopped scoring at all"
    assert scored_denied == 0.0 and hit_denied > 0.0, "the re-arm gate is not biting"

    # Imitation channel: drive the arm the targets name along the retargeted
    # strike poses and the shaped term must light up. Skipped when no targets
    # file is present (runs/ is out of git with the motion data).
    from pathlib import Path
    found = [p for p in (Path("runs/reference/jab_targets.json"),
                         Path("runs/reference/jab_targets_cmu.json"))
             if p.exists()]
    if not found:
        print("no jab_targets*.json -- imitation channel not exercised")
    for targets_file in found:
        env2 = WarpBoxing(nworld=8, spawn_distance=0.45,
                          imitation_path=str(targets_file),
                          spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
        env2.reset(1)
        cols = env2._imit_cols[scene.RED]
        aid = env2._sides[scene.RED]["actuators"][cols]
        ctrl0 = np.asarray(env2._ctrl0)
        action = np.zeros((8, 2, env2.action_size))
        # one clip frame spans the control clock it maps to: 50 Hz control vs
        # the clip's own 30 or 120 fps
        hold = max(1, int(round(50.0 / env2._imit_fps)))
        peak = 0.0
        for f in range(len(env2._imit)):
            action[:, 0, cols] = ((env2._imit[f] - ctrl0[aid])
                                  / env2._action_scale)
            for _ in range(hold):
                obs, reward, done, parts = env2.step(action)
                peak = max(peak, float(parts["imit"].max()))
        print(f"{targets_file.name} ({env2._imit_arm} arm, "
              f"{len(env2._imit)} frames, {env2._imit_fps} fps, hold {hold}): "
              f"peak imitation term {peak:.3f}")
        assert peak > 0.05, "imitation channel dead"
    print("self-check passed")
