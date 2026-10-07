"""Two boxers sparring inside one MJX world, exposed as a Brax pipeline env.

One physics world holds both robots, so the two agents act on the same step and
share one round: as soon as either boxer goes down, out of the ring or out of
range, the round ends for both. That keeps ``done`` a per-world scalar, which is
what makes the multi-agent bookkeeping in :mod:`boxer.selfplay` simple.

Landed punches are read off the real MJX contacts. Every striking pair declared
in :mod:`boxer.scene` is addressable by geom id, so a contact slot is classified
as "red's fist on blue's head" by gathering a per-geom role mask -- no proximity
heuristic stands in for the collision engine.
"""

from __future__ import annotations

from typing import Any, Dict, List

from brax import base as physics_base
from brax import math
from brax.envs import base as env_base
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import scene

NUM_AGENTS = 2
MAX_SEPARATION = 2.4

# Per-geom roles, as a bitmask so that a geom can be both a striker and a target
# (the forearm blocks, and it also lands in a clinch).
RED_STRIKE, BLUE_STRIKE, RED_TARGET, BLUE_TARGET = 1, 2, 4, 8

# What a landed punch on each target is worth, relative to a head shot.
TARGET_WEIGHTS: Dict[str, float] = {"head_collision": 1.0, "torso_collision": 0.6}
for _geom in scene.UPPER_TARGET_GEOMS:
    TARGET_WEIGHTS.setdefault(_geom, 0.25)
for _geom in scene.BODY_TARGET_GEOMS:
    TARGET_WEIGHTS.setdefault(_geom, 0.4)
del _geom

STRIKERS = scene.HAND_GEOMS + scene.FOREARM_GEOMS

_WORLD_UP = jp.array([0.0, 0.0, 1.0])
_WORLD_FORWARD = jp.array([1.0, 0.0, 0.0])

# Observation entries that do not scale with the joint count.
_OBS_FIXED = 41


def _norm(x: jax.Array, axis: int = -1) -> jax.Array:
    """Norm that is safe at zero and honours ``axis``.

    ``brax.math.safe_norm`` reduces over every axis regardless of the ``axis``
    argument, so it cannot be used on the stacked per-hand tensors here.
    """
    return jp.sqrt(jp.sum(jp.square(x), axis=axis) + 1e-12)


def _yaw_quat(angle: jax.Array) -> jax.Array:
    half = angle / 2.0
    return jp.stack([jp.cos(half), jp.zeros_like(half), jp.zeros_like(half), jp.sin(half)])


def _ids(model: mujoco.MjModel, objtype: int, names) -> np.ndarray:
    out = []
    for name in names:
        idx = mujoco.mj_name2id(model, objtype, name)
        if idx < 0:
            raise LookupError(f"{name!r} is missing from the composed scene")
        out.append(idx)
    return np.asarray(out, dtype=np.int32)


def _actuated_joints(model: mujoco.MjModel) -> List[str]:
    """Joint names driven by one G1's actuators, in actuator order."""
    robot = mujoco.MjModel.from_xml_path(str(scene.robot_xml_path()))
    return [
        mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_JOINT, robot.actuator_trnid[a, 0])
        for a in range(robot.nu)
    ]


class BoxingEnv(env_base.PipelineEnv):
    """Self-play boxing for two Unitree G1 humanoids."""

    def __init__(
        self,
        *,
        spawn_distance: float = 1.3,
        spawn_jitter: float = 0.12,
        yaw_jitter: float = 0.3,
        pose_noise: float = 0.02,
        n_frames: int = 5,
        backend: str = "mjx",
        visual: bool = False,
        target_range: float = 0.9,
        ring_radius: float = scene.RING_RADIUS,
        down_up_z: float = 0.15,
        down_head_z: float = 0.55,
        action_scale: float = 1.0,
        hit_depth_ref: float = 0.01,
        punch_speed_ref: float = 5.0,
        punch_speed_min: float = 1.0,
        w_alive: float = 0.2,
        # A round is ~50 control steps in practice, so 0.3 per step was already
        # worth ~15 across a round when it was set -- but the knockout pays once
        # and pays 10, and the measured behaviour is a boxer that dives at the
        # opponent to collect it. Uprightness has to dominate a single terminal
        # event for the collapse race to stop being the best policy.
        w_upright: float = 0.8,
        w_range: float = 0.3,
        w_facing: float = 0.2,
        w_guard: float = 0.2,
        w_strike: float = 0.3,
        w_hit: float = 3.0,
        w_energy: float = 0.05,
        w_ko: float = 3.0,
        w_ring: float = 1.0,
    ):
        model = scene.build_model(spawn_distance=spawn_distance, visual=visual)
        sys = mjcf.load_model(model)
        super().__init__(sys=sys, backend=backend, n_frames=n_frames)

        self._spawn_jitter = spawn_jitter
        self._yaw_jitter = yaw_jitter
        self._pose_noise = pose_noise
        self._target_range = target_range
        self._ring_radius = ring_radius
        self._down_up_z = down_up_z
        self._down_head_z = down_head_z
        self._action_scale = action_scale
        self._hit_depth_ref = hit_depth_ref
        self._punch_speed_ref = punch_speed_ref
        self._punch_speed_min = punch_speed_min
        self._w = dict(
            alive=w_alive, upright=w_upright, range=w_range, facing=w_facing,
            guard=w_guard, strike=w_strike, hit=w_hit, energy=w_energy,
            ko=w_ko, ring=w_ring,
        )
        self._spawn_distance = spawn_distance

        joints = _actuated_joints(model)
        self._num_joints = len(joints)

        self._sensor_adr: Dict[str, int] = {}
        self._sensor_dim: Dict[str, int] = {}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = int(model.sensor_adr[i])
            self._sensor_dim[name] = int(model.sensor_dim[i])

        self._sides: Dict[str, Dict[str, Any]] = {}
        for side in scene.SIDES:
            joint = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, [side + n for n in joints])
            free = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, [side + "floating_base_joint"])[0]
            self._sides[side] = dict(
                qpos_adr=jp.array(model.jnt_qposadr[joint]),
                dof_adr=jp.array(model.jnt_dofadr[joint]),
                free_adr=int(model.jnt_qposadr[free]),
            )
        # The flat action vector is red-then-blue; check that against the model
        # rather than trusting the attach order.
        self._actuators = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        actuator_ids = np.concatenate([
            _ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [side + n for n in joints])
            for side in scene.SIDES
        ])
        if not np.array_equal(np.sort(actuator_ids), np.arange(model.nu)):
            raise AssertionError("actuator order does not match the agent order")

        qpos, ctrl = scene.stance(model)
        self._stance_qpos = jp.array(qpos.astype(np.float32))
        self._stance_ctrl = jp.array(ctrl.astype(np.float32))
        self._stance_joints = self._stance_ctrl.reshape(NUM_AGENTS, self._num_joints)
        self._guard = jp.array(self._guard_targets(model, qpos, ctrl))
        self._ctrl_lo = jp.array(np.asarray(sys.actuator_ctrlrange[:, 0], dtype=np.float32))
        self._ctrl_hi = jp.array(np.asarray(sys.actuator_ctrlrange[:, 1], dtype=np.float32))

        self._contact_head_weight = TARGET_WEIGHTS["head_collision"]
        role, weight = self._contact_tables(model)
        self._slot = self._slot_tables(role, weight)
        self._slot = self._slot_tables(role, weight)

    # -- static tables ----------------------------------------------------

    @staticmethod
    def _guard_targets(model, qpos, ctrl) -> np.ndarray:
        """Where each fist sits in the torso frame in the upstream guard pose."""
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        out = np.zeros((len(scene.SIDES), 2, 3), dtype=np.float32)
        for i, side in enumerate(scene.SIDES):
            torso = data.site(side + "torso")
            to_local = torso.xmat.reshape(3, 3).T
            for j, palm in enumerate(("left_palm", "right_palm")):
                out[i, j] = to_local @ (data.site(side + palm).xpos - torso.xpos)
        return out

    @staticmethod
    def _contact_tables(model) -> tuple[np.ndarray, np.ndarray]:
        role = np.zeros(model.ngeom, dtype=np.int32)
        weight = np.zeros(model.ngeom, dtype=np.float32)
        for side in scene.SIDES:
            strike = RED_STRIKE if side == scene.RED else BLUE_STRIKE
            target = RED_TARGET if side == scene.RED else BLUE_TARGET
            for geom in STRIKERS:
                role[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side + geom)] |= strike
            for geom, value in TARGET_WEIGHTS.items():
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side + geom)
                if gid >= 0:
                    role[gid] |= target
                    weight[gid] = value
        return role, weight

    # -- sizes ------------------------------------------------------------

    @property
    def action_size(self) -> int:
        return self._num_joints

    @property
    def observation_size(self) -> int:
        # Blocks that do not scale with the joint count: 3 up-axis + 3 torso
        # velocity + 3 gyro + 6 guard error + 6 fist velocities, then 3 opponent
        # torso + 3 opponent head + 6 opponent fists + 3 opponent tilt + 3 ring
        # centre + 2 distance/facing.
        return _OBS_FIXED + 3 * self._num_joints

    def actuator_index(self, name: str) -> int:
        """Position of one actuator in the action vector, by scene name.

        Lets scripted opponents and evaluation code refer to ``left_elbow_joint``
        instead of hard-coding an index that shifts with the robot description.
        """
        if name not in self._actuators:
            raise KeyError(f"{name!r} is not an actuator in the composed scene")
        return self._actuators[name]

    # -- sensing ----------------------------------------------------------

    def _read(self, sensordata: jax.Array, name: str) -> jax.Array:
        adr = self._sensor_adr[name]
        return sensordata[..., adr : adr + self._sensor_dim[name]]

    def _frame(self, pipeline_state: physics_base.State, side: str) -> Dict[str, jax.Array]:
        """Everything one boxer can sense, in world coordinates."""
        sensors = pipeline_state.sensordata
        index = self._sides[side]
        return dict(
            torso=self._read(sensors, side + "torso_pos"),
            head=self._read(sensors, side + "head_pos"),
            left_palm=self._read(sensors, side + "left_palm_pos"),
            right_palm=self._read(sensors, side + "right_palm_pos"),
            torso_vel=self._read(sensors, side + "torso_linvel"),
            left_palm_vel=self._read(sensors, side + "left_palm_linvel"),
            right_palm_vel=self._read(sensors, side + "right_palm_linvel"),
            quat=self._read(sensors, side + "torso_quat"),
            gyro=self._read(sensors, side + "imu_in_torso_gyro"),
            qpos=pipeline_state.qpos[index["qpos_adr"]],
            qvel=pipeline_state.qvel[index["dof_adr"]],
        )

    def _upright(self, frame: Dict[str, jax.Array]) -> jax.Array:
        return math.rotate(_WORLD_UP, frame["quat"])[2]

    def _down(self, frame: Dict[str, jax.Array]) -> jax.Array:
        down = (self._upright(frame) < self._down_up_z) | (
            frame["head"][2] < self._down_head_z
        )
        return jp.where(down, 1.0, 0.0)

    def _ring_distance(self, frame: Dict[str, jax.Array]) -> jax.Array:
        return _norm(frame["torso"][:2])

    def _obs(
        self,
        pipeline_state: physics_base.State,
        side: str,
        opponent: str,
        last_action: jax.Array,
    ) -> jax.Array:
        me = self._frame(pipeline_state, side)
        foe = self._frame(pipeline_state, opponent)
        agent = scene.SIDES.index(side)
        quat = me["quat"]
        to_local = lambda v: math.inv_rotate(v, quat)

        offset = foe["torso"] - me["torso"]
        distance = _norm(offset)
        forward = math.rotate(_WORLD_FORWARD, quat)
        guard = self._guard[agent]
        hands = jp.stack([
            to_local(me["left_palm"] - me["torso"]),
            to_local(me["right_palm"] - me["torso"]),
        ])
        to_centre = jp.concatenate([-me["torso"][:2], jp.zeros(1)])

        return jp.concatenate([
            math.rotate(_WORLD_UP, quat),                     # 3   balance
            to_local(me["torso_vel"]),                        # 3
            me["gyro"],                                       # 3
            (hands - guard).reshape(-1),                      # 6   guard error
            to_local(me["left_palm_vel"]),                    # 3
            to_local(me["right_palm_vel"]),                   # 3
            me["qpos"] - self._stance_joints[agent],          # 29
            me["qvel"],                                       # 29
            last_action,                                      # 29
            to_local(offset),                                 # 3
            to_local(foe["head"] - me["torso"]),              # 3
            to_local(foe["left_palm"] - me["torso"]),         # 3
            to_local(foe["right_palm"] - me["torso"]),        # 3
            to_local(math.rotate(_WORLD_UP, foe["quat"])),    # 3   opponent tilt
            to_local(to_centre),                              # 3
            jp.stack([distance, jp.dot(forward, offset / (distance + 1e-6))]),  # 2
        ])

    def _slot_tables(self, role, weight) -> Dict[str, jax.Array]:
        """Folds the per-geom role mask into per-contact-slot constants.

        MJX allocates one contact slot per declared geom pair and the slot to
        pair mapping does not vary with the state, so a slot's role can be
        resolved once here. The hit term is evaluated on every physics substep
        -- five per control step -- and gathering the per-geom mask each time
        accounted for about half of the env step.
        ``tests.test_env.test_contact_slot_mapping_is_static`` pins the
        assumption this rests on, so a MuJoCo change that reorders the buffer
        fails a test instead of silently miscounting punches.
        """
        pipeline_state = self.pipeline_init(
            self._stance_qpos, jp.zeros(self.sys.nv), None, self._stance_ctrl
        )
        geom = np.asarray(pipeline_state.contact.geom)
        near, far = geom[:, 0], geom[:, 1]

        def classify(strike_bit: int, target_bit: int) -> Dict[str, jax.Array]:
            forward = ((role[near] & strike_bit) != 0) & ((role[far] & target_bit) != 0)
            reverse = ((role[far] & strike_bit) != 0) & ((role[near] & target_bit) != 0)
            landed = (forward | reverse).astype(np.float32)
            struck = np.where(forward, weight[far], np.where(reverse, weight[near], 0.0))
            struck = struck.astype(np.float32)
            return {
                "landed": jp.array(landed),
                "weight": jp.array(struck),
                "head": jp.array((struck >= self._contact_head_weight).astype(np.float32)),
            }

        red = classify(RED_STRIKE, BLUE_TARGET)
        blue = classify(BLUE_STRIKE, RED_TARGET)
        return {"red": red, "blue": blue}

    def _contact_damage(self, pipeline_state: physics_base.State) -> Dict[str, jax.Array]:
        """Damage and hit counts, split by who landed the punch.

        MJX keeps a slot for every declared pair whether or not it is colliding,
        and reports the true separation in ``dist``; a slot only counts once that
        separation goes inside the contact margin.
        """
        contact = pipeline_state.contact
        landed = (contact.dist < contact.includemargin).astype(jp.float32)
        depth = jp.clip(-contact.dist / self._hit_depth_ref, 0.0, 1.0)

        out = {}
        for side, opponent in (("red", "blue"), ("blue", "red")):
            slot = self._slot[side]
            strike = landed * slot["landed"]
            out[f"{side}_damage"] = jp.sum(strike * depth * slot["weight"])
            out[f"{side}_hits"] = jp.sum(strike)
            out[f"{side}_heads"] = jp.sum(strike * slot["head"])
        return out

    def _punch_signals(self, frame: Dict[str, jax.Array], foe: Dict[str, jax.Array]):
        """Per-hand (torso-relative fist speed, part of it closing on a target).

        Speed is measured against the striker's own torso because world-frame
        speed cannot tell a punch from a fall. Direction is measured against the
        *target*, which is what speed alone cannot tell a punch from a lunge: a
        boxer falling forward carries a fast fist that is closing on nothing.
        """
        fists = jp.stack([frame["left_palm"], frame["right_palm"]])
        velocity = jp.stack([frame["left_palm_vel"], frame["right_palm_vel"]])
        relative = velocity - frame["torso_vel"]
        targets = jp.stack([foe["head"], foe["torso"]])
        toward = targets[None, :, :] - fists[:, None, :]
        unit = toward / _norm(toward)[:, :, None]
        closing = jp.max(jp.sum(unit * relative[:, None, :], axis=-1), axis=-1)
        return _norm(relative, axis=-1), closing

    def _committed(self, frame: Dict[str, jax.Array], foe: Dict[str, jax.Array]) -> jax.Array:
        """Whether this boxer is throwing a punch, as a 0..1 gate.

        Three conditions, each one added because measurement killed the pair
        before it:

        * fist speed *relative to its own torso*. World-frame fist speed cannot
          separate a punch from a fall -- a collapsing body carries both fists
          along at several m/s.
        * the striker still being upright. Two boxers standing perfectly passively
          drift together, topple, and register 12 "punches" a round purely from
          the arms flailing during the collapse, with torso-relative speed well
          over the threshold.
        * the fast fist actually *closing on* the head or torso. Rendered: a
          trained boxer falls forward with its arm over the opponent's chest,
          which is upright enough and fast enough to pass both tests above, and
          is not a punch. Requiring the velocity to point at a target is what
          distinguishes throwing from toppling.
        """
        speed, closing = self._punch_signals(frame, foe)
        fastest = jp.max(speed)
        low, high = self._punch_speed_min, self._punch_speed_ref
        swinging = jp.clip((fastest - low) / (high - low), 0.0, 1.0)
        aim = jp.clip((jp.max(closing) / (fastest + 1e-6) - 0.5) / 0.3, 0.0, 1.0)
        upright = jp.clip((self._upright(frame) - 0.9) / 0.08, 0.0, 1.0)
        return swinging * aim * upright

    def _reward(
        self,
        pipeline_state: physics_base.State,
        side: str,
        opponent: str,
        action: jax.Array,
        contact: Dict[str, jax.Array],
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        me = self._frame(pipeline_state, side)
        foe = self._frame(pipeline_state, opponent)
        w = self._w
        agent = scene.SIDES.index(side)

        upright = jp.clip(self._upright(me), 0.0, 1.0)
        alive = 1.0 - self._down(me)
        offset = foe["torso"] - me["torso"]
        distance = _norm(offset)
        forward = math.rotate(_WORLD_FORWARD, me["quat"])
        facing = jp.clip(jp.dot(forward, offset / (distance + 1e-6)), 0.0, 1.0)
        in_range = jp.exp(-((distance - self._target_range) / 0.35) ** 2)

        guard = self._guard[agent]
        hands = jp.stack([
            math.inv_rotate(me["left_palm"] - me["torso"], me["quat"]),
            math.inv_rotate(me["right_palm"] - me["torso"], me["quat"]),
        ])
        held = jp.clip(1.0 - _norm(hands - guard) / 0.25, 0.0, 1.0)

        # Dense shaping only: pay a fist that is extending fast *toward* a
        # scoring target. Speed is measured against the torso for the same reason
        # the contact gate does it -- a falling body must not be paid for
        # punching -- and the closing term is there so a forward topple, which is
        # fast and torso-relative too, earns nothing.
        speed, closing = self._punch_signals(me, foe)
        swing = (
            jp.clip(speed / self._punch_speed_ref, 0.0, 1.0)
            * jp.clip(closing / (speed + 1e-6), 0.0, 1.0)
        )
        fists = jp.stack([me["left_palm"], me["right_palm"]])[:, None, :]
        targets = jp.stack([foe["head"], foe["torso"]])[None, :, :]
        gap = _norm(fists - targets)
        strike = jp.sum(swing[:, None] * jp.exp(-(gap / 0.30) ** 2) * jp.array([[1.0, 0.6]]))

        damage = (contact["red_damage"], contact["blue_damage"])
        hits = (contact["red_hits"], contact["blue_hits"])
        raw_landed, raw_taken = hits[agent], hits[1 - agent]
        landed, taken = damage[agent], damage[1 - agent]

        # A contact only counts as a punch if the striking fist is actually
        # moving. Two boxers that simply lean into each other register hand-on-
        # torso contacts (measured: they drift from 1.3 m apart to 0.42 m while
        # both hold a passive guard), and without this gate those collisions are
        # indistinguishable from punches in the arena's win/hit statistics.
        my_commitment = self._committed(me, foe)
        foes_commitment = self._committed(foe, me)
        landed = damage[agent] * my_commitment
        taken = damage[1 - agent] * foes_commitment
        landed_hits = raw_landed * my_commitment
        taken_hits = raw_taken * foes_commitment
        # head_shots must go through the same gate as hits; reporting it ungated
        # made the arena show more head shots than punches, which is impossible
        # when one is a subset of the other.
        head_shots = (
            contact["red_heads"] if agent == 0 else contact["blue_heads"]
        ) * my_commitment

        centre = self._ring_distance(me)
        reward = (
            w["alive"] * alive
            + w["upright"] * upright
            # Being at fighting distance only pays while the boxer is upright.
            # Multiplying by `alive` instead let a forward topple collect the
            # range reward on the way down, which is the exact behaviour the
            # rendered rollout showed.
            + w["range"] * in_range * alive * upright
            + w["facing"] * facing * alive
            + w["guard"] * jp.mean(held) * alive
            + w["strike"] * strike * alive
            + w["hit"] * (landed - taken)
            - w["energy"] * jp.mean(jp.square(action))
            - w["ring"] * jp.clip(centre - 0.75 * self._ring_radius, 0.0, 10.0)
        )
        parts = dict(
            upright=upright, in_range=in_range, facing=facing, guard=jp.mean(held),
            strike=strike, landed=landed, taken=taken, distance=distance, centre=centre,
            landed_hits=landed_hits, taken_hits=taken_hits, head_shots=head_shots,
            contacts=0.5 * (raw_landed + raw_taken),
        )
        return reward, parts

    # -- env API ----------------------------------------------------------

    def reset(self, rng: jax.Array) -> env_base.State:
        rng_spawn, rng_yaw, rng_joints, rng_vel = jax.random.split(rng, 4)

        qpos = self._stance_qpos
        jitter = jax.random.uniform(rng_spawn, (2,), minval=-1.0, maxval=1.0)
        half = self._spawn_distance / 2.0 + jitter[0] * self._spawn_jitter
        lateral = jitter[1] * self._spawn_jitter
        yaw = jax.random.uniform(rng_yaw, (2,), minval=-self._yaw_jitter, maxval=self._yaw_jitter)

        noise = jax.random.normal(rng_joints, (NUM_AGENTS, self._num_joints)) * self._pose_noise
        for i, side in enumerate(scene.SIDES):
            index = self._sides[side]
            free = index["free_adr"]
            sign = -1.0 if i == 0 else 1.0
            qpos = qpos.at[free : free + 3].set(
                jp.stack([sign * half, sign * lateral, qpos[free + 2]])
            )
            # The blue corner already faces -x in the composed scene, so the yaw
            # jitter is applied in each boxer's own frame.
            qpos = qpos.at[free + 3 : free + 7].set(
                math.quat_mul(qpos[free + 3 : free + 7], _yaw_quat(yaw[i]))
            )
            qpos = qpos.at[index["qpos_adr"]].add(noise[i])

        qvel = jax.random.normal(rng_vel, (self.sys.nv,)) * (self._pose_noise * 2.5)
        pipeline_state = self.pipeline_init(qpos, qvel, None, self._stance_ctrl)
        action = jp.zeros((NUM_AGENTS, self._num_joints))
        return env_base.State(
            pipeline_state=pipeline_state,
            obs=self._observe(pipeline_state, action),
            reward=jp.zeros(NUM_AGENTS),
            done=jp.float32(0.0),
            metrics=self._zero_metrics(),
            info={"last_action": action},
        )

    def _observe(self, pipeline_state: physics_base.State, action: jax.Array) -> jax.Array:
        return jp.stack([
            self._obs(pipeline_state, side, scene.OPPONENT[side], action[i])
            for i, side in enumerate(scene.SIDES)
        ])

    def _step_with_contacts(self, pipeline_state, ctrl: jax.Array):
        """Advances one control step, summing contact evidence over every substep.

        A punch is in contact for one or two 4 ms substeps, so sampling the
        contact buffer only at the end of the 20 ms control step would miss most
        of the hits that decide the reward -- and a reward that fires at random
        is worse than no reward, because it teaches the critic that punching
        does not pay.
        """
        def body(state, _):
            state = self._pipeline.step(self.sys, state, ctrl, self._debug)
            return state, self._contact_damage(state)

        pipeline_state, per_substep = jax.lax.scan(
            body, pipeline_state, (), self._n_frames
        )
        return pipeline_state, jax.tree_util.tree_map(jp.sum, per_substep)

    def step(self, state: env_base.State, action: jax.Array) -> env_base.State:
        action = jp.clip(action, -1.0, 1.0)
        span = (self._ctrl_hi - self._ctrl_lo) / 2.0 * self._action_scale
        ctrl = jp.clip(self._stance_ctrl + action.reshape(-1) * span, self._ctrl_lo, self._ctrl_hi)

        pipeline_state, contact = self._step_with_contacts(state.pipeline_state, ctrl)

        frames = {side: self._frame(pipeline_state, side) for side in scene.SIDES}
        down = jp.stack([self._down(frames[side]) for side in scene.SIDES])
        outside = jp.stack([
            self._ring_distance(frames[side]) > self._ring_radius for side in scene.SIDES]
        ).astype(jp.float32)
        separation = _norm(frames[scene.RED]["torso"] - frames[scene.BLUE]["torso"])
        done = jp.clip(jp.max(down) + jp.max(outside) + (separation > MAX_SEPARATION), 0.0, 1.0)

        rewards, parts = [], []
        for i, side in enumerate(scene.SIDES):
            # A knockout only pays if the winner is still upright, and costs the
            # loser the same amount: the round is zero-sum at the terminal step.
            knockout = down[1 - i] * (1.0 - down[i]) - down[i] * (1.0 - down[1 - i])
            reward, part = self._reward(
                pipeline_state, side, scene.OPPONENT[side], action[i], contact
            )
            rewards.append(reward + self._w["ko"] * knockout)
            parts.append(part)

        metrics = {
            "upright": jp.mean(jp.stack([p["upright"] for p in parts])),
            "guard": jp.mean(jp.stack([p["guard"] for p in parts])),
            "strike": jp.mean(jp.stack([p["strike"] for p in parts])),
            "damage": jp.mean(jp.stack([p["landed"] for p in parts])),
            "damage_taken": jp.mean(jp.stack([p["taken"] for p in parts])),
            "hits_landed": jp.mean(jp.stack([p["landed_hits"] for p in parts])),
            "contacts": jp.mean(jp.stack([p["contacts"] for p in parts])),
            "head_shots": jp.mean(jp.stack([p["head_shots"] for p in parts])),
            # Per corner as well as pooled: "who won because they punched" cannot
            # be answered by the mean of two boxers, and the mean is what the
            # arena would otherwise report.
            "hits_red": parts[0]["landed_hits"],
            "hits_blue": parts[1]["landed_hits"],
            "heads_red": parts[0]["head_shots"],
            "heads_blue": parts[1]["head_shots"],
            "knockout": jp.max(down) * (1.0 - jp.min(down)),
            "down_red": down[0],
            "down_blue": down[1],
            "ring_out": jp.max(outside),
            "distance": parts[0]["distance"],
        }
        # `replace` rather than a fresh State: the training wrappers keep their
        # episode bookkeeping in `info`, which a new State would drop.
        state.metrics.update(metrics)
        state.info["last_action"] = action
        return state.replace(
            pipeline_state=pipeline_state,
            obs=self._observe(pipeline_state, action),
            reward=jp.stack(rewards),
            done=done,
        )

    @staticmethod
    def _zero_metrics() -> Dict[str, jax.Array]:
        return {
            name: jp.float32(0.0)
            for name in (
                "upright", "guard", "strike", "damage", "damage_taken",
                "hits_landed", "contacts", "head_shots", "hits_red", "hits_blue",
                "heads_red", "heads_blue", "knockout", "down_red",
                "down_blue", "ring_out", "distance",
            )
        }
