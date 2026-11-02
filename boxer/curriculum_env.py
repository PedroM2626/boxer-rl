"""Unassisted G1 balance/locomotion, built on the pinned MuJoCo Playground task.

All 29 joints, including all 12 leg joints, are policy actions. No pelvis
springs, root tracking, gravity compensation or pose playback are used.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import jax
import mujoco
import jax.numpy as jp
import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick

from boxer import moves
from boxer.scene import ARM_JOINTS, GUARD_ARMS_BOXING
from boxer.curriculum_control import STAGES, TRAINABLE_STAGES

PLAYGROUND_REVISION = "12ad7f3842791ced39f27e5ac96d066bbe361653"
MENAGERIE_REVISION = "1b86ece576591213e2b666ebf59508454200ca97"

# Straight-punch endpoints and timing (see boxer.moves for hooks and guards).
PUNCH_BODY, PUNCH_HEAD = moves.STRAIGHT_BODY, moves.STRAIGHT_HEAD
PUNCH_EXTEND, PUNCH_RETRACT = moves.TIMING[moves.STRAIGHT]
HOOK_FRACTION = .4         # share of generated punches that are hooks
HIGH_GUARD_FRACTION = .4   # share of guard periods held high
PUNCH_WAIT = (5, 60)       # control steps between generated punches
# Rapid punches (``--rapid-punches``): native tactics throw ~2.2 punches/s per
# boxer, nearly back to back, and the arm skill adds up to 0.35 rad to the
# punching arm (boxer.arm_skill.RESIDUAL_SCALE); the default generator throws ~1/s.
RAPID_PUNCH_WAIT = (0, 6)
RAPID_PUNCH_RESIDUAL = .35
# Phantom opponent shins (``--phantom-shins``): in native bouts every knockdown
# after body contact followed the boxers' shins tangling (tools/knockdown_causes.py).
PHANTOM_GAP = (2.0, 6.0)        # seconds between appearances
PHANTOM_SPAN = (0.3, 2.0)       # seconds each appearance lasts
PHANTOM_START = 0.45            # m from the pelvis when an appearance starts
# Closest approach, m from the pelvis. In native bouts shins touch at root
# distances of 0.45-0.9 m with the opponent's shins ~0.1-0.2 m before its pelvis.
PHANTOM_CLOSEST = (0.18, 0.30)
PHANTOM_PARK = jp.array([0., 0., -10.])
LEFT_ARM, RIGHT_ARM = 15, 22  # first shoulder actuator of each arm


def configure_assets():
    cached = Path(os.environ.get("BOXER_CURRICULUM_MENAGERIE",
                                 Path.home() / ".cache/boxing-curriculum/menagerie"))
    if cached.exists():
        revision = subprocess.check_output(
            ["git", "-C", str(cached), "rev-parse", "HEAD"], text=True).strip()
        if revision != MENAGERIE_REVISION:
            raise ValueError(f"curriculum assets need {MENAGERIE_REVISION}, found {revision}")
        mjx_env.MENAGERIE_PATH = epath.Path(cached)
    else:
        mjx_env.ensure_menagerie_exists()


def stage_config(stage, impl="warp"):
    if stage not in TRAINABLE_STAGES:
        raise ValueError(f"unknown curriculum stage {stage!r}")
    cfg = joystick.default_config()
    cfg.impl = impl
    cfg.episode_length = 500
    # One global contact allocation for a modest laptop batch, rather than the
    # upstream 8192-world default. Each robot has very few feet-only contacts.
    cfg.naconmax = 8192
    # Fresh random poses can activate more constraint rows than the upstream
    # 90-row feet-only default, even when the sampled reset is not selected.
    cfg.njmax = 256
    # Train recovery over the same impulse scale used by the frozen gate.
    # These are bounded disturbances, never stabilizing root forces.
    cfg.push_config.enable = True
    cfg.push_config.interval_range = [3.0, 5.0]
    cfg.push_config.magnitude_range = [0.10, 0.30]
    cfg.noise_config.level = 0.25
    cfg.reward_config.scales.action_rate = -0.01
    if stage == "balance":
        # Mostly standing, with gentle weight shifts rather than a hard
        # separation between stationary balance and the first movement commands.
        cfg.lin_vel_x = [-0.03, 0.05]
        cfg.lin_vel_y = [-0.02, 0.02]
        cfg.ang_vel_yaw = [-0.05, 0.05]
        cfg.reward_config.scales.feet_phase = 0.0
        cfg.reward_config.scales.feet_air_time = 0.0
        cfg.reward_config.scales.alive = 1.0
        cfg.reward_config.scales.pose = -0.25
        cfg.reward_config.scales.base_height = -5.0
        cfg.reward_config.base_height_target = 0.755
    else:
        cfg.lin_vel_x = [-0.35, 0.55]
        cfg.lin_vel_y = [-0.25, 0.25]
        cfg.ang_vel_yaw = [-0.5, 0.5]
        if stage == "walk_easy":
            cfg.lin_vel_x = [v * 0.25 for v in cfg.lin_vel_x]
            cfg.lin_vel_y = [v * 0.25 for v in cfg.lin_vel_y]
            cfg.ang_vel_yaw = [v * 0.25 for v in cfg.ang_vel_yaw]
        if stage == "spar":
            # Received punches: more frequent and harder shoves than walking.
            cfg.push_config.interval_range = [1.0, 3.0]
            cfg.push_config.magnitude_range = [0.10, 0.50]
    return cfg


class UnassistedG1(joystick.Joystick):
    """Same action/observation contract in standing and command-following tasks."""

    def __init__(self, stage="balance", impl="warp", config=None):
        configure_assets()
        self.stage = stage
        super().__init__(config=config or stage_config(stage, impl))
        self._phantom = None
        if self._config.get("phantom_shins", False):
            if stage != "spar":
                raise ValueError("phantom shins belong to the spar stage")
            self._add_phantom_shins()
            self._phantom = True
        # Carry a compact boxing guard while balancing/walking, so transferring
        # to the boxing posture will not suddenly shift the centre of mass.
        for hand, values in GUARD_ARMS_BOXING.items():
            for joint, value in zip(ARM_JOINTS, values):
                index = int(self.mj_model.joint(f"{hand}_{joint}_joint").qposadr[0])
                self._init_q = self._init_q.at[index].set(value)
                self._default_pose = self._default_pose.at[index - 7].set(value)
        self.leg_action_indices = np.array([
            a for a in range(self.mj_model.nu)
            if any(word in self.mj_model.actuator(a).name
                   for word in ("hip", "knee", "ankle"))], dtype=np.int32)
        if len(self.leg_action_indices) != 12 or self.action_size != 29:
            raise ValueError("unexpected G1 joint contract")
        if not np.array_equal(self.mj_model.jnt_qposadr[
                self.mj_model.actuator_trnid[:, 0]], np.arange(7, 36)):
            raise ValueError("actuator order must match the observed joint order")
        if self.mj_model.neq or np.any(self.mj_model.body_gravcomp):
            raise ValueError("unassisted curriculum forbids welds and gravity compensation")

    def _metrics(self, data, info):
        velocity = self.get_local_linvel(data, "pelvis")
        return dict(
            upright=self.get_gravity(data, "torso")[-1],
            root_height=data.qpos[2],
            linear_tracking_error=jp.linalg.norm(velocity[:2] - info["command"][:2]),
            yaw_tracking_error=jp.abs(self.get_gyro(data, "pelvis")[2] - info["command"][2]),
            forward_speed=velocity[0],
            leg_motion=jp.mean(jp.abs(data.qpos[7:][self.leg_action_indices]
                                    - self._default_pose[self.leg_action_indices])),
        )

    def sample_command(self, rng):
        motion_rng, standing_rng = jax.random.split(rng)
        command = super().sample_command(motion_rng)
        # Retain standing practice as movement becomes harder. Upstream already
        # has a 10% zero-command branch: account for it in the total probability.
        probability = {"balance": 0.70, "spar": 0.50}.get(self.stage, 0.40)
        extra_zero = (probability - 0.10) / 0.90
        command = jp.where(jax.random.bernoulli(standing_rng, extra_zero), jp.zeros(3), command)
        lateral_probability = self._config.get("lateral_practice_probability", 0.0)
        if self.stage == "walk_easy" and lateral_probability > 0:
            # Replace only moving commands: stationary practice stays at 40%.
            # This requests footwork; no force, joint target, or gait is imposed.
            choose, sign, amplitude = jax.random.split(jax.random.fold_in(rng, 606), 3)
            limit = min(abs(v) for v in self._config.lin_vel_y)
            speed = jax.random.uniform(amplitude, minval=limit * .5, maxval=limit)
            speed *= jp.where(jax.random.bernoulli(sign), 1.0, -1.0)
            lateral = jp.array([0.0, speed, 0.0])
            replace = (jp.any(command != 0)
                       & jax.random.bernoulli(choose, lateral_probability))
            command = jp.where(replace, lateral, command)
        return command

    def _reward_feet_air_time(self, air_time, first_contact, commands, **kwargs):
        reward = super()._reward_feet_air_time(air_time, first_contact, commands, **kwargs)
        return reward * (jp.linalg.norm(commands) > 0.01)

    def reset(self, rng):
        state = super().reset(rng)
        if self.stage == "spar" and self._phantom is not None:
            state.info.update(phantom_timer=jax.random.uniform(jax.random.fold_in(rng, 79), (),
                                                               minval=PHANTOM_GAP[0], maxval=PHANTOM_GAP[1]),
                              phantom_span=jp.array(1.0), phantom_age=jp.array(-1.0),
                              phantom_angle=jp.array(0.0), phantom_yaw=jp.array(0.0),
                              phantom_closest=jp.array(PHANTOM_CLOSEST[1]),
                              phantom_rng=jax.random.fold_in(rng, 80))
            state = state.replace(data=state.data.replace(mocap_pos=PHANTOM_PARK[None]))
        if self.stage == "spar":
            punch_rng = jax.random.fold_in(rng, 77)
            state.info.update(
                punch_rng=punch_rng, punch_age=jp.array(-1.0),
                punch_hand=jp.array(0, jp.int32), punch_target=jp.zeros(4),
                punch_speed=jp.array(1.0), punch_move=jp.array(0, jp.int32),
                punch_aim=jp.array(0.), high_guard=jp.array(False), punch_residual=jp.zeros(4),
                guard_timer=jax.random.randint(jax.random.fold_in(rng, 78), (), 25, 150),
                punch_wait=jax.random.randint(punch_rng, (), 10, 60))
        if self.stage == "approach":
            # The target is a navigation marker, never a physical support.
            forward = state.data.site_xmat[self._pelvis_imu_site_id][:2, 0]
            distance = jax.random.uniform(jax.random.fold_in(rng, 91),
                                          minval=1.8, maxval=2.2)
            state.info["goal"] = state.data.qpos[:2] + distance * forward
            state = self._approach_command(state)
        state.metrics.update(self._metrics(state.data, state.info))
        if self.stage == "approach":
            state.metrics["goal_distance"] = jp.linalg.norm(
                state.info["goal"] - state.data.qpos[:2])
        return state

    def punch_arms(self, info):
        """Arm actions for generated moves; zero holds the normal guard (default pose).

        The policy's own arm outputs are replaced: it only keeps balance and
        footwork while the arms throw randomized straight punches and hooks
        (boxer.moves) at body/head height, with combos, pauses, speed and
        endpoint variation, and switch between the normal and high guard.
        """
        rng, hand_rng, aim_rng, jitter_rng, speed_rng, wait_rng, move_rng, guard_rng, timer_rng = (
            jax.random.split(info["punch_rng"], 9))
        residual_rng = jax.random.fold_in(info["punch_rng"], 1)
        rapid = self._config.get("rapid_punches", False)
        wait_range = RAPID_PUNCH_WAIT if rapid else PUNCH_WAIT
        timer = info["guard_timer"] - 1
        high = jp.where(timer <= 0, jax.random.bernoulli(guard_rng, HIGH_GUARD_FRACTION),
                        info["high_guard"])
        timer = jp.where(timer <= 0, jax.random.randint(timer_rng, (), 25, 150), timer)
        launch = (info["punch_age"] < 0) & (info["punch_wait"] <= 0)
        aim = jp.where(launch, jax.random.uniform(aim_rng), info["punch_aim"])
        move = jp.where(launch, jax.random.bernoulli(move_rng, HOOK_FRACTION).astype(jp.int32),
                        info["punch_move"])
        target = moves.strike_pose(jp, move, aim,
                                   jax.random.uniform(jitter_rng, (4,), minval=-.12, maxval=.12))
        hand = jp.where(launch, jax.random.randint(hand_rng, (), 0, 2), info["punch_hand"])
        target = jp.where(launch, target, info["punch_target"])
        speed = jp.where(launch, jax.random.uniform(speed_rng, minval=.8, maxval=1.25),
                         info["punch_speed"])
        age = jp.where(launch, 0.0, info["punch_age"])
        # Like an arm-skill correction, in the left arm's frame (mirrored with the swing).
        residual = jp.where(launch, jax.random.uniform(residual_rng, (4,), minval=-RAPID_PUNCH_RESIDUAL,
                                                       maxval=RAPID_PUNCH_RESIDUAL) * rapid,
                            info["punch_residual"])
        guards =[self._default_pose[LEFT_ARM:LEFT_ARM + 4], self._default_pose[RIGHT_ARM:RIGHT_ARM + 4]]
        arms = []
        for side, guard in enumerate(guards):
            rest = moves.mirror(jp, moves.rest_pose(jp, moves.mirror(jp, guard, side), high), side)
            punching = (age >= 0) & (hand == side)
            swing = moves.mirror(jp, moves.punch_pose(
                jp, move, aim, target, age, moves.mirror(jp, rest, side), self.dt) + residual, side)
            pose = jp.where(punching, swing, rest)
            arms.append(jp.concatenate([(pose - guard) / self._config.action_scale, jp.zeros(3)]))
        age = jp.where(age >= 0, age + speed, age)
        finished = moves.finished(jp, move, age, self.dt)
        wait = jp.where(age >= 0, info["punch_wait"], info["punch_wait"] - 1)
        wait = jp.where(finished, jax.random.randint(wait_rng, (), *wait_range), wait)
        info.update(punch_rng=rng, punch_hand=hand, punch_target=target, punch_move=move,
                    punch_residual=residual,
                    punch_aim=aim, punch_speed=speed, punch_age=jp.where(finished, -1.0, age),
                    punch_wait=wait.astype(jp.int32), high_guard=high, guard_timer=timer.astype(jp.int32))
        return jp.concatenate(arms)

    def _add_phantom_shins(self):
        """Adds a mocap pair of opponent shins that touch only this robot's shins and feet."""
        old = self._mj_model
        spec = mujoco.MjSpec.from_file(self._xml_path, assets=self._model_assets)
        body = spec.worldbody.add_body(name="phantom", mocap=True, pos=[0, 0, -10])
        for k, y in enumerate((-.1, .1)):
            body.add_geom(name=f"phantom_shin_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0, y, .08, 0, y, .45], size=[.045, 0, 0], contype=0, conaffinity=0,
                          rgba=[.6, .2, .2, .5])
            for part in ("left_shin", "right_shin", "left_foot", "right_foot"):
                spec.add_pair(geomname1=f"phantom_shin_{k}", geomname2=part, condim=3,
                              friction=[.6, .6, .005, .0001, .0001])
        model = spec.compile()
        model.opt.timestep = old.opt.timestep
        model.jnt_range[:] = old.jnt_range
        model.actuator_ctrlrange[:] = old.actuator_ctrlrange
        model.vis.global_.offwidth, model.vis.global_.offheight = old.vis.global_.offwidth, old.vis.global_.offheight
        if model.nq != old.nq or model.nbody != old.nbody + 1 or model.nu != old.nu:
            raise ValueError("phantom shins must not change the robot")
        self._mj_model = model
        self._mjx_model = mjx_env.put_model(model, impl=self._config.impl)

    def phantom(self, state):
        """Moves the phantom shins: idle, or sweeping in toward the pelvis and back out."""
        info = state.info
        keys = jax.random.split(info["phantom_rng"], 4)
        timer = info["phantom_timer"] - self.dt
        start = (info["phantom_age"] < 0) & (timer <= 0)
        # Mostly in front, where the opponent stands, but from any side.
        angle = jp.where(start, jax.random.normal(keys[0]) * .8, info["phantom_angle"])
        yaw = jp.where(start, jax.random.uniform(keys[1], minval=-.6, maxval=.6), info["phantom_yaw"])
        span = jp.where(start, jax.random.uniform(keys[2], minval=PHANTOM_SPAN[0], maxval=PHANTOM_SPAN[1]),
                        info["phantom_span"])
        closest = jp.where(start, jax.random.uniform(jax.random.fold_in(keys[2], 1), minval=PHANTOM_CLOSEST[0],
                                                     maxval=PHANTOM_CLOSEST[1]), info["phantom_closest"])
        age = jp.where(start, 0.0, info["phantom_age"])
        active = age >= 0
        u = jp.clip(age / span, 0., 1.)
        reach = PHANTOM_START - (PHANTOM_START - closest) * jp.sin(jp.pi * u)
        heading = jp.arctan2(state.data.site_xmat[self._pelvis_imu_site_id][1, 0],
                             state.data.site_xmat[self._pelvis_imu_site_id][0, 0])
        direction = heading + angle
        center = state.data.qpos[:2] + reach * jp.array([jp.cos(direction), jp.sin(direction)])
        position = jp.where(active, jp.concatenate([center, jp.zeros(1)]), PHANTOM_PARK)
        # Facing the robot, turned by up to +-0.6 rad.
        facing = direction + jp.pi + yaw
        quat = jp.array([jp.cos(facing / 2), 0., 0., jp.sin(facing / 2)])
        age = jp.where(active, age + self.dt, age)
        done = active & (age >= span)
        timer = jp.where(done, jax.random.uniform(keys[3], minval=PHANTOM_GAP[0], maxval=PHANTOM_GAP[1]),
                         timer)
        info.update(phantom_timer=timer, phantom_span=span, phantom_age=jp.where(done, -1.0, age),
                    phantom_angle=angle, phantom_yaw=yaw, phantom_closest=closest,
                    phantom_rng=jax.random.split(keys[3])[0])
        return state.replace(data=state.data.replace(mocap_pos=position[None], mocap_quat=quat[None]))

    def step(self, state, action):
        if self.stage == "spar" and self._phantom is not None:
            state = self.phantom(state)
        if self.stage == "spar":
            action = action.at[LEFT_ARM:].set(self.punch_arms(state.info))
        state = super().step(state, action)
        if self.stage == "approach":
            state = self._approach_command(state)
        state.metrics.update(self._metrics(state.data, state.info))
        if self.stage == "approach":
            state.metrics["goal_distance"] = jp.linalg.norm(
                state.info["goal"] - state.data.qpos[:2])
        return state

    def _approach_command(self, state):
        delta = state.info["goal"] - state.data.qpos[:2]
        rotation = state.data.site_xmat[self._pelvis_imu_site_id]
        local = rotation[:2, :2].T @ delta
        # Stop at boxing range. Only a requested velocity is changed: the
        # policy must produce the torques that move and balance the robot.
        distance = jp.linalg.norm(delta)
        speed = jp.clip((distance - 0.75) * 0.8, 0.0, 0.45)
        direction = local / jp.maximum(distance, 1e-6)
        state.info["command"] = jp.array([
            speed * direction[0], speed * direction[1],
            jp.clip(jp.arctan2(local[1], local[0]), -0.5, 0.5)])
        contact = jp.array([
            state.data.sensordata[self.mj_model.sensor_adr[sensor]] > 0
            for sensor in self._feet_floor_found_sensor])
        return state.replace(obs=self._get_obs(state.data, state.info, contact))

    def _get_termination(self, data):
        # The upstream task allows extreme tilt before termination. A standing
        # curriculum must end when the torso falls or the pelvis drops, rather
        # than letting a prone robot collect a zero-velocity reward.
        return (super()._get_termination(data)
                | (self.get_gravity(data, "torso")[-1] < 0.60)
                | (data.qpos[2] < 0.55))
