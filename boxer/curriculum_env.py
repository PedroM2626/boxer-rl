"""Unassisted G1 balance/locomotion, built on the pinned MuJoCo Playground task.

All 29 joints, including all 12 leg joints, are policy actions. No pelvis
springs, root tracking, gravity compensation or pose playback are used.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jp
import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick

from boxer.scene import ARM_JOINTS, GUARD_ARMS_BOXING
from boxer.curriculum_control import STAGES, TRAINABLE_STAGES

PLAYGROUND_REVISION = "12ad7f3842791ced39f27e5ac96d066bbe361653"
MENAGERIE_REVISION = "1b86ece576591213e2b666ebf59508454200ca97"

# Left-hand (shoulder pitch, roll, yaw, elbow) strike endpoints at body and head
# height: the FK-checked targets the skills controller already lands with. The
# right hand mirrors roll and yaw. Extension/retraction timing matches it too.
PUNCH_BODY = np.array([-1.18, .04, .10, 1.20])
PUNCH_HEAD = np.array([-1.94, .065, .20, 1.51])
PUNCH_EXTEND, PUNCH_RETRACT = .18, .26
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
        if self.stage == "spar":
            punch_rng = jax.random.fold_in(rng, 77)
            state.info.update(
                punch_rng=punch_rng, punch_age=jp.array(-1.0),
                punch_hand=jp.array(0, jp.int32), punch_target=jp.zeros(4),
                punch_speed=jp.array(1.0),
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
        """Arm actions for the generated punch; zero holds the guard (default pose).

        The policy's own arm outputs are replaced: it only keeps balance and
        footwork while the arms throw randomized straight punches at body/head
        height, with combos, pauses, speed and endpoint variation.
        """
        rng, hand_rng, aim_rng, jitter_rng, speed_rng, wait_rng = jax.random.split(
            info["punch_rng"], 6)
        launch = (info["punch_age"] < 0) & (info["punch_wait"] <= 0)
        aim = jax.random.uniform(aim_rng)
        target = (PUNCH_BODY + aim * (PUNCH_HEAD - PUNCH_BODY)
                  + jax.random.uniform(jitter_rng, (4,), minval=-.12, maxval=.12))
        hand = jp.where(launch, jax.random.randint(hand_rng, (), 0, 2), info["punch_hand"])
        target = jp.where(launch, target, info["punch_target"])
        speed = jp.where(launch, jax.random.uniform(speed_rng, minval=.8, maxval=1.25),
                         info["punch_speed"])
        age = jp.where(launch, 0.0, info["punch_age"])
        extend, retract = PUNCH_EXTEND / self.dt, PUNCH_RETRACT / self.dt
        fraction = jp.where(age < extend, (age + 1) / extend, 1 - (age - extend) / retract)
        fraction = jp.clip(fraction, 0, 1) * (age >= 0)
        sign = jp.where(hand == 0, 1.0, -1.0)
        start = jp.where(hand == 0, LEFT_ARM, RIGHT_ARM)
        guard = jax.lax.dynamic_slice(self._default_pose, (start,), (4,))
        offset = fraction * (target * jp.array([1, sign, sign, 1]) - guard)
        arms = jax.lax.dynamic_update_slice(
            jp.zeros(14), offset / self._config.action_scale, (start - LEFT_ARM,))
        age = jp.where(age >= 0, age + speed, age)
        finished = age >= extend + retract
        wait = jp.where(age >= 0, info["punch_wait"], info["punch_wait"] - 1)
        wait = jp.where(finished, jax.random.randint(wait_rng, (), 5, 60), wait)
        info.update(punch_rng=rng, punch_hand=hand, punch_target=target,
                    punch_speed=speed, punch_age=jp.where(finished, -1.0, age),
                    punch_wait=wait.astype(jp.int32))
        return arms

    def step(self, state, action):
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
