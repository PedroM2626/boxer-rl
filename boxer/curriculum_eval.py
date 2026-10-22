"""Deterministic physical gates for standing, command tracking and approach.

Each trial ends at its first fall; resets cannot inflate survival. The reported
score is separate from training reward. Optional videos replay actual qpos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1, stage_config
from boxer.curriculum_train import contract, inference, load_policy_params, utc
from boxer.curriculum_control import STAGES, COMMANDS, commands_for, passes_gate


def evaluate(checkpoint, stage=None, worlds=32, seeds=(7, 23), video=None, preview=False):
    params, metadata = load_policy_params(checkpoint)
    stage = stage or metadata["stage"]
    config = stage_config(stage)
    config.noise_config.level = 0.0
    # Training disturbances must never leak into the fixed evaluation protocol.
    config.push_config.enable = False
    config.njmax = 90  # Preserve the original benchmark allocation.
    env = UnassistedG1(stage, config=config)
    if contract(env) != metadata["contract"]:
        raise ValueError("evaluation model differs from the checkpoint contract")
    policy = jax.jit(inference(env, params))
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    def set_command(state, command):
        state.info["command"] = command
        contact = jp.array([
            state.data.sensordata[env.mj_model.sensor_adr[sensor]] > 0
            for sensor in env._feet_floor_found_sensor])
        return state.replace(obs=env._get_obs(state.data, state.info, contact))

    command_fn = jax.jit(jax.vmap(set_command))
    @jax.jit
    def retain_finished(previous, current, active):
        def choose(before, after):
            # MJX Warp also carries shared/empty solver buffers without a
            # world axis; these are scratch space, not per-robot state.
            if not after.ndim or after.shape[0] != worlds:
                return after
            return jp.where(active.reshape((worlds,) + (1,) * (after.ndim - 1)),
                            after, before)
        return jax.tree.map(choose, previous, current)
    trials, frames = [], []
    commands = commands_for(stage)
    # Walking gate videos should show actual commanded footwork, rather than
    # always capturing the first (stationary) trial. Trial order is unchanged.
    video_command = commands[1] if stage in ("walk_easy", "walk") else commands[0]
    # Balance includes a velocity impulse: this is an evaluation disturbance,
    # not a stabilizing root controller or a training assistance force.
    disturbances = [False, True] if stage == "balance" else [False]
    if preview:
        seeds = (seeds[0],)
        commands = [commands[1]] if stage in ("walk_easy", "walk") else [COMMANDS[0]]
        disturbances = [False]
    for seed in seeds:
        for command in commands:
            for disturbed in disturbances:
                keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
                state = reset(keys)
                start_position = np.asarray(state.data.qpos[:, :2])
                rotation = np.asarray(state.data.site_xmat[:, env._pelvis_imu_site_id, :2, :2])
                alive = np.ones(worlds, dtype=bool)
                duration = np.zeros(worlds)
                linear, yaw, count = 0.0, 0.0, 0
                displacement = np.zeros((worlds, 2))
                distance = np.full(worlds, 2.0)
                capture = (video is not None and not frames and seed == seeds[0]
                           and not disturbed and command == video_command)
                for t in range(500):
                    if stage != "approach":
                        cmd = jp.broadcast_to(jp.array(command), (worlds, 3))
                        state = command_fn(state, cmd)
                    if disturbed and t == 200:
                        velocity = state.data.qvel.at[:, :2].add(jp.array([0.2, -0.2]))
                        state = state.replace(data=state.data.replace(qvel=velocity))
                    action, _ = policy(state.obs, jax.random.PRNGKey(t))
                    previous = state
                    state = retain_finished(previous, step(state, action), jp.array(alive))
                    done = np.asarray(state.done).astype(bool)
                    duration += alive * env.dt
                    valid = alive & ~done
                    local_delta = np.einsum("nji,nj->ni", rotation,
                        np.asarray(state.data.qpos[:, :2]) - start_position)
                    displacement[valid] = local_delta[valid]
                    if stage == "approach":
                        distance[valid] = np.asarray(state.metrics["goal_distance"])[valid]
                    if t >= 50:
                        linear += float(np.sum(np.asarray(state.metrics["linear_tracking_error"])[valid]))
                        yaw += float(np.sum(np.asarray(state.metrics["yaw_tracking_error"])[valid]))
                        count += int(valid.sum())
                    if capture and alive[0] and t % 2 == 0:
                        frames.append(np.asarray(state.data.qpos[0]).copy())
                    alive &= ~done
                    if not alive.any():
                        break
                axis = np.array(command[:2])
                speed = np.linalg.norm(axis)
                directed = displacement @ (axis / speed) if speed else np.zeros(worlds)
                directed[~alive] = 0.0
                trial = dict(seed=seed, command=list(command), disturbed=disturbed,
                    survival_rate=float(alive.mean()), mean_survival_seconds=float(duration.mean()),
                    linear_tracking_error=linear / max(count, 1),
                    yaw_tracking_error=yaw / max(count, 1),
                    directed_distance=float(directed.mean()),
                    required_distance=float(speed * 10.0 * 0.45),
                    arrival_rate=float(np.mean(alive & (distance <= 0.95))))
                trials.append(trial)
                print(json.dumps(trial), flush=True)
    if video and frames:
        render(env, frames, video, dict(metadata, video_command=list(video_command)))
    return dict(stage=stage, stage_steps=metadata["stage_steps"],
                checkpoint_sha256=metadata["sha256"], updated_utc=utc(),
                assistance=False, episode_seconds=10, worlds_per_trial=worlds,
                kind="development_preview" if preview else "promotion_gate",
                passed=None if preview else passes_gate(stage, trials), trials=trials)


def render(env, frames, output, metadata):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = env.mj_model
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.8, 130, -18
    camera.lookat[:] = [0, 0, 0.7]
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(output, fps=25) as writer:
            for qpos in frames:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                camera.lookat[:2] = qpos[:2]
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 44), fill="black")
                draw.text((10, 6), f"{metadata['stage'].title()} | {metadata['stage_steps']:,} transitions", fill="white")
                command_label = ("Goal navigation" if metadata["stage"] == "approach"
                                 else "Command " + str(metadata.get("video_command", "recorded")))
                draw.text((10, 25), "Free legs | No pelvis support | " + command_label, fill="white")
                writer.append_data(np.asarray(image))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--preview", action="store_true",
                        help="one development trial; never a promotion gate")
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, args.stage, args.worlds, video=args.video, preview=args.preview)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
    main()
