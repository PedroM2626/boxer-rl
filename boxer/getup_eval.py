"""Measures how often a policy gets up from the floor in the ``getup`` stage.

Each world starts lying down (on the back, front or a side), part-way through
a fall, or standing (``curriculum_env._fallen_start``), and runs the policy
deterministically for ``--seconds``. A robot counts as up when its pelvis is
at least ``UP_HEIGHT`` high and its torso within ~25 degrees of upright.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.curriculum_env import UnassistedG1, stage_config
from boxer.curriculum_train import inference, jax_compat, load_policy_params

UP_HEIGHT, UP_GRAVITY = .72, .9


def summarize(heights, upright, kinds, dt):
    """heights/upright: (steps, worlds); kinds: 0 standing, 1 falling, 2 lying."""
    up = (heights >= UP_HEIGHT) & (upright >= UP_GRAVITY)
    first = np.where(up.any(0), up.argmax(0), -1)
    result = {}
    for k, name in enumerate(("standing", "falling", "lying")):
        sel = kinds == k
        if not sel.any():
            continue
        got = first[sel] >= 0
        result[name] = dict(
            worlds=int(sel.sum()), up_at_end=float(up[-1, sel].mean()), ever_up=float(got.mean()),
            seconds_to_up=float(first[sel][got].mean() * dt) if got.any() else None,
            up_share_of_time=float(up[:, sel].mean()),
            peak_height=float(heights[:, sel].max(0).mean()), end_height=float(heights[-1, sel].mean()),
            peak_upright=float(upright[:, sel].max(0).mean()))
    return result


def evaluate(env, policy, worlds, steps, seed):
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(policy))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), worlds))
    lying = np.asarray(state.metrics["getup_start_lying"]) > 0
    standing = np.asarray(state.data.qpos[:, 2]) > .7
    kinds = np.where(lying, 2, np.where(standing, 0, 1))
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), worlds)
    heights, upright, qpos = [], [], []
    for _ in range(steps):
        action, _ = act(state.obs, keys)
        state = step(state, action)
        heights.append(np.asarray(state.data.qpos[:, 2]))
        upright.append(np.asarray(state.metrics["upright"]))
        qpos.append(np.asarray(state.data.qpos[kinds == 2][:4]))
    return np.array(heights), np.array(upright), kinds, np.array(qpos)


def render(env, qpos, path, every=2):
    """qpos: (steps, robots, nq) -> one clip per robot, back to back, at real time."""
    import mediapy
    import mujoco
    try:  # WSL often has no system ffmpeg; imageio-ffmpeg ships one
        import imageio_ffmpeg
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    model = env.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.6, -15., 135.
    frames = []
    for robot in range(qpos.shape[1]):
        for t in range(0, len(qpos), every):
            data.qpos[:] = qpos[t, robot]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .45]
            renderer.update_scene(data, camera)
            frames.append(renderer.render())
    mediapy.write_video(str(path), frames, fps=round(1 / (every * env.dt)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assist", type=float, default=0.,
                        help="the training aid's upward torso force, as a fraction of body weight (default: none)")
    parser.add_argument("--video", type=Path, help="render the first lying starts one after another")
    args = parser.parse_args(argv)
    jax_compat()
    config = stage_config("getup")
    config.naconmax = max(8192, args.worlds * 32)
    config.getup_assist = args.assist
    env = UnassistedG1("getup", config=config)
    params, manifest = load_policy_params(args.checkpoint)
    steps = int(round(args.seconds / env.dt))
    heights, upright, kinds, rollout = evaluate(env, inference(env, params), args.worlds, steps, args.seed)
    result = dict(checkpoint=str(args.checkpoint), stage=manifest["stage"], stage_steps=manifest["stage_steps"],
                  seconds=args.seconds, worlds=args.worlds, seed=args.seed, assist=args.assist,
                  up_height=UP_HEIGHT, up_gravity=UP_GRAVITY, **summarize(heights, upright, kinds, env.dt))
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    if args.video:
        render(env, rollout, args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
