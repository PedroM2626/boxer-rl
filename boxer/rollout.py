"""Renders a fight so it can actually be watched.

A green training curve says nothing about whether the boxers box, so this rolls
one round out in the full visual scene and writes it as an interactive Brax
viewer, a video, or a contact sheet of still frames.

By default both corners run the given checkpoint, which is the self-play view.
Pass ``--opponent`` to render the learner against ``passive``, ``jabber`` or a
different checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jp
import numpy as np
from brax.io import html
from PIL import Image

from boxer import policy as policy_module
from boxer.env import BoxingEnv


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path, help="directory written by boxer.train")
    parser.add_argument("--opponent", default=None,
                        help="'passive', 'jabber', or another checkpoint dir")
    parser.add_argument("--learner-side", choices=("red", "blue"), default="red")
    parser.add_argument("--output", type=Path, default=Path("runs/rollout"))
    parser.add_argument("--format", choices=("html", "mp4", "sheet"), default="mp4")
    parser.add_argument("--length", type=int, default=500, help="control steps to roll out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", default=None, help="named camera in the scene")
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample from the policy instead of taking its mode")
    parser.add_argument("--sheet-tiles", type=int, default=4)
    return parser.parse_args(argv)


def _corner_action(one: policy_module.Policy, obs_row, clock, key) -> jax.Array:
    """Runs one policy on a single corner, restoring the batch axis it expects."""
    return one(obs_row[None], clock, key)[0]


def collect(env: BoxingEnv, red, blue, length: int, seed: int):
    """Rolls one round out and returns the physics trajectory plus final metrics."""

    @jax.jit
    def step(state, key):
        key_red, key_blue, next_key = jax.random.split(key, 3)
        clock = jp.broadcast_to(state.pipeline_state.time, (1,))
        action = jp.stack([
            _corner_action(red, state.obs[0], clock, key_red),
            _corner_action(blue, state.obs[1], clock, key_blue),
        ])
        return env.step(state, action), next_key

    state = env.reset(jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    trajectory = [state.pipeline_state]
    for _ in range(length):
        state, key = step(state, key)
        trajectory.append(state.pipeline_state)
    metrics = {name: float(np.asarray(value)) for name, value in state.metrics.items()}
    return trajectory, metrics


def _contact_sheet(frames, tiles: int) -> Image.Image:
    rows = int(np.ceil(len(frames) / tiles))
    width, height = frames[0].shape[1], frames[0].shape[0]
    sheet = Image.new("RGB", (width * tiles, height * rows))
    for index, frame in enumerate(frames):
        sheet.paste(Image.fromarray(np.asarray(frame)),
                    ((index % tiles) * width, (index // tiles) * height))
    return sheet


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    env = BoxingEnv(n_frames=args.n_frames, visual=True,
                    spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    checkpoint = policy_module.resolve_checkpoint(args.checkpoint)
    print(f"devices: {jax.devices()}", flush=True)
    print(f"loading {checkpoint}", flush=True)
    learner = policy_module.load_policy(checkpoint, deterministic=not args.stochastic)
    opponent = (
        learner if args.opponent is None
        else policy_module.make_opponent(args.opponent, env)
    )
    red, blue = (learner, opponent) if args.learner_side == "red" else (opponent, learner)

    trajectory, metrics = collect(env, red, blue, args.length, args.seed)
    print(f"final metrics: {metrics}", flush=True)

    frames = env.render(trajectory, height=args.height, width=args.width, camera=args.camera)
    frames = [np.asarray(frame) for frame in frames]

    if args.format == "html":
        path = args.output / "rollout.html"
        html.save(str(path), env.sys, trajectory)
    elif args.format == "mp4":
        path = args.output / "rollout.mp4"
        imageio.mimsave(path, frames, fps=int(round(1.0 / float(env.dt))))
    else:
        path = args.output / "rollout.png"
        picks = np.linspace(0, len(frames) - 1, args.sheet_tiles ** 2).astype(int)
        _contact_sheet([frames[i] for i in picks], args.sheet_tiles).save(path)

    print(f"wrote {path} ({len(frames)} frames at {1.0 / float(env.dt):.0f} fps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
