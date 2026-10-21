"""Render a warp-trained policy from the engine it was trained in.

The MJX rollout path (``boxer/rollout.py``) was the only renderer this project
had, and it lied: the final warp1 checkpoint reads as upright and idle in the
warp log (reward +0.89/step) and as "both boxers collapsed in two seconds" when
the same weights drive the MJX env, because the guard-error block of the
observation comes from SITE frames there and from sensors in the warp env. The
deliverable of this project is a video of two robots boxing, so the video has to
come from the physics the policy actually trained against.

Physics: ``WarpBoxing`` with one world, exactly the training build (visual geoms
discarded). Pictures: a parallel visual build of the same scene -- the two
models share ``nq``/``nv``, so ``qpos``/``qvel`` transfer verbatim and
``mj_forward`` on the visual model puts the meshes where the simulation says
they are. The policy forward is the trainer's numpy port (``boxer.warp_train
.numpy_forward``) over the checkpoint's leaves: no jax computation runs here, and
the warp venv ships a CPU-only jaxlib, so the CUDA card stays warp's alone either
way.

    MUJOCO_GL=egl PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python \
        tools/warp_video.py runs/warp1 --length 400 --output fights/warp1_final.mp4
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import imageio
import mujoco
import numpy as np

from boxer import scene
from boxer.warp_env import WarpBoxing
from boxer.warp_train import numpy_forward, warp_run

CONTROL_DT = 0.02  # 5 physics substeps of 4 ms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("checkpoint", help="runs/<warpN>, its checkpoint dir, or a step .pkl")
    ap.add_argument("--output", type=Path, default=Path("runs/warp_video.mp4"))
    ap.add_argument("--length", type=int, default=500, help="control steps to render")
    ap.add_argument("--camera", default="red/track")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions the way training did, instead of the mode")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="also write a contact sheet of --sheet-tiles frames")
    ap.add_argument("--sheet-tiles", type=int, default=9)
    args = ap.parse_args(argv)

    _root, step, cfg = warp_run(str(args.checkpoint))
    with step.open("rb") as handle:
        params_p, _params_v, _opt, norm, trained_steps = pickle.load(handle)
    print(f"{step.name}: policy trained to {trained_steps:,} learner steps "
          f"({ 'sampled' if args.stochastic else 'deterministic'} actions)", flush=True)
    forward = numpy_forward(cfg, params_p, norm, args.stochastic)

    spawn = cfg["spawn_distance"]
    env = WarpBoxing(nworld=1, n_frames=cfg["n_frames"], spawn_distance=spawn,
                     action_scale=cfg["action_scale"],
                     round_steps=cfg["round_steps"])
    visual = scene.build_model(spawn_distance=spawn, visual=True)
    if (visual.nq, visual.nv) != (env._model.nq, env._model.nv):
        raise SystemExit("the visual build changed the joint layout; the qpos "
                         "transfer below would be silently wrong")
    host = mujoco.MjData(visual)
    renderer = mujoco.Renderer(visual, height=args.height, width=args.width)

    obs = env.reset(seed=args.seed)
    # New observation entries are appended, so an older checkpoint reads the leading
    # slice of whatever this env now produces.
    ckpt_obs = cfg["obs_size"]
    pick_at = set(np.linspace(0, args.length - 1, args.sheet_tiles).astype(int).tolist())
    frames, picks = [], []
    hits = 0.0
    for t in range(args.length):
        action = forward(obs[..., :ckpt_obs].reshape(2, ckpt_obs)).reshape(
            1, 2, env.action_size)
        obs, reward, done, parts = env.step(action)
        hits += float(parts["scored"][0])
        host.qpos[:] = env._qpos_sync[0]
        host.qvel[:] = env._qvel_sync[0]
        host.time = (t + 1) * CONTROL_DT
        mujoco.mj_forward(visual, host)
        renderer.update_scene(host, camera=args.camera)
        frame = renderer.render()
        frames.append(frame)
        if t in pick_at:
            picks.append(np.asarray(frame).copy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, [np.asarray(f) for f in frames], fps=int(1 / CONTROL_DT))
    print(f"wrote {args.output}: {len(frames)} frames at {1 / CONTROL_DT:.0f} fps "
          f"({len(frames) * CONTROL_DT:.1f} s), red scored {hits:.1f} hit-value "
          f"over the round", flush=True)

    if args.sheet:
        from PIL import Image
        tiles = args.sheet_tiles
        cols, rows = 3, int(np.ceil(tiles / 3))
        sheet = Image.new("RGB", (args.width * cols, args.height * rows))
        for i, frame in enumerate(picks[:tiles]):
            sheet.paste(Image.fromarray(np.asarray(frame)),
                        ((i % cols) * args.width, (i // cols) * args.height))
        sheet.save(args.sheet)
        print(f"wrote {args.sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
