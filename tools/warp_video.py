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
import json
import pickle
from pathlib import Path

import imageio
import mujoco
import numpy as np

from boxer import scene
from boxer.warp_config import make_env
from boxer.warp_train import numpy_forward, warp_run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="runs/<warpN> or a step .pkl")
    ap.add_argument("--scripted", action="store_true",
                    help="render the assisted skill controller baseline, without a network")
    ap.add_argument("--output", type=Path, default=Path("runs/warp_video.mp4"))
    ap.add_argument("--length", type=int, default=500, help="control steps to render")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions the way training did, instead of the mode")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="also write a contact sheet of --sheet-tiles frames")
    ap.add_argument("--sheet-tiles", type=int, default=9)
    args = ap.parse_args(argv)
    if args.scripted and args.checkpoint:
        ap.error("choose a checkpoint or --scripted")
    if not args.scripted and not args.checkpoint:
        ap.error("give a checkpoint or --scripted")
    if args.length < 1 or args.sheet_tiles < 1:
        ap.error("length and sheet-tiles must be positive")
    if args.scripted:
        cfg = dict(environment="skills", spawn_distance=0.62, n_frames=5,
                   round_steps=600, damage_cap=6.0, stun_steps=10)
        trained_steps = 0
        label = "Scripted skill baseline | Balance assistance ON"
    else:
        _root, step, cfg = warp_run(str(args.checkpoint))
        with step.open("rb") as handle:
            params_p, _params_v, _opt, norm, trained_steps = pickle.load(handle)
        print(f"{step.name}: policy trained to {trained_steps:,} learner steps "
              f"({'sampled' if args.stochastic else 'deterministic'} actions)", flush=True)
        forward = numpy_forward(cfg, params_p, norm, args.stochastic, seed=args.seed)
        label = f"PPO self-play | {trained_steps:,} learner steps"
        if cfg.get("balance_assistance", False):
            label += " | Balance assistance ON"

    spawn = cfg["spawn_distance"]
    env = make_env(cfg, nworld=1)
    physics = getattr(env, "physics", env)
    physics.capture_render_state = True
    dt = env.n_frames * physics._model.opt.timestep
    visual = scene.build_model(spawn_distance=spawn, visual=True)
    # Distinguish the corners without changing collision surfaces or dynamics.
    for g in range(visual.ngeom):
        name = mujoco.mj_id2name(visual, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(visual, mujoco.mjtObj.mjOBJ_BODY,
                                 int(visual.geom_bodyid[g])) or ""
        if any(part in body for part in ("torso_link", "wrist_yaw_link", "pelvis")):
            if body.startswith(scene.RED):
                visual.geom_rgba[g] = [0.8, 0.12, 0.16, 1]
                visual.geom_matid[g] = -1
            elif body.startswith(scene.BLUE):
                visual.geom_rgba[g] = [0.12, 0.35, 0.85, 1]
                visual.geom_matid[g] = -1
        if name == scene.FLOOR:
            visual.geom_matid[g] = -1
            visual.geom_rgba[g] = [0.12, 0.17, 0.23, 1]
    if (visual.nq, visual.nv) != (env._model.nq, env._model.nv):
        raise SystemExit("the visual build changed the joint layout; the qpos "
                         "transfer below would be silently wrong")
    host = mujoco.MjData(visual)
    renderer = mujoco.Renderer(visual, height=args.height, width=args.width)

    obs = env.reset(seed=args.seed)
    # New observation entries are appended, so an older checkpoint reads the leading
    # slice of whatever this env now produces.
    ckpt_obs = cfg.get("obs_size", env.obs_size)
    camera = args.camera
    if camera == "overview":
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.75]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 105, -12
    pick_at = set(np.linspace(0, args.length - 1, args.sheet_tiles).astype(int).tolist())
    picks = []
    hits = np.zeros(2)
    counts = np.zeros(2, dtype=int)
    falls = kos = rounds = 0
    from PIL import Image, ImageDraw
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=round(1 / dt), macro_block_size=2)
    for t in range(args.length):
        if args.scripted:
            from boxer.skills_env import scripted_actions
            action = scripted_actions(env, t)
        else:
            action = forward(obs[..., :ckpt_obs].reshape(2, ckpt_obs)).reshape(
                1, 2, env.action_size)
        obs, reward, done, parts = env.step(action)
        hits += parts["scored_by"][0]
        counts += parts["scored_by"][0] > 0
        falls += int(parts["fell"][0])
        kos += int(parts["judged_out"][0])
        rounds += int(done[0])
        host.qpos[:] = physics.render_qpos[0]
        host.qvel[:] = physics.render_qvel[0]
        host.time = (t + 1) * dt
        mujoco.mj_forward(visual, host)
        renderer.update_scene(host, camera=camera)
        picture = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, args.width, 50), fill=(13, 18, 28))
        draw.text((14, 8), label, fill="white")
        draw.text((14, 29), f"RED  {counts[0]} clean hits   |   BLUE  {counts[1]} clean hits"
                  f"   |   {host.time:.1f}s   |   Round {rounds + 1}", fill="white")
        frame = np.asarray(picture)
        writer.append_data(frame)
        if t in pick_at:
            picks.append(np.asarray(frame).copy())

    writer.close()
    renderer.close()
    report = dict(controller="scripted" if args.scripted else "ppo",
                  balance_assistance=cfg.get("environment") == "skills",
                  trained_steps=trained_steps, seed=args.seed,
                  stochastic=args.stochastic, control_steps=args.length,
                  seconds=args.length * dt, clean_hits=counts.tolist(),
                  punch_value=hits.tolist(), falls=falls, knockouts=kos,
                  closed_rounds=rounds)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}: {report}", flush=True)

    if args.sheet:
        from PIL import Image
        tiles = args.sheet_tiles
        cols, rows = 3, int(np.ceil(tiles / 3))
        sheet = Image.new("RGB", (args.width * cols, args.height * rows))
        for i, frame in enumerate(picks[:tiles]):
            sheet.paste(Image.fromarray(np.asarray(frame)),
                        ((i % cols) * args.width, (i // cols) * args.height))
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.sheet)
        print(f"wrote {args.sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
