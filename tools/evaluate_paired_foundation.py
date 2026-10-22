"""Validate native two-body policy transfer before authorizing boxing PPO.

This is diagnostic simulation, separately counted from foundation optimization.
Commands request navigation only; no scripted strike or root controller is used.
"""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, inference, load_policy_params, utc
from boxer.paired_foundation import PairedFoundation


def evaluate(checkpoint, trials=2, seeds=(7, 23, 31, 47), seconds=10., video=None):
    params, metadata = load_policy_params(checkpoint)
    foundation = UnassistedG1('approach', impl='jax')
    if contract(foundation) != metadata['contract']:
        raise ValueError('foundation source contract mismatch')
    policy = jax.jit(inference(foundation, params))
    env = PairedFoundation(foundation)
    results, frames = [], []
    simulated_transitions = 0
    for mode in ('stand', 'approach'):
        for seed in seeds:
            for swap in (False, True):
                for trial in range(trials):
                    env.reset(seed=seed + 1009 * trial, swap=swap)
                    start = np.array([env.data.qpos[m['root']:m['root']+2].copy() for m in env.maps])
                    total_impulse = np.zeros(2)
                    errors = []
                    capture = video and mode == 'approach' and seed == seeds[0] and not swap and trial == 0
                    for step in range(round(seconds/env.dt)):
                        commands = env.approach_commands() if mode == 'approach' else np.zeros((2, 3))
                        observation = env.observation(commands)
                        # Privileged values belong to the training critic. The
                        # frozen actor reads only the faithfully mapped state.
                        action, _ = policy({'state': jp.array(observation),
                                            'privileged_state': jp.zeros((2, 216))},
                                           jax.random.PRNGKey(0))
                        outcome = env.step(np.asarray(action))
                        simulated_transitions += 1
                        total_impulse += outcome['legal_contact_impulse']
                        errors.append([float(np.linalg.norm(observation[i, :2] - commands[i, :2])) for i in range(2)])
                        if capture and step % 2 == 0:
                            frames.append(env.data.qpos.copy())
                        if outcome['done']:
                            break
                    end = np.array([env.data.qpos[m['root']:m['root']+2].copy() for m in env.maps])
                    result = dict(mode=mode, seed=seed, swap=swap, trial=trial,
                        seconds=env.elapsed, survived=not env.done, fallen=env.fallen().tolist(),
                        finite=bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()),
                        initial_separation=float(np.linalg.norm(start[1]-start[0])),
                        final_separation=float(np.linalg.norm(end[1]-end[0])),
                        root_displacements=np.linalg.norm(end-start, axis=1).tolist(),
                        mean_linear_errors=np.mean(errors,axis=0).tolist(),
                        raw_legal_surface_impulses=total_impulse.tolist())
                    results.append(result)
                    print(json.dumps(result), flush=True)
    if frames:
        render(env, frames, video)
    return dict(kind='paired_transfer_diagnostic', updated_utc=utc(),
        foundation_sha256=metadata['sha256'], foundation_main_transitions=metadata['total_steps'],
        assistance=False, simulated_world_transitions=simulated_transitions,
        optimization_transitions=0, physics_dt=env.model.opt.timestep, control_dt=env.dt,
        free_roots=2, equality_constraints=env.model.neq, gravity_compensation=False,
        actor_inputs=103, per_corner_actions=29, contact_pairs=env.model.npair,
        punch_training=False, raw_contact_impulses_are_not_punch_scores=True,
        trials=results)


def render(env, frames, path):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=mujoco.MjData(env.model)
    camera=mujoco.MjvCamera();camera.distance=4.;camera.azimuth=100;camera.elevation=-18
    camera.lookat[:]=[0,0,.75]
    with mujoco.Renderer(env.model,height=480,width=640) as renderer:
        with imageio.get_writer(path,fps=25) as writer:
            for qpos in frames:
                data.qpos[:]=qpos;mujoco.mj_forward(env.model,data)
                renderer.update_scene(data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,640,40),fill='black')
                draw.text((8,8),'Paired transfer diagnostic | Free bodies | Navigation only',fill='white')
                writer.append_data(np.asarray(image))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('checkpoints/curriculum/approach'))
    parser.add_argument('--output',type=Path,default=Path('runs/paired-transfer/evaluation.json'))
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,31,47])
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    if args.trials < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('positive trials and finite positive duration required')
    state_path = args.output.parent / 'state.json'
    state = dict(status='running', pid=os.getpid(), started_utc=utc(),
                 kind='paired_transfer_diagnostic', optimization_transitions=0,
                 output=str(args.output), diagnostic_compute_separate=True)
    atomic_json(state_path,state)
    try:
        result=evaluate(args.checkpoint,args.trials,args.seeds,args.seconds,args.video)
        atomic_json(args.output,result)
        state.update(status='completed', updated_utc=utc(),
                     simulated_world_transitions=result['simulated_world_transitions'])
        atomic_json(state_path,state)
    except BaseException as error:
        state.update(status='failed',updated_utc=utc(),error=str(error))
        atomic_json(state_path,state)
        raise


if __name__=='__main__':
    main()
