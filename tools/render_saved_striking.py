"""Render two predeclared saved trajectories without stepping physical simulation."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MUJOCO_GL','egl')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU saved-state rendering required')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import build_pair
from tools.striking_terminal_dwell import sha

SOURCE=Path('docs/results/striking-pilot/terminal-dwell-v3')
VIDEO=Path('fights/striking-pilot/terminal-dwell-v3-recorded.mp4')


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'state.json').exists() or VIDEO.exists():raise ValueError('existing render requires evidence review before recovery')
        source=json.loads((SOURCE/'audit.json').read_text());review=json.loads((SOURCE/'analysis.json').read_text())
        assert source['phase']=='independently_reviewed' and review['status']=='reviewed'
        assert review['formal_world_controls_total']==2051072 and review['candidate_events']==0
        manifest=json.loads(Path('checkpoints/curriculum/approach/latest.json').read_text())
        assert sha(Path('checkpoints/curriculum/approach')/manifest['file'])==manifest['sha256']==source['motor_sha256']
        clips=[]
        for corner in (0,1):
            row=next(r for r in source['trials'] if (r['corner'],r['seed'],r['swap'])==(corner,7,False))
            file=SOURCE/'full'/Path(row['full_trace_file']).name;assert sha(file)==row['full_trace_sha256']
            with np.load(file) as a:
                assert len(a['qpos'])==500 and row['survived'] and not a['fallen'].any()
                np.testing.assert_array_equal(a['scored'],0.)
                indices=np.arange(0,500,2);times=a['time_after'][indices].copy()
                np.testing.assert_allclose(np.diff(times),.04,atol=1e-12,rtol=0)
                clips.append(dict(corner=corner,case=row['case'],source_full_sha256=sha(file),
                    qpos=a['qpos'][indices].copy(),qvel=a['qvel'][indices].copy(),times=times,indices=indices.tolist()))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),source_audit_sha256=sha(SOURCE/'audit.json'),
            source_review_sha256=sha(SOURCE/'analysis.json'),video=str(VIDEO),new_physical_controls=0,
            new_optimizer_updates=0,new_diagnostic_cloned_queries=0,render_frame_forwards=0,decoded_frames=0,
            expected_frames=500,fps=25,scripted_diagnostic=True,closed_fist_visuals=True,physics_stepped=False,
            formal_world_controls_retained=2051072,diagnostic_cloned_queries_retained=6253,
            selection='Seed 7, no swap, both acting corners, in corner order; chosen before rendering.')
        atomic_json(root/'state.json',state)
        try:
            foundation=UnassistedG1('approach',impl='jax');model=build_pair(foundation,visual_fists=True)
            assert model.nq==clips[0]['qpos'].shape[1] and model.neq==0 and not np.any(model.body_gravcomp)
            data=mujoco.MjData(model);camera=mujoco.MjvCamera()
            camera.distance=4.;camera.azimuth=100;camera.elevation=-18;camera.lookat[:]=[0,0,.75]
            VIDEO.parent.mkdir(parents=True,exist_ok=True)
            with mujoco.Renderer(model,height=480,width=640) as renderer,imageio.get_writer(VIDEO,fps=25,codec='libx264',quality=8) as writer:
                for clip in clips:
                    for index,(qpos,qvel,time) in enumerate(zip(clip['qpos'],clip['qvel'],clip['times'])):
                        state['render_frame_forwards']+=1
                        atomic_json(root/'progress.json',dict(render_frame_forwards=state['render_frame_forwards'],new_physical_controls=0,case=clip['case']))
                        data.qpos[:]=qpos;data.qvel[:]=qvel;data.time=time;mujoco.mj_forward(model,data)
                        renderer.update_scene(data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                        draw.rectangle((0,0,640,57),fill='black')
                        draw.text((8,5),'Terminal dwell v3 | Recorded states | Scripted diagnostic',fill='white')
                        actor='Red' if clip['corner']==0 else 'Blue'
                        draw.text((8,22),f'{actor} extension attempts | Valid impacts: 0 | Free bodies',fill='white')
                        draw.text((8,39),f'Simulation {time:.2f} s | Seed 7 | No swap | Original speed',fill='white')
                        writer.append_data(np.asarray(frame))
                        if index==125:frame.save(root/f'corner-{clip["corner"]}-preview.png')
            assert state['render_frame_forwards']==500
            with imageio.get_reader(VIDEO) as reader:
                metadata=reader.get_meta_data();assert abs(metadata['fps']-25)<1e-9
                for frame in reader:
                    assert frame.shape==(480,640,3);state['decoded_frames']+=1
            assert state['decoded_frames']==500
            state.update(status='completed',updated_utc=utc(),video_sha256=sha(VIDEO),duration_seconds=20.,
                clips=[{k:v for k,v in c.items() if k not in ('qpos','qvel','times')} for c in clips])
            atomic_json(root/'state.json',state)
            atomic_json(SOURCE/'video.json',state)
            print(json.dumps(dict(status='completed',video=str(VIDEO),decoded_frames=500,new_physical_controls=0)))
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(root/'state.json',state);raise


if __name__=='__main__':main()
