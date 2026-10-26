"""Render every final fresh self-play trial from reviewed states, without new physics."""
import argparse,json,os
from pathlib import Path
from contextlib import ExitStack
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import build_pair
from tools.striking_motion_cycle_supervisor import sha

ROOT=Path('docs/results/striking-pilot/motion-cycle-v14')
SOURCE=ROOT/'gates/fresh-final'
VIDEO=Path('fights/striking-pilot/motion-cycle-v14-final-fresh-selfplay.mp4')
MAX_FRAMES=1565


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    assert os.environ['JAX_PLATFORMS']=='cpu'
    with ExitStack() as stack:
        for p in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock'),Path('runs/unassisted_boxing.lock')):
            h=stack.enter_context(p.open('a'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'state.json').exists() or VIDEO.exists():raise ValueError('existing render needs evidence/accounting review')
        review=json.loads((ROOT/'completed-review.json').read_text());report=json.loads((SOURCE/'selfplay.json').read_text())
        assert review['status']=='independently_reviewed_completed_allocation' and review['formal_world_controls_total']==2140974
        assert review['checkpoint_final_sha256']==report['striking_sha256']
        assert len(report['trials'])==8 and {(r['seed'],r['swap']) for r in report['trials']}=={(s,w) for s in range(33000003,33000007) for w in (False,True)}
        final=next(g for g in review['gates'] if g['label']=='fresh-final');assert final['valid_cycles']==0
        clips=[]
        for row in report['trials']:
            path=SOURCE/'selfplay/full'/Path(row['trace_file']).name;assert sha(path)==row['trace_sha256']
            with np.load(path) as a:
                assert all(np.isfinite(a[k]).all() for k in a.files)
                indices=np.arange(1,len(a['qpos']),2)
                if not len(indices) or indices[-1]!=len(a['qpos'])-1:indices=np.r_[indices,len(a['qpos'])-1]
                clips.append(dict(seed=row['seed'],swap=row['swap'],survived=row['survived'],
                    source_sha256=row['trace_sha256'],qpos=a['qpos'][indices].copy(),qvel=a['qvel'][indices].copy(),
                    times=a['time_after'][indices].copy(),legacy=a['scored'].sum(axis=1).cumsum()[indices],indices=indices.tolist()))
        assert sum(len(c['times']) for c in clips)==MAX_FRAMES
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),source_report_sha256=sha(SOURCE/'selfplay.json'),
            source_review_sha256=sha(ROOT/'completed-review.json'),checkpoint_sha256=report['striking_sha256'],
            new_physical_controls=0,new_optimizer_updates=0,new_diagnostic_cloned_queries=0,
            physics_stepped=False,learned_checkpoint=True,closed_fist_visuals=True,
            render_frame_forwards=0,maximum_render_forwards=MAX_FRAMES,prior_render_forwards=500,
            fps=25,selection='All eight fresh-final self-play trials in archived order, including both first-fall episodes.',
            formal_world_controls_retained=2140974,diagnostic_ppo_retained=3212544,cloned_queries_retained=6349,
            video=str(VIDEO))
        atomic_json(root/'state.json',state)
        try:
            foundation=UnassistedG1('approach',impl='jax');model=build_pair(foundation,visual_fists=True)
            assert model.nq==clips[0]['qpos'].shape[1] and model.neq==0 and not np.any(model.body_gravcomp)
            data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=4.;camera.azimuth=100;camera.elevation=-18;camera.lookat[:]=[0,0,.75]
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
            VIDEO.parent.mkdir(parents=True,exist_ok=True)
            with mujoco.Renderer(model,height=480,width=640) as renderer,imageio.get_writer(VIDEO,fps=25,codec='libx264',quality=8) as writer:
                for clip_index,clip in enumerate(clips):
                    for i,(qpos,qvel,time,legacy) in enumerate(zip(clip['qpos'],clip['qvel'],clip['times'],clip['legacy'])):
                        state['render_frame_forwards']+=1;assert state['render_frame_forwards']<=MAX_FRAMES
                        atomic_json(root/'progress.json',dict(render_frame_forwards=state['render_frame_forwards'],clip_index=clip_index,new_physical_controls=0))
                        data.qpos[:]=qpos;data.qvel[:]=qvel;data.time=time;mujoco.mj_forward(model,data)
                        renderer.update_scene(data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                        draw.rectangle((0,0,640,79),fill='black')
                        lines=['Motion-cycle PPO v14 | Final learned checkpoint',
                            f'Fresh self-play | Seed {clip["seed"]} | Swap {int(clip["swap"])} | Trial {clip_index+1}/8',
                            f'Valid cycles: 0 | Contact events: {int(legacy)} | Result: {"survived" if clip["survived"] else "fall"}',
                            f'Recorded simulation {time:.2f} s | Free bodies | No new training']
                        for y,line in enumerate(lines):draw.text((8,3+19*y),line,fill='white',font=font)
                        writer.append_data(np.asarray(frame))
                        if (clip_index==0 and i==125) or (clip_index==5 and i==len(clip['times'])-1):frame.save(root/f'clip-{clip_index}-preview.png')
            count=0
            with imageio.get_reader(VIDEO) as reader:
                assert abs(reader.get_meta_data()['fps']-25)<1e-9
                for frame in reader:
                    assert frame.shape==(480,640,3);count+=1
            assert count==state['render_frame_forwards']==MAX_FRAMES
            state.update(status='completed',finished_utc=utc(),decoded_frames=count,video_sha256=sha(VIDEO),
                duration_seconds=count/25,cumulative_render_forwards=500+count,
                clips=[{k:v for k,v in c.items() if k not in ('qpos','qvel','times','legacy')} for c in clips],
                timing_note='Terminal frames retained; fixed-fps duration can differ by less than one frame per truncated episode.')
            atomic_json(root/'state.json',state);atomic_json(ROOT/'final-video.json',state)
            print(json.dumps(dict(status='completed',video=str(VIDEO),frames=count,seconds=count/25,new_physical_controls=0)))
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(root/'state.json',state);raise


if __name__=='__main__':main()
