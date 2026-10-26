"""Read-only render of two preselected verified stochastic windows alongside training."""
import argparse,json,os
from pathlib import Path
from contextlib import ExitStack
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import build_pair
from boxer.curriculum_train import utc
from tools.striking_motion_cycle_supervisor import sha

ROOT=Path('docs/results/striking-pilot/contact-progress-v15')
VIDEO=Path('fights/striking-pilot/contact-progress-v15-verified-training-cycles.mp4')


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--side-view',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    video=VIDEO.with_name(VIDEO.stem+'-side-view.mp4') if args.side_view else VIDEO
    prior=2131 if args.side_view else 2065
    assert os.environ['JAX_PLATFORMS']=='cpu'
    with ExitStack() as stack:
        # Separate data/model, no mj_step/actor execution: do not interrupt the live learner.
        for path in (out/'.render.lock',Path('runs/striking_saved_event_render.lock')):
            h=stack.enter_context(path.open('a'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if video.exists() or (out/'state.json').exists():raise ValueError('existing render requires evidence/accounting review')
        source=ROOT/'first-captured-event-review.json';review=json.loads(source.read_text())
        assert review['status']=='independent_saved_window_review' and review['first_deterministic_gate']['component_counts']['valid']==0
        selected=review['selected_for_render'];assert len(selected)==2 and [r['active_opponent'] for r in selected]==[False,True]
        clips=[]
        for row in selected:
            file=ROOT/'reviewed-training-events'/row['native_file'];assert sha(file)==row['native_sha256']
            with np.load(file) as a:
                assert len(a['time_before'])==650
                indices=np.r_[np.arange(19,650,20),649]
                clips.append(dict(row=row,qpos=a['evidence_qpos_before'][indices].copy(),qvel=a['evidence_qvel_before'][indices].copy(),times=a['time_before'][indices].copy(),indices=indices.tolist()))
        assert sum(len(c['times']) for c in clips)==66
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),source_review_sha256=sha(source),
            maximum_render_forwards=66,render_forwards=0,prior_render_forwards=prior,
            new_physical_controls=0,new_optimizer_updates=0,new_diagnostic_geometry_queries=0,
            physics_stepped=False,actor_executed=False,live_learner_interrupted=False,
            scope='Verified stochastic training events, not deterministic gate success',video=str(video))
        atomic_json(out/'state.json',state)
        try:
            foundation=UnassistedG1('approach',impl='jax');model=build_pair(foundation,visual_fists=True)
            assert model.nq==72 and model.neq==0 and not np.any(model.body_gravcomp)
            data=mujoco.MjData(model);cam=mujoco.MjvCamera();cam.distance=3.2;cam.azimuth=100;cam.elevation=-18
            roots=model.jnt_qposadr[np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)]
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
            video.parent.mkdir(parents=True,exist_ok=True)
            with mujoco.Renderer(model,height=480,width=640) as renderer,imageio.get_writer(video,fps=25,codec='libx264',quality=8) as writer:
                for index,clip in enumerate(clips):
                    at=int(np.argmin(abs(clip['times']-clip['row']['event']['onset_time'])))
                    q=clip['qpos'][at if args.side_view else 0];cam.lookat[:]=[np.mean([q[r] for r in roots]),np.mean([q[r+1] for r in roots]),.75]
                    if args.side_view:cam.azimuth=float(np.degrees(np.arctan2(q[roots[1]+1]-q[roots[0]+1],q[roots[1]]-q[roots[0]]))+100.)
                    frames=[]
                    for qpos,qvel,time in zip(clip['qpos'],clip['qvel'],clip['times']):
                        state['render_forwards']+=1;assert state['render_forwards']<=66
                        data.qpos[:]=qpos;data.qvel[:]=qvel;data.time=time;mujoco.mj_forward(model,data)
                        renderer.update_scene(data,camera=cam);frames.append(renderer.render().copy())
                    e=clip['row']['event'];at=int(np.argmin(abs(clip['times']-e['onset_time'])))
                    for mode,repeats in (('NORMAL',1),('SLOW 0.25x',4)):
                        for i,(pixels,time) in enumerate(zip(frames,clip['times'])):
                            frame=Image.fromarray(pixels);draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,99),fill='black')
                            role='Active frozen opponent' if clip['row']['active_opponent'] else 'Neutral partner; both bodies free'
                            lines=[f'V15 | Recorded stochastic training | {mode}',
                                f'{role} | Learner corner {e["corner"]}, right fist',
                                f'Onset {e["onset_relative_speed"]:.2f} m/s | Travel {100*e["pre_max_displacement_m"]:.1f} cm | Return {100*e["return_toward_pre_window_position_m"]:.1f} cm',
                                f'Impulse {e["event_window_impulse_ns"]:.2f} N s | Reference cycle criteria met',
                                f'Deterministic gate: 0 valid cycles | Recorded time {time:.3f} s']
                            for y,line in enumerate(lines):draw.text((7,3+19*y),line,font=font,fill='white')
                            for _ in range(repeats):writer.append_data(np.asarray(frame))
                            if mode=='NORMAL' and i==at:frame.save(out/f'event-{index}-preview.png')
            count=0
            with imageio.get_reader(video) as reader:
                assert abs(reader.get_meta_data()['fps']-25)<1e-9
                for frame in reader:
                    assert frame.shape==(480,640,3);count+=1
            assert state['render_forwards']==66 and count==330
            state.update(status='completed',finished_utc=utc(),decoded_frames=count,duration_seconds=count/25,
                video_sha256=sha(video),cumulative_render_forwards=prior+66,
                selections=[dict(**c['row'],rendered_indices=c['indices']) for c in clips])
            state['camera_rule']='Side view relative to root separation at onset' if args.side_view else 'Fixed azimuth'
            atomic_json(out/'state.json',state);atomic_json(ROOT/('verified-training-video-side-view.json' if args.side_view else 'verified-training-video.json'),state)
            print(json.dumps(dict(status='completed',video=str(video),frames=count,render_forwards=66)))
        except BaseException as e:
            state.update(status='failed',error=str(e));atomic_json(out/'state.json',state);raise


if __name__=='__main__':main()
