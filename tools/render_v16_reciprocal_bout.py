"""Render an independently verified full deterministic reciprocal bout from saved states."""
import fcntl
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import build_pair
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_rebalance_supervisor import independent_cycle

ARCHIVE = Path('docs/results/striking-pilot/contact-rebalance-v16')
GATE = ARCHIVE / 'gates/step-524288'
OUT = Path('/home/pedro/boxing-runs/v16-first-reciprocal-video')
VIDEO = Path('fights/striking-pilot/v16-reciprocal-seed31-swapped.mp4')
POLICY = '2d618717c1a1e3ccded77311d3bf45cd9a5737b30653836034337be019df20e3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for lock in (OUT / '.lock', Path('runs/striking_saved_event_render.lock')):
            fcntl.flock(stack.enter_context(lock.open('a')), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'state.json').exists() or VIDEO.exists():
            raise ValueError('Existing render requires accounting review before retry')
        independent = json.loads((ARCHIVE / 'step-524288-independent-review.json').read_text())
        assert independent['sha256'] == POLICY and not independent['passes_complete']
        report = json.loads((GATE / 'selfplay.json').read_text())
        assert report['striking_sha256'] == POLICY and not report['sampled_policy'] and not report['assistance']
        trial = next(r for r in report['trials'] if r['seed'] == 31 and r['swap'])
        assert trial['survived'] and trial['finite']
        native = GATE / 'selfplay/native' / Path(trial['punch_trace_file']).name
        trace = GATE / 'selfplay/full' / Path(trial['trace_file']).name
        assert sha(native) == trial['punch_trace_sha256'] and sha(trace) == trial['trace_sha256']
        with np.load(native) as a:
            events = describe_events(a)
            assert events == trial['punch_measurements']
            valid = [e for e in events if independent_cycle(e)]
        assert [sum(e['corner'] == c for e in valid) for c in (0, 1)] == [3, 1]
        with np.load(trace) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            times, qpos, qvel = (a[k][1::2].copy() for k in ('time_after', 'qpos', 'qvel'))
        assert len(times) == 250 and np.allclose(np.diff(times), .04, atol=1e-10)
        state = dict(status='running', started_utc=utc(), policy_sha256=POLICY, seed=31, swap=True,
                     source_native_sha256=sha(native), source_full_sha256=sha(trace), events=valid,
                     prior_render_forwards=2697, maximum_render_forwards=250, render_forwards=0,
                     new_physical_controls=0, new_optimizer_updates=0, physics_stepped=False)
        atomic_json(OUT / 'state.json', state)
        try:
            model = build_pair(UnassistedG1('approach', impl='jax'), visual_fists=True)
            assert model.nq == 72 and model.neq == 0 and not np.any(model.body_gravcomp)
            data = mujoco.MjData(model)
            roots = model.jnt_qposadr[np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)]
            at = int(np.argmin(abs(times - valid[0]['onset_time']))); q = qpos[at]
            cam = mujoco.MjvCamera(); cam.distance = 3.8; cam.elevation = -18
            cam.azimuth = float(np.degrees(np.arctan2(q[roots[1]+1]-q[roots[0]+1], q[roots[1]]-q[roots[0]]))+100)
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
            with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(VIDEO, fps=25, codec='libx264', quality=8) as writer:
                for i, (t, q, v) in enumerate(zip(times, qpos, qvel)):
                    data.qpos[:] = q; data.qvel[:] = v; data.time = t
                    cam.lookat[:] = [np.mean(q[roots]), np.mean(q[roots+1]), .8]
                    mujoco.mj_forward(model, data); state['render_forwards'] += 1
                    renderer.update_scene(data, camera=cam)
                    frame = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 65), fill='black')
                    scores = [sum(e['corner'] == c and e['event_time'] <= t for e in valid) for c in (0, 1)]
                    draw.text((8, 4), 'Deterministic self-play | Selected surviving bout', font=font, fill='white')
                    draw.text((8, 32), f'Time {t:.2f} s | Valid strikes: corner 0 {scores[0]}/3, corner 1 {scores[1]}/1', font=font, fill='white')
                    writer.append_data(np.asarray(frame))
                    if i == at:
                        frame.save(OUT / 'preview.png')
            with imageio.get_reader(VIDEO) as reader:
                assert reader.get_meta_data()['fps'] == 25
                frames = sum(1 for _ in reader)
            assert frames == state['render_forwards'] == 250
            state.update(status='completed', finished_utc=utc(), frames=frames, video_sha256=sha(VIDEO),
                         cumulative_render_forwards=2947, video=str(VIDEO),
                         limitation='Selected known-seed example; the full survival gate failed.')
            atomic_json(OUT / 'state.json', state)
            atomic_json(ARCHIVE / 'reciprocal-seed31-video.json', state)
            print(json.dumps({'status': 'completed', 'video': str(VIDEO), 'frames': frames}))
        except BaseException as error:
            state.update(status='failed', error=str(error)); atomic_json(OUT / 'state.json', state); raise


if __name__ == '__main__':
    main()
