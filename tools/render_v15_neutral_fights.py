"""Render two selected, independently reviewed deterministic neutral bouts from saved states."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from contextlib import ExitStack

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
from tools.striking_contact_progress_supervisor import independent_cycle

GATE = Path('docs/results/striking-pilot/contact-progress-v15/gates/fresh-final')
ARCHIVE = Path('docs/results/striking-pilot/contact-progress-v15')
OUTPUT = Path('/home/pedro/boxing-runs/v15-neutral-fight-videos')
POLICY_SHA = '9132ab4c2c86ed2b2958d551125b001370bf0f4e80f403490d33111ac3bb634a'
SELECTIONS = [(0, 34000004, False, 6), (1, 34000006, False, 4)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for lock in (OUTPUT / '.lock', Path('runs/striking_saved_event_render.lock')):
            handle = stack.enter_context(lock.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / 'state.json').exists():
            raise ValueError('Existing render must be inspected before any repeat')
        review = json.loads((ARCHIVE / 'fresh-final-independent-review.json').read_text())
        assert review['sha256'] == POLICY_SHA and review['all_saved_arrays_finite']
        clips = []
        for role, seed, swap, expected in SELECTIONS:
            report = json.loads((GATE / f'neutral-{role}.json').read_text())
            assert report['striking_sha256'] == POLICY_SHA and report['neutral_partner_learner_corner'] == role
            row = next(x for x in report['trials'] if x['seed'] == seed and x['swap'] == swap)
            assert row['survived'] and row['finite']
            native = GATE / f'neutral-{role}' / 'native' / Path(row['punch_trace_file']).name
            full = GATE / f'neutral-{role}' / 'full' / Path(row['trace_file']).name
            assert sha(native) == row['punch_trace_sha256'] and sha(full) == row['trace_sha256']
            with np.load(native) as arrays:
                events = describe_events(arrays)
                assert events == row['punch_measurements']
                valid = [e for e in events if e['corner'] == role and independent_cycle(e)]
            assert len(valid) == expected
            with np.load(full) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                times = arrays['time_after'][1::2].copy()
                positions = arrays['qpos'][1::2].copy()
                velocities = arrays['qvel'][1::2].copy()
            assert len(times) == 250 and np.allclose(np.diff(times), .04, atol=1e-10)
            clips.append(dict(role=role, seed=seed, swap=swap, events=valid, times=times,
                              qpos=positions, qvel=velocities, native_sha=sha(native), full_sha=sha(full)))
        state = dict(status='running', started_utc=utc(), policy_sha256=POLICY_SHA,
                     maximum_render_forwards=500, prior_render_forwards=2197, render_forwards=0,
                     new_physical_controls=0, new_optimizer_updates=0, physics_stepped=False, videos=[])
        atomic_json(OUTPUT / 'state.json', state)
        try:
            model = build_pair(UnassistedG1('approach', impl='jax'), visual_fists=True)
            assert model.nq == 72 and model.neq == 0 and not np.any(model.body_gravcomp)
            data = mujoco.MjData(model)
            roots = model.jnt_qposadr[np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)]
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
            video_dir = Path('fights/striking-pilot'); video_dir.mkdir(parents=True, exist_ok=True)
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                for clip in clips:
                    role = clip['role']; filename = video_dir / f'v15-neutral-corner-{role}-fresh-bout.mp4'
                    if filename.exists():
                        raise ValueError(f'Video already exists: {filename}')
                    onset = clip['events'][0]['onset_time']
                    q = clip['qpos'][int(np.argmin(abs(clip['times'] - onset)))]
                    camera = mujoco.MjvCamera(); camera.distance = 3.8; camera.elevation = -18
                    camera.azimuth = float(np.degrees(np.arctan2(q[roots[1]+1]-q[roots[0]+1], q[roots[1]]-q[roots[0]]))+100)
                    with imageio.get_writer(filename, fps=25, codec='libx264', quality=8) as writer:
                        for i, (qpos, qvel, time) in enumerate(zip(clip['qpos'], clip['qvel'], clip['times'])):
                            data.qpos[:] = qpos; data.qvel[:] = qvel; data.time = time
                            camera.lookat[:] = [np.mean(qpos[roots]), np.mean(qpos[roots + 1]), .8]
                            mujoco.mj_forward(model, data); state['render_forwards'] += 1
                            assert state['render_forwards'] <= 500
                            renderer.update_scene(data, camera=camera)
                            frame = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(frame)
                            draw.rectangle((0, 0, 640, 63), fill='black')
                            done = sum(e['event_time'] <= time for e in clip['events'])
                            draw.text((8, 4), f'Deterministic evaluation | Neutral partner | Learner corner {role}', font=font, fill='white')
                            draw.text((8, 30), f'Time {time:4.2f} s | Valid strikes {done}/{len(clip["events"])}', font=font, fill='white')
                            writer.append_data(np.asarray(frame))
                            if i == int(np.argmin(abs(clip['times']-clip['events'][0]['event_time']))):
                                frame.save(OUTPUT / f'corner-{role}-impact.png')
                    with imageio.get_reader(filename) as reader:
                        assert abs(reader.get_meta_data()['fps'] - 25) < 1e-9
                        count = sum(1 for _ in reader)
                    assert count == 250
                    state['videos'].append(dict(file=str(filename), sha256=sha(filename), frames=count,
                                                seed=clip['seed'], swap=clip['swap'], learner_corner=role,
                                                native_sha256=clip['native_sha'], full_sha256=clip['full_sha'], events=clip['events']))
            assert state['render_forwards'] == 500
            state.update(status='completed', finished_utc=utc(), cumulative_render_forwards=2697)
            atomic_json(OUTPUT / 'state.json', state)
            atomic_json(ARCHIVE / 'deterministic-neutral-videos.json', state)
            print(json.dumps({'status': state['status'], 'videos': [v['file'] for v in state['videos']]}))
        except BaseException as error:
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(OUTPUT / 'state.json', state)
            raise


if __name__ == '__main__':
    main()
