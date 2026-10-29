"""Describe failed straight-cycle components and verify model frame conventions."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import numpy as np
from boxer import scene
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation


def main():
    root = Path('docs/results/striking-pilot/legacy-punch-cycles-v1')
    out = root/'component-and-frame-review.json'
    if out.exists(): raise ValueError('Existing review must be retained before a new probe')
    state = json.loads((root/'state.json').read_text())
    assert state['status'] == 'completed' and len(state['episodes']) == 16
    for row in state['episodes']:
        assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest() == row['trace_sha256']
    events = [e for row in state['episodes'] for e in row['events']]
    failed = dict(speed_below_1=sum(e['onset_relative_speed'] < 1. for e in events),
        forward_cosine_below_07=sum(e['onset_forward_cosine'] < .7 for e in events),
        travel_below_12cm=sum(e['pre_max_displacement_m'] < .12 for e in events),
        return_missing_or_below_6cm=sum(e['return_toward_pre_window_position_m'] is None or e['return_toward_pre_window_position_m'] < .06 for e in events),
        incomplete_post_window=sum(not e['post_window_complete'] for e in events),
        contact_longer_than_100ms=sum(e['post_contact_seconds'] > .10 for e in events))
    old = scene.build_model(.75, visual=False)
    data = mujoco.MjData(old); data.qpos[:] = scene.stance(old, 'natural')[0]
    mujoco.mj_forward(old, data)
    current = PairedFoundation(UnassistedG1('approach', impl='jax'))
    frames = []
    for label, model, posed in (('legacy', old, data), ('current', current.model, current.data)):
        for side in scene.SIDES:
            transform = posed.site(side+'imu_in_torso').xmat.reshape(3, 3).T @ posed.body(side+'torso_link').xmat.reshape(3, 3)
            np.testing.assert_allclose(transform, np.eye(3), atol=1e-12)
            frames.append(dict(model=label, corner=side, imu_to_torso_body=transform.tolist()))
    atomic_json(out, dict(status='reviewed', fist_only_events=len(events), component_failures=failed,
        components_overlap=True, posed_model_frames=frames, native_physics_substeps=0,
        new_formal_controls=0, new_ppo_steps=0,
        interpretation='Most legacy events fail the forward straight-punch reference; short travel and return also occur. This does not rule out hooks or other punch types. The tested IMU forward axes agree with their torso body frames, so a fixed frame-axis rotation does not explain these component failures.',
        decision='Do not retroactively weaken straight-cycle gates or declare all historical contacts fake. Retain the separate range/posture transfer deficit and validate batched GPU mechanics before further PPO.'))


if __name__ == '__main__': main()
