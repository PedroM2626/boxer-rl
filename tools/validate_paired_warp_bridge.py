"""Check exported GPU forces against the unchanged native measurement API."""
import hashlib
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import mujoco_warp as mjwarp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_physics import PairedWarpPhysics
from boxer.warp_readback import WarpReadback
from tools.validate_paired_warp_dynamics import native_force, gpu_force


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefetch-study', action='store_true')
    args = parser.parse_args()
    out = Path('/home/pedro/boxing-runs/gpu-bridge-prefetch-v1' if args.prefetch_study else '/home/pedro/boxing-runs/gpu-bridge-v1')
    out.mkdir(parents=True, exist_ok=False)
    pair = PairedFoundation(UnassistedG1('approach', impl='jax'))
    pair.model.opt.iterations = 30; pair.model.opt.ls_iterations = 20
    physics = PairedWarpPhysics(pair.model)
    mirror = WarpReadback(physics.data)
    root = Path('docs/results/striking-pilot/contact-progress-v15')
    chosen = json.loads((root/'first-captured-event-review.json').read_text())['selected_for_render']
    results = []
    for selected in chosen:
        path = root/'reviewed-training-events'/selected['native_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == selected['native_sha256']
        with np.load(path) as data:
            index = int(np.argmin(abs(data['time_before']-selected['event']['event_time'])))
            physics.load(data['evidence_qpos_before'][index:index+1], data['evidence_qvel_before'][index:index+1],
                         data['evidence_ctrl'][index:index+1], data['time_before'][index:index+1])
        # Forward snapshots are sufficient here; no dynamics are advanced.
        health = physics.health()
        mjwarp.get_data_into(pair.data, pair.model, physics.data)
        mirrored = mujoco.MjData(pair.model)
        mirror.export(mirrored, pair.model)
        for key in ('qpos', 'qvel', 'ctrl', 'sensordata', 'geom_xpos', 'site_xmat', 'cvel', 'subtree_com', 'efc_force'):
            np.testing.assert_array_equal(getattr(mirrored, key), getattr(pair.data, key))
        np.testing.assert_array_equal(mirrored.contact.geom, pair.data.contact.geom)
        np.testing.assert_array_equal(mirrored.contact.frame, pair.data.contact.frame)
        direct = gpu_force(physics, pair.fists, pair.targets)
        exported = native_force(pair.model, pair.data, pair.fists, pair.targets)
        np.testing.assert_allclose(direct, exported, rtol=1e-6, atol=1e-5)
        assert direct.max() > 1.
        np.testing.assert_allclose(pair.data.qpos, physics.data.qpos.numpy()[0], atol=1e-12)
        results.append(dict(source_sha256=selected['native_sha256'], direct_force=direct.tolist(),
                            exported_force=exported.tolist(), health=health, prefetched_export_matches_exactly=True))
    atomic_json(out/'result.json', dict(status='completed', cases=results,
        gpu_forwards=2, gpu_substeps=0, native_substeps=0, new_formal_controls=0,
        limitation='Measurement export only; not a rollout or a promotion gate'))


if __name__ == '__main__':
    main()
