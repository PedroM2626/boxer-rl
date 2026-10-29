"""Pinned readback must preserve exact solver values and retained host snapshots.

This GPU test uses three forwards and zero physics steps/PPO controls. Account
that diagnostic compute separately when invoking it; do not run it implicitly
in repeated training checks.
"""
from pathlib import Path
import numpy as np
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_physics import PairedWarpPhysics
from boxer.paired_batch_metrics import metric_snapshot
from test_paired_foundation import foundation


def test_packed_values_forces_and_snapshot_lifetime_match_direct_reads(foundation):
    pair = PairedFoundation(foundation)
    physics = PairedWarpPhysics(pair.model, worlds=1)
    physics.prepare_packed_readback()
    sources = [None]+sorted(Path('docs/results/striking-pilot/contact-progress-v15/reviewed-training-events').glob('*.npz'))[:2]
    previous = frozen = None
    for source in sources:
        qpos, qvel, ctrl = pair.data.qpos.copy(), pair.data.qvel.copy(), pair.data.ctrl.copy()
        if source is not None:
            with np.load(source) as rows:
                index = np.nonzero(rows['event'])[0][0]
                qpos, qvel, ctrl = (rows[key][index].copy() for key in
                    ('evidence_qpos_before', 'evidence_qvel_before', 'evidence_ctrl'))
        physics.load(qpos[None], qvel[None], ctrl[None])
        packed = physics.packed_readback()
        actual = metric_snapshot(packed, gpu=True)
        expected = metric_snapshot(physics.data, gpu=True)
        for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
        assert physics.health(packed, synchronized=True) == physics.health()
        for a, b in zip(physics.normal_forces(packed), physics.normal_forces()):
            np.testing.assert_array_equal(a, b)
        if previous is not None:
            for key in frozen: np.testing.assert_array_equal(previous[key], frozen[key])
        previous = actual; frozen = {key: value.copy() for key, value in actual.items()}
