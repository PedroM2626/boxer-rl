"""Independent native API oracle for batched kinematics, forces and falls."""
from pathlib import Path
import mujoco
import numpy as np
import pytest
from boxer.paired_foundation import PairedFoundation
from boxer.paired_batch_metrics import PairedBatchMetrics, metric_snapshot
from boxer.striking_punch_trace import PunchTrace
from boxer.striking_contact_progress import target_distances
from test_paired_foundation import foundation


def oracle(pair):
    positions = pair.data.geom_xpos.copy()
    velocities = {}
    for geom in set(pair.fists.ravel()) | pair.targets[0] | pair.targets[1]:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(pair.model, pair.data, mujoco.mjtObj.mjOBJ_GEOM, int(geom), v, 0)
        velocities[geom] = v[3:].copy()
    torso = []
    for m in pair.maps:
        site = pair.model.site(m['side']+'imu_in_torso').id
        v = np.zeros(6)
        mujoco.mj_objectVelocity(pair.model, pair.data, mujoco.mjtObj.mjOBJ_SITE, site, v, 0)
        torso.append((pair.data.site_xpos[site].copy(), v))
    trace = PunchTrace(); trace.before_substep(pair, positions, velocities, torso)
    trace.pending['target_center_distance'] = target_distances(pair, positions)
    present = np.zeros((2, 2), bool)
    impulses, closing, advancing = (np.zeros((2, 2)) for _ in range(3))
    geoms, forces = [], []
    for k, contact in enumerate(pair.data.contact):
        pair_geoms = set(map(int, contact.geom))
        force = np.zeros(6); mujoco.mj_contactForce(pair.model, pair.data, k, force)
        geoms.append(contact.geom.copy()); forces.append(max(0., force[0]))
        for c in (0, 1):
            for h, fist in enumerate(pair.fists[c]):
                targets = pair_geoms & pair.targets[1-c]
                if fist not in pair_geoms or not targets: continue
                target = next(iter(targets)); present[c, h] = True
                impulses[c, h] += max(0., force[0])*pair.model.opt.timestep
                direction = positions[target]-positions[fist]
                direction /= max(np.linalg.norm(direction), 1e-8)
                closing[c, h] = max(closing[c, h], (velocities[fist]-velocities[target])@direction)
                origin, velocity = torso[c]
                own = velocity[3:]+np.cross(velocity[:3], positions[fist]-origin)
                advancing[c, h] = max(advancing[c, h], (velocities[fist]-own)@direction)
    return trace.pending, dict(present=present, impulse=impulses, contact_closing=closing,
        contact_advancing=advancing), np.asarray(geoms, int).reshape(-1, 2), np.asarray(forces)


def test_native_kinematics_and_contact_metrics_match_with_reordered_geom_roles(foundation):
    pair = PairedFoundation(foundation); metrics = PairedBatchMetrics(pair)
    snapshots, before, expected, worlds, geoms, forces, falls = [], [], [], [], [], [], []
    sources = [None] + sorted(Path('docs/results/striking-pilot/contact-progress-v15/reviewed-training-events').glob('*.npz'))[:2]
    assert len(sources) == 3
    for world, source in enumerate(sources):
        pair.reset(seed=73, swap=bool(world % 2))
        if source is not None:
            with np.load(source) as a:
                index = min(392, len(a['time_before'])-1)
                pair.data.qpos[:] = a['evidence_qpos_before'][index]
                pair.data.qvel[:] = a['evidence_qvel_before'][index]
                pair.data.ctrl[:] = a['evidence_ctrl'][index]
                pair.data.time = a['time_before'][index]
            mujoco.mj_forward(pair.model, pair.data)
        first, after, geometry, force = oracle(pair)
        snapshots.append(metric_snapshot(pair.data)); before.append(first); expected.append(after)
        worlds.extend([world]*len(geometry)); geoms.extend(geometry[:, ::-1]); forces.extend(force)
        falls.append(pair.fallen())
    state = {k: np.concatenate([s[k] for s in snapshots]) for k in snapshots[0]}
    actual_before = metrics.before(state)
    for key in ('local_position', 'local_velocity', 'elbow', 'toward_nearest_target', 'target_center_distance'):
        np.testing.assert_allclose(actual_before[key], [r[key] for r in before], atol=2e-14)
    for key, actual in metrics.contacts(actual_before, np.asarray(worlds), np.asarray(geoms), np.asarray(forces)).items():
        np.testing.assert_allclose(actual, [r[key] for r in expected], atol=2e-14)
    np.testing.assert_array_equal(metrics.fallen(state, worlds, geoms), falls)
    # Contact presence terminates self-leg collisions even if its force is zero.
    corner, first, second = metrics.leg_pairs[0]
    changed = metrics.fallen(state, [1], [[second, first]])
    assert changed[1, corner]
    np.testing.assert_array_equal(changed[0], falls[0])
    with pytest.raises(ValueError):
        metrics.contacts(actual_before, np.array([-1]), np.array([[first, second]]), np.array([0.]))
