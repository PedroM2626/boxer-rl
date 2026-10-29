"""Independently recheck saved dynamics arrays and prototype accounting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json


def main():
    root = Path('docs/results/striking-pilot')
    out = root/'gpu-preflight-review.json'
    if out.exists(): raise ValueError('Existing review requires an explicit new revision')
    reviews = []
    native, gpu = 80, 80  # Retained interrupted first attempt, separately archived.
    for name in ('gpu-dynamics-v1', 'gpu-dynamics-ls-v1'):
        folder = root/name
        result = json.loads((folder/'result.json').read_text())
        assert result['status'] == 'completed' and len(result['cases']) == 9
        files = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in folder.glob('*.npz')}
        assert len(files) == 9
        for case in result['cases']:
            with np.load(files[case['trace_sha256']]) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                assert arrays['native_qpos'].shape == arrays['gpu_qpos'].shape == (250, 72)
                pe = abs(arrays['native_qpos']-arrays['gpu_qpos'])
                ve = abs(arrays['native_qvel']-arrays['gpu_qvel'])
                assert float(pe.max()) == case['qpos_max_abs'] and float(ve.max()) == case['qvel_max_abs']
                assert not np.any(arrays['gpu_flags'].astype(int) & ~1536), 'No capacity overflow permitted'
                np.testing.assert_allclose(arrays['native_force'].sum(axis=0)*.002, case['native_legal_impulse'], atol=1e-12)
                np.testing.assert_allclose(arrays['gpu_force'].sum(axis=0)*.002, case['gpu_legal_impulse'], atol=1e-12)
                joints = list(range(7, 36))+list(range(43, 72))
                reviews.append(dict(study=name, case=case['name'], iterations=case['iterations'],
                    ls_iterations=case.get('ls_iterations', 5), qpos_coordinate_error_max=float(pe.max()),
                    joint_error_max_rad=float(pe[:, joints].max()),
                    root_position_error_max_m=float(pe[:, [0, 1, 2, 36, 37, 38]].max()),
                    solver_flag_observations=int(np.count_nonzero(arrays['gpu_flags'])),
                    legal_impulse_error_max_ns=float(np.max(abs((arrays['native_force']-arrays['gpu_force']).sum(axis=0)*.002)))))
        native += result['native_substeps']; gpu += result['gpu_world_substeps']
    benchmark = json.loads((root/'gpu-benchmark-v1/result.json').read_text())
    assert benchmark['status'] == 'completed' and [c['worlds'] for c in benchmark['cases']] == [1, 32, 128, 512]
    for case in benchmark['cases']:
        assert case['world_controls_per_second'] == case['worlds']*40/case['seconds']
        assert not any(f & ~1536 for f in case['final_health']['overflow_flags'])
    gpu += benchmark['gpu_world_substeps']
    assert native == 4580 and gpu == 308109
    atomic_json(out, dict(status='reviewed', scope='Preflights only; closed-loop comparison is separately accounted',
        native_experimental_substeps=native, gpu_experimental_world_substeps=gpu,
        historical_and_new_gpu_world_forwards=1382, new_formal_controls=0, new_learner_steps=0,
        reviewed_dynamics=reviews, benchmark_world_controls_per_second={str(c['worlds']): c['world_controls_per_second'] for c in benchmark['cases']},
        decision='Fast batching is feasible; original low solver limits are not converged. No training/backend adoption or boxing completion from preflights.'))


if __name__ == '__main__': main()
