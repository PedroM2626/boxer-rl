"""CPU/disk-only accounting benchmark on the completed audit; no physics calls."""
import json
from pathlib import Path
import time
from boxer.checkpoint_io import atomic_json
from boxer.gpu_training_ledger import hot_progress


def main():
    root = Path('/home/pedro/boxing-runs/gpu-learning-v1')
    state = json.loads((root/'state.json').read_text())
    out = Path('/home/pedro/boxing-runs/gpu-accounting-profile-v1'); out.mkdir(exist_ok=False)
    results = []
    for name in ('legacy_full', 'compact'):
        start = time.perf_counter()
        for _ in range(128):
            if name == 'legacy_full':
                controls = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in (root/'gates').rglob('*-progress.json'))
                assert controls == state['formal_new']
                atomic_json(out/(name+'.json'), state)
            else: atomic_json(out/(name+'.json'), hot_progress(state))
        results.append(dict(mode=name, durable_writes=128, seconds=time.perf_counter()-start,
            bytes_per_write=(out/(name+'.json')).stat().st_size))
    for key in ('consumed_steps', 'optimized_steps', 'world_control_steps', 'diagnostic_ppo_total', 'formal_total'):
        assert json.loads((out/'legacy_full.json').read_text())[key] == json.loads((out/'compact.json').read_text())[key]
    result = dict(status='completed', source='Completed gpu-learning-v1 state and native child ledgers',
        cases=results, new_simulation_controls=0, new_gpu_forwards=0, new_ppo_steps=0,
        limitation='CPU/disk-only reconstruction of accounting costs; not a GPU training throughput benchmark')
    atomic_json(Path('docs/results/striking-pilot/gpu-learning-v1/accounting-profile.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
