"""Recheck immutable mechanics/benchmark ledgers and a captured complete cycle."""
import hashlib
import json
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.striking_punch_trace import describe_events
from tools.run_gpu_closed_loop import valid


def main():
    base = Path('docs/results/striking-pilot')
    failed = json.loads((base/'batched-gpu-mechanics-v1/progress.json').read_text())
    mechanics = json.loads((base/'batched-gpu-mechanics-v1/attempt-2/progress.json').read_text())
    benchmark = json.loads((base/'batched-gpu-benchmark-v1/progress.json').read_text())
    assert failed['status'] == 'failed' and mechanics['status'] == benchmark['status'] == 'completed'
    assert len(mechanics['checks']) == 5 and mechanics['checks'][0]['observation_error_max'] == 0.
    previous = sum(e['gpu_world_substeps'] for e in failed['engine_ledgers'])+failed['reference_gpu_substeps']
    assert previous == mechanics['prior_attempt_gpu_substeps'] == 22
    total = previous+sum(e['gpu_world_substeps'] for e in mechanics['engine_ledgers'])
    assert total == benchmark['prior_development_gpu_substeps'] == 725
    for ledger in mechanics['engine_ledgers']+benchmark['ledgers']:
        assert ledger['gpu_world_substeps'] == (ledger['gpu_active_world_substeps']
            +ledger['gpu_discarded_terminal_world_substeps']+ledger['gpu_warmup_world_substeps'])
    total += sum(e['gpu_world_substeps'] for e in benchmark['ledgers'])
    assert total <= 200000
    captures = []
    for path in (base/'batched-gpu-benchmark-v1').glob('worlds-*-first-corner-*.json'):
        row = json.loads(path.read_text()); raw = path.parent/row['native_file']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row['native_sha256']
        with np.load(raw) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
            assert not data['applied_force_norm'].any()
            assert set(data['policy_optimized_steps']) == {9420800}
            accepted = [e for e in describe_events(data) if valid(e)]
            assert row['event'] in accepted
        captures.append(dict(source_sha256=row['source_sha256'], native_sha256=row['native_sha256'],
            event=row['event'], learner_corner=row['learner_corner'], active_opponent=row['active_opponent']))
    assert captures, 'A full captured positive-cycle history is required for integration readiness'
    result = dict(status='reviewed', development_gpu_world_substeps_including_failed_attempt=total,
        positive_cycle_captures=captures, complete_collection_benchmark=benchmark['cases'],
        new_ppo_steps=0, new_formal_controls=0, main_retained=476041216,
        formal_retained=2298509, diagnostic_ppo_retained=9504000,
        limitation='Identical-trajectory native measurement/control/reward parity is distinct from independent CPU/CUDA trajectory equivalence. No boxing gate passes or long-training adoption.',
        decision='The predeclared one-block 1,024-credit warm-start integration may proceed; no larger allocation implied')
    atomic_json(base/'batched-gpu-mechanics-v1/completed-review.json', result)


if __name__ == '__main__': main()
