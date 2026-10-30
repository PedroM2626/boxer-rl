"""Crash/frontier regression tests for compact durable training accounting."""
import copy
import json
from pathlib import Path
import pytest
from boxer.gpu_training_ledger import hot_progress, reconcile_prefix, write_progress


def state():
    return dict(source_sha256='a'*64, maximum_new_ppo_steps=1000,
        diagnostic_ppo_before=100, diagnostic_ppo_ceiling=1100,
        formal_before=200, formal_ceiling=400, main_retained=300,
        consumed_steps=500, optimized_steps=500, world_control_steps=450,
        new_ppo_steps=0, diagnostic_ppo_total=100, formal_new=0, formal_total=200,
        engine_ledger={'world_controls': 0, 'gpu_world_substeps': 8},
        gates=[{'large_evidence': 'kept in the full audit'}], rollouts=[{'sha256': 'b'*64}])


def test_hot_progress_retains_attempted_credit_without_rewriting_history(tmp_path):
    audit = state(); write_progress(tmp_path, audit, full=True)
    first_audit = (tmp_path/'state.json').read_bytes()
    attempted = copy.deepcopy(audit); attempted['consumed_steps'] += 128
    attempted['world_control_steps'] += 128; attempted['new_ppo_steps'] += 128
    attempted['diagnostic_ppo_total'] += 128
    attempted['engine_ledger']['world_controls'] += 128
    write_progress(tmp_path, attempted)
    assert (tmp_path/'state.json').read_bytes() == first_audit
    hot = json.loads((tmp_path/'progress.json').read_text())
    assert hot['consumed_steps'] == 628 and 'gates' not in hot and 'rollouts' not in hot
    with pytest.raises(ValueError, match='unoptimized'):
        reconcile_prefix(audit, hot, audit)


def test_completed_new_frontier_and_identity_checks(tmp_path):
    audit = state(); completed = copy.deepcopy(audit)
    for key in ('consumed_steps', 'optimized_steps', 'world_control_steps'): completed[key] += 128
    completed['engine_ledger']['gpu_world_substeps'] += 1280
    result = reconcile_prefix(audit, hot_progress(completed), completed)
    assert result['consumed_steps'] == 628 and result['gates'] == audit['gates']
    assert result['engine_ledger']['gpu_world_substeps'] == 1288
    wrong = hot_progress(completed); wrong['diagnostic_ppo_ceiling'] += 128
    with pytest.raises(ValueError, match='identity'): reconcile_prefix(audit, wrong, completed)
    # A newer full audit must win over an old hot write after an interrupted flush.
    assert reconcile_prefix(completed, hot_progress(audit), completed)['consumed_steps'] == 628
