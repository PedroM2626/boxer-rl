"""Ablated head reports cannot substitute for complete original physical gates."""
import json
from pathlib import Path
import pytest
from tools.review_striking_credit_horizon import full_profiles


@pytest.mark.parametrize('marker', ['diagnostic_zero_command_offsets','diagnostic_zero_arm_residuals'])
def test_masked_report_cannot_certify_original_gate(tmp_path,marker):
    source=Path('docs/results/striking-pilot/fixed-repeat-1908736/gate-8192/screen')
    for name in ('neutral-0.json','neutral-0-progress.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    path=tmp_path/'neutral-0.json'; report=json.loads(path.read_text()); report[marker]=True
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='Masked actions'):
        full_profiles(tmp_path,[7,23,31,47],report['striking_sha256'])


def test_scaled_arm_report_cannot_certify_original_gate(tmp_path):
    source=Path('docs/results/striking-pilot/fixed-repeat-1908736/gate-8192/screen')
    for name in ('neutral-0.json','neutral-0-progress.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    path=tmp_path/'neutral-0.json'; report=json.loads(path.read_text())
    report['diagnostic_arm_residual_scale']=.5
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='Scaled arm actions'):
        full_profiles(tmp_path,[7,23,31,47],report['striking_sha256'])


@pytest.mark.parametrize('changes', [{'physics_backend': 'warp'}, {'diagnostic_solver_override': True}])
def test_experimental_physics_cannot_certify_native_gate(tmp_path, changes):
    source=Path('docs/results/striking-pilot/fixed-repeat-1908736/gate-8192/screen')
    for name in ('neutral-0.json','neutral-0-progress.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    path=tmp_path/'neutral-0.json'; report=json.loads(path.read_text()); report.update(changes)
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='Experimental physics'):
        full_profiles(tmp_path,[7,23,31,47],report['striking_sha256'])
