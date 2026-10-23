"""Complete physical evidence must retain both roles, seeds and source identity."""
import json
from pathlib import Path
import shutil

import pytest

from tools.review_striking_credit_horizon import full_profiles

REPORT = Path('docs/results/striking-pilot/credit-horizon-1671168/gamma_0999/fresh')
SHA = '22f4908af40b7c5269952d40a98f35630c2300d50d2bf8ba54c125f36cb6dd55'
SEEDS = list(range(10000003, 10000007))


def test_active_strikes_do_not_mask_failed_complete_neutral_gates():
    result, controls = full_profiles(REPORT, SEEDS, SHA)
    assert controls > 0 and not result['passes_complete']
    assert result['profiles']['selfplay']['surviving_two_way_episodes'] == 1
    assert len(result['profiles']['selfplay']['own_surviving_scoring_seeds']) == 4
    assert not result['profiles']['neutral-1']['passes_profile']


@pytest.mark.parametrize('mutation', ['missing_swap', 'wrong_hash', 'nonfinite'])
def test_incomplete_or_mismatched_physical_evidence_rejected(tmp_path, mutation):
    shutil.copytree(REPORT, tmp_path / 'reports')
    path = tmp_path / 'reports/selfplay.json'
    report = json.loads(path.read_text())
    if mutation == 'missing_swap':
        report['trials'].pop()
    elif mutation == 'wrong_hash':
        report['striking_sha256'] = '0' * 64
    else:
        report['trials'][0]['finite'] = False
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        full_profiles(path.parent, SEEDS, SHA)
