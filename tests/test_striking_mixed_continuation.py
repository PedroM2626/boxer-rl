"""A diagnostic stop must not masquerade as relaxed complete physical criteria."""
import copy
import json
from pathlib import Path

from tools.striking_mixed_partner_continue import no_measured_improvement, progress_vector


def source_screen():
    report = json.loads(Path('docs/results/striking-pilot/mixed-partner-1736704/analysis.json').read_text())
    return next(a for a in report['arms'] if a['arm'] == 'mixed')['gates'][-1]['screen']


def test_equal_or_worse_failed_screen_stops_without_erasing_a_generation():
    source = source_screen(); worse = copy.deepcopy(source)
    assert progress_vector(source) == (7,1,8,0,6,2,2)
    assert no_measured_improvement(source, source)
    worse['profiles']['selfplay']['pair_survivors'] = 5
    assert no_measured_improvement(source, worse)


def test_partial_measured_progress_permits_only_the_remainder_not_a_phase_pass():
    source = source_screen(); current = copy.deepcopy(source)
    current['profiles']['neutral-1']['own_surviving_scoring_seeds'] = [7]
    assert not current['passes_complete'] and not no_measured_improvement(source, current)


def test_complete_pass_is_never_stopped_as_measurement_stagnation():
    source = source_screen(); current = copy.deepcopy(source)
    current['passes_complete'] = True
    assert not no_measured_improvement(source, current)
