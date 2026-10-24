import copy
from tools.striking_gae_generalization import cross_policy_summary, render_notice_only


def trials():
    return [dict(seed=seed,swap=swap,survived=True,
        scored_by_corner=[1.,1.] if seed in (1,2) and not swap else [0.,0.])
        for seed in (1,2,3,4) for swap in (False,True)]


def test_cross_policy_requires_survival_and_distinct_scoring_seeds_on_both_sides():
    rows=trials()
    assert cross_policy_summary(rows,0)['passes_cross_policy_diagnostic']
    assert cross_policy_summary(rows,1)['passes_cross_policy_diagnostic']
    rows[0]['survived']=False
    check=cross_policy_summary(rows,0)
    assert not check['passes_cross_policy_diagnostic']
    assert check['own_scoring_seeds']==check['opponent_scoring_seeds']==[2]


def test_only_exact_render_process_notice_is_recognized():
    notice=("/usr/lib/python3.12/subprocess.py:1885: RuntimeWarning: os.fork() was called. "
        "os.fork() is incompatible with multithreaded code, and JAX is multithreaded, "
        "so this will likely lead to a deadlock.\n  self.pid = _fork_exec(\n")
    assert render_notice_only('') and render_notice_only(notice)
    assert not render_notice_only(notice+'mjWARN_BADQPOS\n')
    assert not render_notice_only('RuntimeWarning: unknown failure\n')
    assert not render_notice_only('mjWARN_CNSTRFULL\n')


def test_same_seed_both_swaps_or_only_opponent_impacts_cannot_fake_breadth():
    rows=trials();rows[2]['scored_by_corner']=[0.,0.];rows[1]['scored_by_corner']=[1.,1.]
    assert not cross_policy_summary(rows,0)['passes_cross_policy_diagnostic']
    rows=copy.deepcopy(trials())
    for row in rows:row['scored_by_corner'][0]=0.
    check=cross_policy_summary(rows,0)
    assert check['own_scoring_seeds']==[] and check['opponent_scoring_seeds']==[1,2]
    assert not check['passes_cross_policy_diagnostic']
