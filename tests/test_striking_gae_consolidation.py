import copy
from tools.striking_gae_consolidate import advance_frontier
from tools.striking_mixed_partner_continue import progress_vector


def screen():
    return dict(passes_complete=False,profiles={name:dict(pair_survivors=n,
        own_surviving_scoring_seeds=list(range(k)),surviving_two_way_episodes=1)
        for name,n,k in (('neutral-0',4,1),('neutral-1',5,2),('selfplay',8,3))})


def test_failed_repeats_count_stagnation_and_cannot_shrink_the_frontier():
    initial=screen();frontier=progress_vector(initial)
    for expected in (1,2,3):
        frontier,streak=advance_frontier(frontier,expected-1,initial)
        assert streak==expected and frontier==progress_vector(initial)
    worse=copy.deepcopy(initial);worse['profiles']['neutral-0']['pair_survivors']=3
    newer,streak=advance_frontier(frontier,streak,worse)
    assert newer==frontier and streak==4  # severe-regression check is separately mandatory


def test_new_component_resets_stagnation_without_becoming_a_full_pass():
    initial=screen();better=copy.deepcopy(initial)
    better['profiles']['neutral-1']['pair_survivors']=6
    frontier,streak=advance_frontier(progress_vector(initial),2,better)
    assert streak==0 and not better['passes_complete']
    assert frontier[2]==6
    passed=copy.deepcopy(better);passed['passes_complete']=True
    _,streak=advance_frontier(frontier,2,passed)
    assert streak==0  # consecutive full-pass counting still occurs independently
