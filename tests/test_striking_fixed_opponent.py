"""Stationary adversary provenance and learner/opponent separation."""
import copy,json
from pathlib import Path
import jax,numpy as np,pytest
from boxer.striking_fixed_opponent import prepare_fixed_state,fixed_config,opponent_state,fingerprint,FROZEN_FINGERPRINT,SOURCE_SHA
from boxer.striking_train import require_ppo_checkpoint,main
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192')


def test_transition_retains_all_learned_state_and_rejects_changed_frozen_opponent():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=prepare_fixed_state(saved,SOURCE_SHA);require_ppo_checkpoint(new)
    for k in saved:
        if k=='opponent_mode':continue
        for a,b in zip(jax.tree_util.tree_leaves(saved[k]),jax.tree_util.tree_leaves(new[k])):np.testing.assert_equal(a,b)
    assert fixed_config(json.loads((SOURCE/'config.json').read_text()))['version']==8
    changed=copy.deepcopy(new);changed['frozen_opponent']['norm']['mean'][0]+=1
    with pytest.raises(ValueError):require_ppo_checkpoint(changed)
    for key in ('fixed_opponent_transition','fixed_opponent_fingerprint','frozen_opponent'):
        bad=copy.deepcopy(new);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):main(['--fixed-opponent-ppo'])


def test_opponent_is_invariant_to_learner_parameter_and_normalizer_updates():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=prepare_fixed_state(saved,SOURCE_SHA)
    changed=jax.tree_util.tree_map(lambda x:np.asarray(x)+.1,saved['actor']);norm=dict(saved['norm'],count=123.)
    a,n=opponent_state(new,changed,norm)
    assert fingerprint(dict(actor=a,norm=n))==FROZEN_FINGERPRINT
    x,y=opponent_state(saved,changed,norm);assert x is changed and y is norm
