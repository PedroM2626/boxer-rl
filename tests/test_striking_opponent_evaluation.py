"""Independent opposing action routing and separation from ordinary promotion gates."""
import json
from pathlib import Path
import numpy as np,pytest
from test_paired_foundation import foundation
from boxer.striking_opponent_evaluation import route_opponent_action
from tools.review_striking_credit_horizon import full_profiles
from tools.evaluate_striking import main


@pytest.mark.parametrize('corner',[0,1])
def test_only_opposing_corner_changes_and_inputs_remain_untouched(corner):
    original=np.linspace(-.8,.8,34).reshape(2,17);before=original.copy();other=np.full(17,.25)
    result=route_opponent_action(original,other,corner)
    np.testing.assert_array_equal(result[corner],original[corner]);np.testing.assert_array_equal(result[1-corner],other)
    np.testing.assert_array_equal(original,before)


def test_incompatible_or_nonfinite_opponent_is_rejected():
    for action in (np.ones(16),np.full(17,np.nan),np.full(17,2.)):
        with pytest.raises(ValueError):route_opponent_action(np.zeros((2,17)),action,0)
    with pytest.raises(ValueError):main(['--opponent-checkpoint','unused'])
    with pytest.raises(ValueError):main(['--evaluated-corner','0'])


def test_cross_policy_result_cannot_certify_ordinary_mirror_gate(tmp_path):
    source=Path('docs/results/striking-pilot/fixed-opponent-continue-1875968/gate-32768/screen')
    for fn in ('neutral-0.json','neutral-0-progress.json'):(tmp_path/fn).write_bytes((source/fn).read_bytes())
    p=tmp_path/'neutral-0.json';r=json.loads(p.read_text());r['opponent_sha256']='f'*64;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError,match='Mixed-policy'):
        full_profiles(tmp_path,[7,23,31,47],r['striking_sha256'])


def test_same_policy_routing_preserves_native_physics_exactly(foundation):
    import jax,jax.numpy as jp
    from boxer.striking_contact import ContactAwareStrikingEnv
    from boxer.striking_train import Actor,normalize
    from tools.striking_contact_compare import load_checked
    checkpoint=Path('checkpoints/striking-validation/fixed-opponent-continue-1875968/gate-32768')
    saved,_=load_checked(checkpoint,'8003a7bfdf829141634ce63019dea82a700db79613d78afaae0974346130053b')
    actor=Actor();mean=jax.jit(lambda obs:jp.tanh(actor.apply(saved['actor'],obs)[0]))
    def motor(obs,key):return jp.ones((2,29))*.02,{}
    baseline=ContactAwareStrikingEnv(foundation,None,motor=motor);paired=ContactAwareStrikingEnv(foundation,None,motor=motor)
    a=baseline.reset(23);b=paired.reset(23)
    for _ in range(5):
        expected=np.asarray(mean(jp.array(normalize(a,saved['norm']))))
        own=np.asarray(mean(jp.array(normalize(b,saved['norm']))))
        other=np.asarray(mean(jp.array(normalize(b,saved['norm']))))[1]
        routed=route_opponent_action(own,other,0)
        np.testing.assert_array_equal(expected,routed)
        a,ra,da,oa=baseline.step(expected);b,rb,db,ob=paired.step(routed)
        for x,y in ((a,b),(ra,rb),(baseline.pair.data.qpos,paired.pair.data.qpos),(baseline.pair.data.qvel,paired.pair.data.qvel),(oa['scored'],ob['scored'])):np.testing.assert_array_equal(x,y)
        assert da==db
