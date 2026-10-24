"""On-policy covariance change preserves learned means, leaves and density accounting."""
import copy,json,math
import jax,jax.numpy as jp,numpy as np,pytest
from boxer.striking_sampling_profile import OFFSET,SOURCE_SHA,prepare_sampling_state,sampling_config,sampling_offset
from boxer.striking_train import Actor,log_probability,require_ppo_checkpoint,main
from tools.striking_objective_audit import SOURCE
from tools.striking_contact_compare import load_checked


def test_same_parameters_preserve_mean_and_half_the_complete_gaussian_std():
    saved,_=load_checked(SOURCE,SOURCE_SHA)
    obs=jp.array(np.random.default_rng(53).normal(size=(8,148)),dtype=jp.float32)
    mean,logstd=Actor().apply(saved['actor'],obs)
    other,shifted=Actor(sampling_log_std_offset=OFFSET).apply(saved['actor'],obs)
    np.testing.assert_array_equal(mean,other)
    np.testing.assert_allclose(np.exp(shifted),np.exp(logstd)*.5,rtol=4*np.finfo(np.float32).eps)
    np.testing.assert_array_equal(logstd,np.clip(np.asarray(saved['actor']['params']['log_std']),-3.,-.5))
    noise=jax.random.normal(jax.random.PRNGKey(7),mean.shape)
    old=mean+jp.exp(logstd)*noise;new=mean+jp.exp(shifted)*noise
    np.testing.assert_allclose(new-mean,.5*(old-mean),atol=1e-7)
    np.testing.assert_allclose(log_probability(new,mean,shifted),log_probability(old,mean,logstd)+17*math.log(2),atol=1e-4)
    logp=log_probability(new,mean,shifted)
    np.testing.assert_array_equal(jp.exp(log_probability(new,other,shifted)-logp),np.ones(8))
    assert not np.allclose(jp.exp(log_probability(new,mean,logstd)-logp),np.ones(8))


def test_transition_preserves_all_learned_and_sampler_leaves_and_requires_profile():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=prepare_sampling_state(saved,SOURCE_SHA);require_ppo_checkpoint(new)
    for key in saved:
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(new[key])):np.testing.assert_equal(a,b)
    assert sampling_offset(saved)==0. and sampling_offset(new)==OFFSET
    assert sampling_config(json.loads((SOURCE/'config.json').read_text()))['version']==10
    for key in ('sampling_profile_transition','sampling_log_std_offset','reduced_noise_ppo'):
        bad=copy.deepcopy(new);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_sampling_state(new,SOURCE_SHA)
    with pytest.raises(ValueError):main(['--reduced-noise-ppo'])
