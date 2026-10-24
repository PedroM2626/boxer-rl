"""Mode decomposition must retain the training batch's globally normalized advantages."""
import jax
import jax.numpy as jp
import numpy as np

from boxer.striking_train import Actor, log_probability
from tools.striking_gae_gradient_audit import policy_loss, vector


def test_equal_mode_gradient_average_recovers_whole_batch_without_renormalization():
    actor=Actor();obs=jp.array(np.random.default_rng(4).normal(size=(8,148)),dtype=jp.float32)
    params=actor.init(jax.random.PRNGKey(0),obs);mean,std=actor.apply(params,obs)
    raw=mean+jp.array(np.random.default_rng(5).normal(size=(8,17))*.15,dtype=jp.float32)
    advantage=jp.array([-2.,-1.,-.4,.2,.3,.5,1.,1.4])
    advantage=(advantage-advantage.mean())/advantage.std()
    batch=dict(obs=obs,raw=raw,logp=log_probability(raw,mean,std),adv=advantage)
    fn=jax.value_and_grad(lambda p,b:policy_loss(actor,p,b))
    loss,whole=fn(params,batch)
    parts=[]
    for indices in (np.array([0,2,4,6]),np.array([1,3,5,7])):
        _,g=fn(params,{k:v[indices] for k,v in batch.items()});parts.append(vector(g))
    assert np.isfinite(loss) and np.linalg.norm(vector(whole))>0
    np.testing.assert_allclose(vector(whole),(parts[0]+parts[1])*.5,rtol=2e-5,atol=1e-6)
