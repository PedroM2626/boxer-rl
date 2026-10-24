"""Independently verify frozen assay batches and archive all source comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_train import Actor, Critic, advantages, normalize, log_probability, require_ppo_checkpoint
from boxer.striking_retention import retention_loss
from tools.striking_contact_compare import load_checked
from tools.striking_gae_gradient_audit import JOBS, policy_loss, vector, cosine


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;state=json.loads((live/'gradients.json').read_text())
    assert state['status']=='completed' and state['diagnostic_world_controls']==12288
    assert state['new_optimizer_updates']==state['new_ppo_training_credits']==state['new_supervised_updates']==0
    assert state['new_diagnostic_retention_rows']==3072 and len(state['variants'])==3
    out=Path('docs/results/striking-pilot/gae-gradient-1998848');out.mkdir(parents=True,exist_ok=True)
    actor=Actor();critic=Critic();predict=jax.jit(lambda p,x:actor.apply(p,x))
    value=jax.jit(lambda p,x:critic.apply(p,x))
    gradient=jax.jit(jax.grad(lambda p,b:policy_loss(actor,p,b)))
    retention_gradient=jax.jit(jax.grad(lambda p,b:retention_loss(actor,p,b['retention_obs'],b['retention_labels'],b['retention_weights'])))
    critic_gradient=jax.jit(jax.grad(lambda p,b:.25*jp.mean((critic.apply(p,b['obs'])-b['returns'])**2)))
    result=dict(status='reviewed',diagnostic_ppo_total=2065664,formal_world_controls_total=1804660,
        new_diagnostic_collection_world_controls=12288,new_optimizer_updates=0,new_ppo_training_credits=0,
        new_supervised_updates=0,new_diagnostic_retention_rows=3072,main_adoption=False,promotion=False,variants=[])
    labels={}
    for entry,(name,relative,sha) in zip(state['variants'],JOBS):
        assert entry['variant']==name and entry['sha256']==sha and len(entry['blocks'])==4
        saved,_=load_checked(Path('checkpoints/striking-validation')/relative,sha);require_ppo_checkpoint(saved)
        rows=[]
        for block in entry['blocks']:
            number=block['block'];path=live/f'{name}-block-{number}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==block['batch_sha256']
            with np.load(path) as arrays:
                assert all(np.isfinite(x).all() for x in arrays.values())
                assert arrays['rollout_obs'].shape==(128,8,148) and arrays['raw'].shape==(1024,17)
                assert arrays['retention_labels'].shape==(256,17)
                batch={k:jp.array(arrays[k]) for k in ('obs','raw','logp','adv','returns')}
                np.testing.assert_array_equal(arrays['obs'],normalize(arrays['rollout_obs'].reshape(-1,148),saved['norm']))
                mean,std=predict(saved['actor'],batch['obs'])
                np.testing.assert_allclose(log_probability(batch['raw'],mean,std),arrays['logp'],rtol=1e-5,atol=1e-5)
                values=np.asarray(value(saved['critic'],batch['obs'])).reshape(128,8)
                np.testing.assert_allclose(values,arrays['rollout_value'],rtol=1e-5,atol=1e-5)
                adv,returns=advantages(arrays['rollout_reward'],arrays['rollout_done'],values,arrays['bootstrap'],gamma=.999,lam=.99)
                np.testing.assert_allclose(returns.reshape(-1),arrays['returns'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(((adv-adv.mean())/(adv.std()+1e-8)).reshape(-1),arrays['adv'],rtol=1e-5,atol=1e-5)
                actual=vector(gradient(saved['actor'],batch))
                np.testing.assert_allclose(np.linalg.norm(actual),block['measurements']['ppo']['gradient_l2'],rtol=1e-5)
                retention=vector(retention_gradient(saved['actor'],{k:jp.array(arrays[k]) for k in ('retention_obs','retention_labels','retention_weights')}))
                cg=vector(critic_gradient(saved['critic'],batch))
                np.testing.assert_allclose(np.linalg.norm(retention),block['measurements']['retention']['gradient_l2'],rtol=1e-5)
                np.testing.assert_allclose(np.linalg.norm(cg),block['measurements']['critic']['gradient_l2'],rtol=1e-5)
                np.testing.assert_allclose(cosine(actual,retention),block['retention_ppo_cosine'],rtol=1e-4,atol=1e-6)
                modes=arrays['mode_labels'];assert modes.shape==(1024,) and np.sum(modes==0)==np.sum(modes==1)==512
                parts=[vector(gradient(saved['actor'],{k:v[modes==i] for k,v in batch.items()})) for i in (0,1)]
                np.testing.assert_allclose(actual,(parts[0]+parts[1])*.5,rtol=1e-4,atol=1e-6)
                if number not in labels:labels[number]=(arrays['retention_labels'].copy(),arrays['retention_weights'].copy())
                else:
                    np.testing.assert_array_equal(labels[number][0],arrays['retention_labels'])
                    np.testing.assert_array_equal(labels[number][1],arrays['retention_weights'])
                rows.append(dict(block=number,ppo_l2=block['measurements']['ppo']['gradient_l2'],
                    retention_l2=block['measurements']['retention']['gradient_l2'],critic_l2=block['measurements']['critic']['gradient_l2'],
                    value_mse=block['value_mse'],retention_ppo_cosine=block['retention_ppo_cosine'],neutral_active_ppo_cosine=cosine(*parts)))
            shutil.copy2(path,out/path.name)
        load_checked(Path('checkpoints/striking-validation')/relative,sha)
        result['variants'].append(dict(variant=name,sha256=sha,blocks=rows,
            mean_ppo_l2=float(np.mean([r['ppo_l2'] for r in rows])),mean_retention_l2=float(np.mean([r['retention_l2'] for r in rows])),
            mean_value_mse=float(np.mean([r['value_mse'] for r in rows])),
            sampled_completed_rounds=entry['completed_rounds'],sampled_fall_rounds=entry['fall_rounds']))
    result['limitations']=state['limitations']
    errors=Path('runs/striking_gae_gradient_audit_host_error.log').read_text()
    assert all(line.startswith("wsl: Failed to translate") for line in errors.splitlines() if line.strip())
    (out/'host-errors.txt').write_text(errors)
    result['decision']=('The assay does not support retention-gradient dominance or a larger final critic/GAE error as an established cause of regression. Mean retention gradient is about .30-.33 versus PPO 6.93-7.99; final critic error is not larger than source/passing in these small frozen batches. This does not rule out long-run/minibatch/momentum effects. Do not tune another hyperparameter or restore the earlier passing actor. Permit one bounded patient consolidation of the latest final 3052e0f6 with unchanged version 12, lambda .99, original physics/rewards/frozen opponent and all learned/Adam/normalizer/sampler state: at most 65536 new learner/world credits, full gates every 16384, new 29M source/final full sets, at most 72000 evaluation controls. Stop on severe regression, three consecutive failed screens without a new best progress-vector component, or two distinct consecutive complete passes (then freeze latest and review fresh, never automatically adopt). Diagnostic PPO starts 2065664, ceiling 2131200; formal simulation starts 1804660, ceiling 1876660. This bounded consolidation is justified by repeatable active survival and intermittent complete passes, not by an identified causal cure. Preserve all failures and consumed compute; no rollback, gate weakening, main adoption or automatic further extension.')
    shutil.copy2(live/'gradients.json',out/'completed-gradients.json');atomic_json(out/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('limitations',)},indent=2))


if __name__=='__main__':main()
