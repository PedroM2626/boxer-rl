"""Archive and independently check a completed matched contact-history comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_train import Actor, normalize, require_ppo_checkpoint
from tools.striking_contact_compare import SOURCE, SOURCE_SHA, load_checked, load_demonstrations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='completed durable experiment directory')
    args=parser.parse_args()
    state=json.loads((args.output/'comparison.json').read_text())
    if state['status']!='completed' or len(state['arms'])!=2:
        raise ValueError('both arms must complete before review')
    old,_=load_checked(SOURCE,SOURCE_SHA)
    migrated=(args.output/'expanded-source.pkl').read_bytes()
    if hashlib.sha256(migrated).hexdigest()!=state['expanded_source_sha256']:
        raise ValueError('expanded source checksum mismatch')
    expanded=pickle.loads(migrated)
    observations,labels,flags,files=load_demonstrations()
    roles=np.repeat([int(name.split('-role-')[1][0]) for name in files],500)
    prediction=jax.jit(lambda p,x:jp.tanh(Actor().apply(p,x)[0]))
    result=dict(source_sha256=SOURCE_SHA,arms={},policy_adoption='none',promotion=False,
        new_ppo_transitions=0,new_supervised_updates=state['new_supervised_updates'],
        cumulative_supervised_updates=2000+state['new_supervised_updates'],
        cumulative_example_presentations=(2000+state['new_supervised_updates'])*256,
        formal_world_controls_before_job=366084,diagnostic_ppo_retained=1188096,
        main_project_consumed=476041216,foundation_consumed=474992640,main_boxing_consumed=1048576)
    controls=0
    archive=Path('docs/results/striking-pilot/contact-history-bc2000')
    models=Path('checkpoints/striking-validation/contact-history-bc2000')
    def equal(a,b):
        if jax.tree_util.tree_structure(a)!=jax.tree_util.tree_structure(b):
            raise ValueError('learning-state structure mismatch')
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(x,y)
    for arm in state['arms']:
        label=arm['arm'];folder=args.output/label
        saved,manifest=load_checked(folder,arm['checkpoint_sha256'])
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('nonfinite learning state')
        for key in ('critic','norm','key'):equal(saved[key],expanded[key])
        equal(saved['actor']['params']['log_std'],old['actor']['params']['log_std'])
        for key in ('consumed_steps','optimized_steps','world_control_steps','foundation_sha256'):
            if saved[key]!=old[key]:raise ValueError('changed inherited lineage')
        if saved['supervised_updates']!=4000 or int(saved['optimizer'][1][0].count)!=4000:
            raise ValueError('supervised update accounting mismatch')
        if saved['sampler_state']!=arm['sampler_final']:
            raise ValueError('sampler snapshot mismatch')
        try:require_ppo_checkpoint(saved)
        except ValueError:pass
        else:raise ValueError('supervised payload accepted for PPO')
        if label=='zero_history':
            for tree in (saved['actor'],saved['optimizer'][1][0].mu,saved['optimizer'][1][0].nu):
                np.testing.assert_array_equal(tree['params']['Dense_0']['kernel'][146:],0.)
        summaries={}
        for corner in (0,1,None):
            name='selfplay' if corner is None else 'neutral-'+str(corner)
            report=json.loads((folder/(name+'.json')).read_text())
            progress=json.loads((folder/(name+'-progress.json')).read_text())
            if (progress['status']!='completed' or progress['simulated_world_transitions']!=report['simulated_world_transitions'] or
                    report['striking_sha256']!=manifest['sha256'] or report['actor_inputs']!=148 or
                    not report['contact_history'] or report['certification'] or report['assistance']):
                raise ValueError('physical report provenance mismatch')
            rows=report['trials']
            if (len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in (7,23,31,47) for w in (False,True)} or
                    not all(r['finite'] for r in rows)):
                raise ValueError('incomplete physical trials')
            controls+=report['simulated_world_transitions']
            summaries[name]=dict(pair_survivors=sum(r['survived'] for r in rows),
                own_surviving_events=sum((sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])
                    for r in rows if r['survived']),
                own_surviving_scoring_seeds=sorted({r['seed'] for r in rows if r['survived'] and
                    (sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])>0}),
                surviving_two_way_episodes=sum(r['survived'] and all(n>0 for n in r['scored_by_corner']) for r in rows))
        thresholds=(summaries['neutral-0']['pair_survivors']>=7 and summaries['neutral-1']['pair_survivors']==8 and
            summaries['selfplay']['pair_survivors']>=7 and
            all(len(summaries['neutral-'+str(c)]['own_surviving_scoring_seeds'])>=2 for c in (0,1)))
        result['arms'][label]=dict(sha256=manifest['sha256'],finite_learning_state=True,
            retained_critic_normalizer_key_logstd_and_lineage=True,ppo_resume_rejected=True,
            complete_screen_thresholds_met=thresholds,evaluations=summaries)
        inputs=np.c_[observations,flags if label=='contact_history' else np.zeros_like(flags)]
        fitted=np.asarray(prediction(saved['actor'],jp.array(normalize(inputs,saved['norm']))))
        zero=np.asarray(prediction(saved['actor'],jp.array(normalize(np.c_[observations,np.zeros_like(flags)],saved['norm']))))
        fit={}
        for role in (0,1):
            for after in (False,True):
                mask=(roles==role)&(np.any(flags,axis=1)==after)
                residual=np.abs(fitted[mask]-labels[mask])
                fit[f'role_{role}_'+('after_hit' if after else 'before_hit')]=dict(rows=int(mask.sum()),
                    mean_absolute_commands=float(residual[:,:3].mean()),
                    mean_absolute_left_arm=float(residual[:,3:10].mean()),
                    mean_absolute_right_arm=float(residual[:,10:].mean()),
                    max_absolute_action_error=float(residual.max()),
                    mean_action_change_from_history=float(np.abs(fitted[mask]-zero[mask]).mean()))
        result['arms'][label]['offline_demonstration_fit']=fit
        for name,summary in arm['evaluations'].items():
            if any(summaries[name][k]!=v for k,v in summary.items()):
                raise ValueError('parent and independent physical summaries disagree')
        for file in folder.glob('*.log'):
            text=file.read_text().lower()
            if any(w in text for w in ('badqpos','badqvel','badqacc','cnstrfull','contactfull','nefc overflow','nonfinite','traceback')):
                raise ValueError('native failure in evaluation logs: '+str(file))
        destination=archive/label;destination.mkdir(parents=True,exist_ok=True)
        model=models/label;model.mkdir(parents=True,exist_ok=True)
        for file in folder.iterdir():
            if file.suffix in ('.json','.log'):shutil.copy2(file,destination/file.name)
            if file.name in ('latest.json','config.json',manifest['file']):shutil.copy2(file,model/file.name)
    if controls!=state['diagnostic_evaluation_world_controls'] or controls>24000:
        raise ValueError('physical accounting mismatch')
    if state['arms'][0]['sampler_final']!=state['arms'][1]['sampler_final']:
        raise ValueError('matched sample streams diverged')
    result.update(diagnostic_world_controls=controls,formal_world_controls_total=366084+controls,
        matched_sampler_final=True,native_numerical_or_capacity_warnings=False,
        decision=('Reject both arms for adoption and stop further blind imitation. Neither complete screen passes; '
            'contact feedback does not establish stable own strikes in both roles. Retain all branch checkpoints '
            'without substituting an earlier source. Recorded-state fit remains worse before impact. '
            'Next investigate policy-state coverage and first-fall traces in a documented bounded read-only '
            'diagnostic; allocate no more supervised or PPO learning without that review.'),
        offline_fit_limitation='Recorded successful demonstration states only; not independent physical evidence or proof of a failure cause.')
    shutil.copy2(args.output/'comparison.json',archive/'comparison.json')
    shutil.copy2(args.output/'expanded-source.pkl',models/'expanded-source.pkl')
    atomic_json(archive/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
