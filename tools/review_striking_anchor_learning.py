"""Independently archive anchored actors and all original/fresh physical checks."""
import argparse
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.striking_anchor import load_anchor_candidates
from boxer.striking_train import Actor,normalize,require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_coverage_audit import SOURCE,SOURCE_SHA
from tools.striking_anchor_learning import FRESH_SEEDS


def thresholds(checks):
    return (checks['neutral-0']['pair_survivors']>=7 and checks['neutral-1']['pair_survivors']==8 and
        checks['selfplay']['pair_survivors']>=7 and
        all(len(checks['neutral-'+str(c)]['own_surviving_scoring_seeds'])>=2 for c in (0,1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=json.loads((args.output/'learning.json').read_text())
    if state['status']!='completed' or len(state['arms'])!=2:raise ValueError('both matched arms must finish')
    source,_=load_checked(SOURCE,SOURCE_SHA);data,dataset=load_anchor_candidates()
    archive=Path('docs/results/striking-pilot/motor-anchoring-bc4000');archive.mkdir(parents=True,exist_ok=True)
    models=Path('checkpoints/striking-validation/motor-anchoring-bc4000')
    predict=jax.jit(lambda params,x:jp.tanh(Actor().apply(params,x)[0]))
    inputs=jp.array(normalize(data['observation'],source['norm']))
    baseline=np.asarray(predict(source['actor'],inputs))
    controls=0

    def review_reports(folder,destination,seeds,expected_sha):
        nonlocal controls
        destination.mkdir(parents=True,exist_ok=True)
        summaries={}
        for corner in (0,1,None):
            name='selfplay' if corner is None else 'neutral-'+str(corner)
            report=json.loads((folder/(name+'.json')).read_text())
            progress=json.loads((folder/(name+'-progress.json')).read_text())
            if (report['striking_sha256']!=expected_sha or report['simulated_world_transitions']!=progress['simulated_world_transitions'] or
                    progress['status']!='completed' or report['assistance'] or report['certification'] or
                    report['actor_inputs']!=148 or report['actions']!=17 or not report['contact_history'] or
                    report['sampled_policy'] or report['diagnostic_zero_command_offsets'] or report['diagnostic_zero_arm_residuals'] or
                    report['episode_seconds']!=10. or report['neutral_partner_learner_corner']!=corner):
                raise ValueError('complete physical profile provenance mismatch')
            rows=report['trials']
            if (len(rows)!=8 or {(r['seed'],r['swap']) for r in rows}!={(s,w) for s in seeds for w in (False,True)} or
                    not all(r['finite'] for r in rows)):raise ValueError('incomplete independent trials')
            controls+=report['simulated_world_transitions']
            summaries[name]=dict(pair_survivors=sum(r['survived'] for r in rows),
                own_surviving_events=sum((sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])
                    for r in rows if r['survived']),
                own_surviving_scoring_seeds=sorted({r['seed'] for r in rows if r['survived'] and
                    (sum(r['scored_by_corner']) if corner is None else r['scored_by_corner'][corner])>0}))
            if corner is None:
                summaries[name].update(surviving_two_way_episodes=sum(r['survived'] and all(n>0 for n in r['scored_by_corner']) for r in rows),
                    surviving_scoring_seeds_by_corner={str(c):sorted({r['seed'] for r in rows if r['survived'] and r['scored_by_corner'][c]>0})
                        for c in (0,1)})
            for file in folder.glob(name+'*.log'):
                text=file.read_text().lower()
                if any(w in text for w in ('badqpos','badqvel','badqacc','cnstrfull','contactfull','nefc overflow','nonfinite','traceback')):
                    raise ValueError('native diagnostic failure: '+str(file))
            for file in (folder/(name+'.json'),folder/(name+'-progress.json'),folder/(name+'.log'),folder/(name+'-errors.log')):
                shutil.copy2(file,destination/file.name)
        return summaries

    result=dict(source_sha256=SOURCE_SHA,anchor_dataset_sha256=dataset['sha256'],arms={},
        source_anchor_mean_abs_commands=float(np.abs(baseline[:,:3]).mean()),
        source_anchor_mean_abs_arms=float(np.abs(baseline[:,3:]).mean()),
        source_fresh=review_reports(args.output/'source-fresh',archive/'source-fresh',FRESH_SEEDS,SOURCE_SHA))
    for profile,summary in state['source_fresh'].items():
        if any(result['source_fresh'][profile][k]!=v for k,v in summary.items()):
            raise ValueError('source parent/review disagreement')
    for arm in state['arms']:
        name=arm['arm'];folder=args.output/name;saved,manifest=load_checked(folder,arm['checkpoint_sha256'])
        if (saved['supervised_updates']!=5000 or int(saved['optimizer'][1][0].count)!=5000 or
                saved['source_bc_sha256']!=SOURCE_SHA or saved['anchor_dataset_sha256']!=dataset['sha256'] or
                saved['sampler_state']!=arm['sampler_final']):raise ValueError('anchored lineage mismatch')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('nonfinite anchored state')
        for key in ('critic','norm','key'):
            for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(source[key])):
                np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(saved['actor']['params']['log_std'],source['actor']['params']['log_std'])
        for key in ('consumed_steps','optimized_steps','world_control_steps','foundation_sha256'):
            if saved[key]!=source[key]:raise ValueError('inherited main/PPO/physical lineage changed')
        try:require_ppo_checkpoint(saved)
        except ValueError:pass
        else:raise ValueError('BC payload accepted for PPO')
        screen=review_reports(folder/'screen',archive/name/'screen',[7,23,31,47],manifest['sha256'])
        fresh=review_reports(folder/'fresh',archive/name/'fresh',FRESH_SEEDS,manifest['sha256'])
        for label,reviewed in (('screen',screen),('fresh',fresh)):
            for profile,summary in arm[label].items():
                if any(reviewed[profile][k]!=v for k,v in summary.items()):raise ValueError('parent/review disagreement')
        predicted=np.asarray(predict(saved['actor'],inputs))
        result['arms'][name]=dict(sha256=manifest['sha256'],screen=screen,fresh=fresh,
            screen_thresholds_met=thresholds(screen),fresh_thresholds_met=thresholds(fresh),
            optimizer_count=int(saved['optimizer'][1][0].count),sampler_final_matches_parent=True,
            retained_critic_norm_key_logstd_and_lineage=True,finite=True,ppo_resume_rejected=True,
            anchor_mean_abs_commands=float(np.abs(predicted[:,:3]).mean()),anchor_mean_abs_arms=float(np.abs(predicted[:,3:]).mean()),
            source_preservation_mse=float(np.mean((predicted-baseline)**2)))
        destination=models/name;destination.mkdir(parents=True,exist_ok=True)
        for file in (folder/'latest.json',folder/'config.json',folder/manifest['file']):shutil.copy2(file,destination/file.name)
        for file in (folder/'latest.json',folder/'config.json'):shutil.copy2(file,archive/name/file.name)
    if controls!=state['diagnostic_world_controls'] or controls>60000:raise ValueError('world accounting mismatch')
    if state['arms'][0]['sampler_final']!=state['arms'][1]['sampler_final']:raise ValueError('matched samplers diverged')
    result.update(diagnostic_world_controls=controls,formal_world_controls_total=403150+controls,
        new_supervised_updates=state['new_supervised_updates'],cumulative_supervised_updates=6000+state['new_supervised_updates'],
        cumulative_example_presentations=(6000+state['new_supervised_updates'])*256,new_ppo_transitions=0,diagnostic_ppo_retained=1188096,
        foundation_consumed=474992640,main_boxing_consumed=1048576,main_project_consumed=476041216,
        native_numerical_or_capacity_warnings=False,matched_sampler_final=True,policy_adoption='none',promotion=False,
        limitation='Two matched arms are not consecutive checkpoint passes. Physical outcomes, not anchor fit, determine continuation.',
        decision=('Neither arm passes complete screen/fresh scoring criteria. Stop further identical imitation '
            'and preserve both branches. Use the latest zero-anchor BC5000 branch as a diagnostic continuation '
            'source, not an adopted actor. Implement and validate explicit actor/critic optimizer migration '
            'before the separately documented bounded physical PPO comparison; direct BC resume remains rejected.'))
    shutil.copy2(args.output/'learning.json',archive/'learning.json');atomic_json(archive/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
