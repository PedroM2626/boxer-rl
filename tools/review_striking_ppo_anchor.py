"""Verify the completed matched PPO comparison and measure offline policy drift."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_anchor import load_anchor_candidates
from boxer.striking_ppo_warmstart import validate_split_checkpoint
from boxer.striking_train import Actor,normalize
from tools.striking_contact_compare import load_checked,load_demonstrations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();live=args.input
    state=json.loads((live/'comparison.json').read_text())
    assert state['status']=='completed' and len(state['arms'])==2
    assert state['diagnostic_consumed_steps']==262144
    assert state['diagnostic_collection_world_controls']==131072
    report=Path('docs/results/striking-pilot/ppo-anchor-bc5000')
    models=Path('checkpoints/striking-validation/ppo-anchor-bc5000')
    source,_=load_checked(live/'migration',state['migrated_sha256'])
    datasets={}
    anchors,_=load_anchor_candidates();datasets['approach_anchors']=(anchors['observation'],anchors['action'])
    obs,labels,flags,_=load_demonstrations();datasets['strike_demonstrations']=(np.c_[obs,flags],labels)
    actor=Actor()
    predict=jax.jit(lambda params,x:jp.tanh(actor.apply(params,x)[0]))
    def metrics(params,norm):
        out={}
        for name,(observations,target) in datasets.items():
            pred=np.asarray(predict(params,jp.array(normalize(observations,norm))))
            assert np.isfinite(pred).all()
            out[name]=dict(rows=len(pred),mse=float(np.mean((pred-target)**2)),
                command_mse=float(np.mean((pred[:,:3]-target[:,:3])**2)),
                left_arm_mse=float(np.mean((pred[:,3:10]-target[:,3:10])**2)),
                mean_abs_action=float(np.abs(pred).mean()))
        return out
    results=dict(status='reviewed',main_adoption=False,promotion=False,
        diagnostic_ppo_total=state['diagnostic_ppo_before_job']+262144,
        formal_world_controls_total=state['formal_evaluation_world_controls_before_job']+state['diagnostic_evaluation_world_controls'],
        new_ppo_collection_world_controls=131072,cumulative_supervised_updates=8000,
        source_offline=metrics(source['actor'],source['norm']),arms=[],foundation_history_records={})
    expected_eval=0
    for log in live.rglob('*errors.log'):
        if log.read_text().strip():raise ValueError(f'Nonempty native error log: {log}')
    for stage in ('balance','walk_easy','walk','approach'):
        history=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert history==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text())
        assert promotion_ready(history,stage)
        folder=Path('checkpoints/curriculum')/stage
        m=json.loads((folder/'latest.json').read_text())
        payload=(folder/m['file']).read_bytes();assert hashlib.sha256(payload).hexdigest()==m['sha256']
        import pickle
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(payload)))
        results['foundation_history_records'][stage]=len(history)
    main_folder=Path('checkpoints/striking-pilot/step-1048576')
    m=json.loads((main_folder/'latest.json').read_text())
    assert hashlib.sha256((main_folder/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    for arm in state['arms']:
        name=arm['arm'];folder=live/name
        saved,manifest=load_checked(folder/'checkpoint',arm['final_checkpoint_sha256'])
        validate_split_checkpoint(saved)
        assert saved['consumed_steps']==saved['optimized_steps']==1572864
        assert saved['world_control_steps']==983040
        assert int(saved['optimizer']['actor_adam'][0].count)==7048
        assert int(saved['optimizer']['critic_adam'][0].count)==30720
        assert saved['bc_sampler_state']==source['bc_sampler_state']
        checks={}
        for sample,seeds in (('screen',[7,23,31,47]),('fresh',state['fresh_seeds'])):
            profiles={}
            for role in (0,1,None):
                profile='selfplay' if role is None else f'neutral-{role}'
                r=json.loads((folder/sample/(profile+'.json')).read_text());rows=r['trials']
                assert len(rows)==8 and {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in seeds for w in (False,True)}
                assert all(t['finite'] for t in rows) and r['striking_sha256']==manifest['sha256']
                expected_eval+=r['simulated_world_transitions']
                own_seeds=sorted({t['seed'] for t in rows if t['survived'] and (sum(t['scored_by_corner']) if role is None else t['scored_by_corner'][role])>0})
                survival=sum(t['survived'] for t in rows)
                profiles[profile]=dict(survivors=survival,own_scoring_seeds=own_seeds,
                    surviving_events_by_corner=np.sum([t['scored_by_corner'] for t in rows if t['survived']],axis=0).tolist(),
                    surviving_two_way_episodes=sum(t['survived'] and all(x>0 for x in t['scored_by_corner']) for t in rows),
                    all_episode_events_by_corner=np.sum([t['scored_by_corner'] for t in rows],axis=0).tolist(),
                    mean_final_root_distance=float(np.mean([t['final_distance'] for t in rows])),
                    passes_profile=survival>=(8 if role==1 else 7) and (role is None or len(own_seeds)>=2))
            checks[sample]=dict(profiles=profiles,passes_complete=all(p['passes_profile'] for p in profiles.values()))
            shutil.copytree(folder/sample,report/name/sample,dirs_exist_ok=True)
        assert not any(c['passes_complete'] for c in checks.values())
        out=models/name;out.mkdir(parents=True,exist_ok=True)
        for filename in (manifest['file'],'latest.json'):shutil.copy2(folder/'checkpoint'/filename,out/filename)
        shutil.copy2(folder/'config.json',out/'config.json')
        for filename in ('progress.json','state.json','initial-block-verified.json'):
            shutil.copy2(folder/filename,report/name/filename)
        metrics_rows=[]
        for label in ('initial-block','remaining'):
            metrics_rows.extend(json.loads(line) for line in (folder/(label+'.log')).read_text().splitlines() if line.strip())
        assert len(metrics_rows)==128 and all(np.isfinite(r['mean_loss']) for r in metrics_rows)
        assert [r['optimized_steps'] for r in metrics_rows]==list(range(1442816,1572865,1024))
        (report/name/'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics_rows))
        results['arms'].append(dict(arm=name,sha256=manifest['sha256'],checks=checks,
            actor_adam_count=7048,critic_adam_count=30720,
            final_actor_final_norm=metrics(saved['actor'],saved['norm']),
            final_actor_source_norm=metrics(saved['actor'],source['norm']),
            source_actor_final_norm=metrics(source['actor'],saved['norm']),
            log_std_source=np.asarray(source['actor']['params']['log_std']).tolist(),
            log_std_final=np.asarray(saved['actor']['params']['log_std']).tolist()))
    for path in (live/'source-fresh').glob('*-progress.json'):
        expected_eval+=json.loads(path.read_text())['simulated_world_transitions']
    assert expected_eval==state['diagnostic_evaluation_world_controls']==53652
    finals=[load_checked(live/a['arm']/'checkpoint',a['final_checkpoint_sha256'])[0] for a in state['arms']]
    assert finals[0]['aux_sampler_state']==finals[1]['aux_sampler_state']
    results.update(native_error_logs_empty=True,auxiliary_sampler_streams_match=True,
        decision='Reject both final actors for adoption; no PPO extension. Next: bounded saved-Gaussian evaluation of source and both final actors to distinguish deterministic-mean regression from sampled policy behavior. No altered noise or gates; diagnostic only.')
    shutil.copy2(live/'comparison.json',report/'completed-comparison.json')
    atomic_json(report/'analysis.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
