"""Independently review the complete v14 ledger and all saved native evaluations."""
import hashlib,json,os,pickle
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_motion_cycle_supervisor import independent_cycle,SEEDS,FRESH,PPO_BEFORE,EVAL_BEFORE,MAX_PPO,MAX_EVAL
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_pool_compare import verify_prerequisites


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('docs/results/striking-pilot/motion-cycle-v14')
    state=json.loads((root/'job-state.json').read_text());progress=json.loads((root/'training-progress.json').read_text())
    assert state['status']=='completed' and state['phase']=='awaiting_interpretation'
    assert state['new_ppo_steps']==state['optimized_new_steps']==MAX_PPO
    assert progress['consumed_steps']==progress['optimized_steps']==3129344
    assert progress['world_control_steps']==2506752
    assert (progress['actor_adam_count'],progress['critic_adam_count'])==(31368,55040)
    assert state['diagnostic_ppo_total']==PPO_BEFORE+MAX_PPO
    assert state['new_collection_world_controls']==MAX_PPO
    assert state['main_project_consumed_retained']==476041216 and state['main_boxing_retained']==1048576
    assert verify_prerequisites()==state['prerequisites']
    labels=['baseline','step-262144','step-524288','step-786432','step-1048576','fresh-final']
    assert [g['label'] for g in state['gates']]==labels
    summaries=[];controls=0;native_files=0
    for gate in state['gates']:
        label=gate['label'];folder=root/'gates'/label
        cp=Path('checkpoints/striking-validation/motion-cycle-v14')/label
        manifest=json.loads((cp/'latest.json').read_text());payload=cp/manifest['file']
        assert sha(payload)==manifest['sha256']==gate['sha256']
        saved=pickle.loads(payload.read_bytes());require_ppo_checkpoint(saved)
        assert saved['consumed_steps']==saved['optimized_steps']==2080768+gate['new_ppo_steps']
        original,n=full_profiles(folder,FRESH if label=='fresh-final' else SEEDS,gate['sha256'])
        assert original==gate['original_contact_gate'];controls+=n
        events=[];files=[];trials=0
        for role in ('neutral-0','neutral-1','selfplay'):
            report=json.loads((folder/(role+'.json')).read_text())
            assert not (folder/(role+'-errors.log')).read_text().strip()
            report_controls=0;valid=np.zeros(2,int);own_seeds=set();reciprocal=set()
            corner=None if role=='selfplay' else int(role[-1])
            for row in report['trials']:
                full=folder/role/'full'/Path(row['trace_file']).name
                native=folder/role/'native'/Path(row['punch_trace_file']).name
                assert sha(full)==row['trace_sha256'] and sha(native)==row['punch_trace_sha256']
                with np.load(full) as a:
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    report_controls+=len(a['time_after'])
                    assert a['time_after'][-1]==row['elapsed']
                    np.testing.assert_array_equal(a['scored'].sum(axis=0),row['scored_by_corner'])
                    assert row['survived']==(not a['fallen'][-1].any())
                with np.load(native) as a:
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    np.testing.assert_allclose(np.diff(a['time_before']),.002,atol=1e-12,rtol=0)
                    measured=describe_events(a);assert measured==row['punch_measurements']
                counts=np.zeros(2,int)
                for e in measured:
                    accepted=independent_cycle(e)
                    if accepted:counts[e['corner']]+=1
                    events.append(dict(role=role,**e,valid_cycle=accepted))
                valid+=counts
                if row['survived']:
                    if corner is not None and counts[corner]>0:own_seeds.add(row['seed'])
                    if corner is None and np.all(counts>0):reciprocal.add(row['seed'])
                files.append(dict(full_trace_sha256=row['trace_sha256'],native_trace_sha256=row['punch_trace_sha256']))
                trials+=1;native_files+=1
            assert report_controls==report['simulated_world_transitions']
            actual=dict(valid_cycles_by_corner=valid.tolist(),own_surviving_valid_seeds=sorted(own_seeds),
                reciprocal_surviving_valid_seeds=sorted(reciprocal),passes=len(reciprocal if corner is None else own_seeds)>=2)
            assert actual==gate['motion'][role]
        assert len(events)==len(gate['events']) and sum(e['valid_cycle'] for e in events)==sum(e['valid_cycle'] for e in gate['events'])
        summaries.append(dict(label=label,new_ppo_steps=gate['new_ppo_steps'],sha256=gate['sha256'],trials=trials,
            pair_survivors={k:v['pair_survivors'] for k,v in original['profiles'].items()},
            original_contact_gate_passes=original['passes_complete'],valid_cycles=sum(e['valid_cycle'] for e in events),
            legacy_events=len(events),onset_speed_and_direction_events=sum(e['onset_relative_speed']>=1 and e['onset_forward_cosine']>=.7 for e in events),
            complete_gate_passes=gate['passes_complete'],source_files=files))
    assert native_files==144 and controls==state['diagnostic_evaluation_controls']<=MAX_EVAL
    assert state['formal_world_controls_total']==EVAL_BEFORE+controls
    result=dict(status='independently_reviewed_completed_allocation',new_ppo_steps=MAX_PPO,new_evaluation_controls=controls,
        new_collection_world_controls=MAX_PPO,diagnostic_ppo_total=PPO_BEFORE+MAX_PPO,
        formal_world_controls_total=EVAL_BEFORE+controls,unused_evaluation_controls_expire=MAX_EVAL-controls,
        main_project_consumed_retained=476041216,checkpoint_final_sha256=state['gates'][-1]['sha256'],
        training_valid_learner_cycles=progress['motion_learner_cycles_total'],training_valid_opponent_cycles=progress['motion_opponent_cycles_total'],
        gates=summaries,new_physical_controls_for_review=0,new_optimizer_updates_for_review=0,new_geometry_queries=0,
        finished_utc=state['finished_utc'],main_adoption=False,original_history_unchanged=True,
        decision='No learned valid-cycle milestone or boxing certification. Do not resume an exhausted allocation blindly. Analyze reward components and exploration before a separately justified continuation; this finite block does not prove learning impossible.')
    (root/'completed-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','new_ppo_steps','new_evaluation_controls','formal_world_controls_total','unused_evaluation_controls_expire','training_valid_learner_cycles')}))


if __name__=='__main__':main()
