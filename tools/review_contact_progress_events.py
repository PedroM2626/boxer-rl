"""Verify immutable captured training windows and the first full v15 gate; no simulation."""
import argparse,hashlib,json,os,pickle,shutil
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_motion_cycle_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites
from tools.review_striking_credit_horizon import full_profiles


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    root=parser.parse_args().input;state=json.loads((root/'audit.json').read_text())
    assert state['status']=='running' and verify_prerequisites()==state['prerequisites']
    archive=Path('docs/results/striking-pilot/contact-progress-v15');destination=archive/'reviewed-training-events';destination.mkdir(parents=True,exist_ok=True)
    files=sorted((root/'training/motion-event-evidence').glob('*.json'));assert files
    reviewed=[];policies={}
    for file in files:
        item=json.loads(file.read_text());native=Path(item['native_file']);assert sha(native)==item['native_sha256']
        with np.load(native) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            assert a['evidence_qpos_before'].shape==(len(a['time_before']),72)
            assert a['evidence_qvel_before'].shape==(len(a['time_before']),70)
            assert a['evidence_ctrl'].shape==(len(a['time_before']),58)
            np.testing.assert_array_equal(a['applied_force_norm'],0.)
            np.testing.assert_allclose(np.diff(a['time_before']),.002,atol=1e-12,rtol=0)
            matches=[e for e in describe_events(a) if (e['corner'],e['hand'],e['event_time'])==(item['event']['corner'],item['event']['hand'],item['event']['event_time'])]
            assert len(matches)==1 and matches[0]==item['event'] and independent_cycle(matches[0])
            e=matches[0];assert e['event_window_impulse_ns']>=.5 and e['native_contact_window_impulse_ns']>=.5
            assert e['onset_contact_closing']>=.3 and e['onset_contact_advancing']>=.2
            assert set(map(int,np.unique(a['policy_optimized_steps'])))=={r['optimized_steps'] for r in item['policy_checkpoints_in_window']}
        assert item['controlled_by']=='learner' and item['event']['corner']==item['learner_corner']
        for policy in item['policy_checkpoints_in_window']:
            point=policy['optimized_steps']
            if point not in policies:
                payload=Path(policy['file']).read_bytes();assert hashlib.sha256(payload).hexdigest()==policy['sha256']
                saved=pickle.loads(payload);require_ppo_checkpoint(saved)
                assert saved['optimized_steps']==point and saved['foundation_sha256']==item['foundation_sha256']
                cp=Path('checkpoints/striking-validation/contact-progress-v15/captured-event-policies')/f'step-{point}'
                cp.mkdir(parents=True,exist_ok=True);(cp/'policy.pkl').write_bytes(payload)
                manifest=json.loads((root/'training/checkpoint/latest.json').read_text());manifest.update(file='policy.pkl',sha256=policy['sha256'],
                    consumed_steps=saved['consumed_steps'],optimized_steps=point,world_control_steps=saved['world_control_steps'])
                (cp/'latest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(root/'training/config.json',cp/'config.json')
                policies[point]=dict(optimized_steps=point,sha256=policy['sha256'],archived_folder=str(cp))
            assert policies[point]['sha256']==policy['sha256']
        shutil.copy2(file,destination/file.name);shutil.copy2(native,destination/native.name)
        reviewed.append(dict(manifest=file.name,native_file=native.name,native_sha256=item['native_sha256'],
            consumed_at_recognition=item['consumed_at_recognition'],new_credits_at_recognition=item['consumed_at_recognition']-3129344,
            learner_corner=item['learner_corner'],active_opponent=item['active_opponent'],reset_seed=item['reset_seed'],swap=item['swap'],event=item['event']))
    reviewed.sort(key=lambda r:r['consumed_at_recognition'])
    gate=next(g for g in state['gates'] if g['label']=='step-524288')
    folder=archive/'gates/step-524288';original,controls=full_profiles(folder,[7,23,31,47],gate['sha256'])
    assert original==gate['original_contact_gate'] and controls==9928 and not gate['passes_complete']
    component_counts={k:0 for k in ('events','speed','direction','pretravel','complete_post','return','short_contact','speed_direction','valid')}
    for role in ('neutral-0','neutral-1','selfplay'):
        report=json.loads((folder/(role+'.json')).read_text());assert not (folder/(role+'-errors.log')).read_text().strip()
        for row in report['trials']:
            native=folder/role/'native'/Path(row['punch_trace_file']).name;assert sha(native)==row['punch_trace_sha256']
            with np.load(native) as a:
                assert all(np.isfinite(a[k]).all() for k in a.files)
                events=describe_events(a);assert events==row['punch_measurements']
            for e in events:
                values=dict(events=True,speed=e['onset_relative_speed']>=1.,direction=e['onset_forward_cosine']>=.7,
                    pretravel=e['pre_max_displacement_m']>=.12,complete_post=e['post_window_complete'],
                    return_ok=e['return_toward_pre_window_position_m'] is not None and e['return_toward_pre_window_position_m']>=.06,
                    short_contact=e['post_contact_seconds']<=.1,speed_direction=e['onset_relative_speed']>=1. and e['onset_forward_cosine']>=.7,
                    valid=independent_cycle(e))
                values['return']=values.pop('return_ok')
                for k,v in values.items():component_counts[k]+=int(v)
    selected=[next(r for r in reviewed if not r['active_opponent']),next(r for r in reviewed if r['active_opponent'])]
    result=dict(status='independent_saved_window_review',new_physical_controls=0,new_optimizer_updates=0,new_geometry_queries=0,
        captured_training_events=len(reviewed),training_events_by_corner={str(c):sum(r['learner_corner']==c for r in reviewed) for c in (0,1)},
        active_opponent_events=sum(r['active_opponent'] for r in reviewed),events=reviewed,policy_snapshots=list(policies.values()),
        selected_for_render=selected,selection_rule='Earliest captured verified event in neutral and active-opponent practice respectively, by consumed-at-recognition order.',
        first_deterministic_gate=dict(new_ppo_steps=524288,sha256=gate['sha256'],controls=controls,
            pair_survivors={k:v['pair_survivors'] for k,v in original['profiles'].items()},component_counts=component_counts,
            complete_original_contact_pass=original['passes_complete'],complete_motion_pass=False),
        formal_world_controls_total=2140974+controls,main_adoption=False,main_project_consumed_retained=476041216,
        limitation='Verified stochastic training windows are not repeatable deterministic held-out policy passes. Some windows span PPO updates. Earlier counter-only events have no raw windows.',
        decision='Keep the productive approved PPO allocation unchanged. First gate has zero valid cycles and fails survival criteria; no promotion, rollback, weakened gate or unbounded extension.')
    (archive/'training-event-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('captured_training_events','training_events_by_corner','active_opponent_events','first_deterministic_gate')}))


if __name__=='__main__':main()
