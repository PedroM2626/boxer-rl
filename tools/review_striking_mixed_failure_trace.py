"""Independently archive exact passive replays and classify first-fall associations."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
import jax,numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_mixed_failure_trace import summarize_trial


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);live=parser.parse_args().input
    state=json.loads((live/'trace-review.json').read_text());assert state['status']=='completed' and len(state['checks'])==6
    assert state['diagnostic_world_controls']==21499 and state['new_optimizer_updates']==0
    reference=Path('docs/results/striking-pilot/mixed-confirm-1851392');out=Path('docs/results/striking-pilot/mixed-failure-trace-1859584')
    result=dict(status='reviewed',diagnostic_ppo_total=1844480,formal_world_controls_total=1106474,new_evaluation_world_controls=21499,
        new_optimizer_updates=0,new_collection_world_controls=0,new_supervised_updates=0,main_adoption=False,promotion=False,
        native_error_logs_empty=True,checks=[],foundation_history_records={})
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text());assert h==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        f=Path('checkpoints/curriculum')/stage;m=json.loads((f/'latest.json').read_text());b=(f/m['file']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(b)))
        result['foundation_history_records'][stage]=len(h)
    f=Path('checkpoints/striking-pilot/step-1048576');m=json.loads((f/'latest.json').read_text());assert hashlib.sha256((f/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    controls=0;all_falls=[]
    for check in state['checks']:
        name=check['name'];variant,profile=name.split('-',1);folder=live/name
        original=json.loads((reference/('source-fresh' if variant=='source' else 'final-fresh')/(profile+'.json')).read_text())
        report=json.loads((folder/'evaluation.json').read_text());progress=json.loads((folder/'evaluation-progress.json').read_text())
        model=Path('checkpoints/striking-validation')/('actor-rate-1835008/control/gate-16384' if variant=='source' else 'mixed-confirm-1851392/gate-8192')
        saved,_=load_checked(model,check['sha256']);require_ppo_checkpoint(saved)
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((saved['actor'],saved['critic'],saved['optimizer'],saved['norm'])))
        assert progress['status']=='completed' and report['striking_sha256']==original['striking_sha256']==progress['striking_sha256']==check['sha256']
        assert not (folder/'errors.log').read_text().strip() and not report['sampled_policy'] and not report['assistance']
        assert len(report['trials'])==8 and progress['simulated_world_transitions']==report['simulated_world_transitions']
        old={(r['seed'],r['swap']):r for r in original['trials']};summaries=[]
        for trial in report['trials']:
            assert all(trial.get(k)==v for k,v in old[trial['seed'],trial['swap']].items())
            summary=summarize_trial(trial);summaries.append(summary)
        assert summaries==check['trials'];controls+=progress['simulated_world_transitions']
        falls=[r for r in summaries if not r['survived']];all_falls.extend(falls)
        result['checks'].append(dict(name=name,episodes=8,falls=len(falls),all_original_physical_fields_exact=True,
            precontact_falls=sum(r['first_interbody_contact_time'] is None or r['first_interbody_contact_time']>r['first_fall_time'] for r in falls),trials=summaries))
        shutil.copytree(folder,out/name,dirs_exist_ok=True)
    assert controls==21499
    result.update(episodes=48,falls=len(all_falls),low_pelvis_falls=sum(any(v['low_pelvis'] for v in r['terminal_corners']) for r in all_falls),
        tilted_falls=sum(any(v['tilted'] for v in r['terminal_corners']) for r in all_falls),self_leg_contact_falls=sum(any(v['self_leg_contact'] for v in r['terminal_corners']) for r in all_falls),
        precontact_falls=sum(r['first_interbody_contact_time'] is None or r['first_interbody_contact_time']>r['first_fall_time'] for r in all_falls))
    result['decision']='No unchanged PPO allocation. Test one diagnostic-only base approach yaw scale 0.5 versus original 1.0 on the latest final, retaining all learned action channels, physics, rewards and thresholds. Reuse exact 16M control, evaluate modified 16M and both controllers on fresh 17M profiles, at most 36000 world controls. This is a command-path causal probe, not evidence for a leg-authority change or policy adoption. No optimization or automatic follow-on allocation.'
    shutil.copy2(live/'trace-review.json',out/'completed-trace-review.json');atomic_json(out/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
