"""Bounded unseen-reset full groups and independent learned-opponent diagnostics."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from tools.review_striking_credit_horizon import full_profiles

SOURCE=Path('checkpoints/striking-validation/gae-consolidate-1998848/gate-65536')
SHA='bea8147f6a13dd2d473a558f2e2ac68d3492e3136d2899e2d9daf139364bb870'
OPPONENTS=[
    ('training_snapshot',Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192'),'6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a'),
    ('prior_long_gae',Path('checkpoints/striking-validation/gae-1966080/long_gae/gate-16384'),'95dbdff01def598cdc907e93f1dc7bceb2f1d6b58bc22f488a1936afa86553f9')]
VIDEO=Path('fights/striking-pilot/gae-readiness-2064384.mp4')


def cross_policy_summary(rows,corner):
    surviving=[r for r in rows if r['survived']]
    own=sorted({r['seed'] for r in surviving if r['scored_by_corner'][corner]>0})
    other=sorted({r['seed'] for r in surviving if r['scored_by_corner'][1-corner]>0})
    two=sum(all(v>0 for v in r['scored_by_corner']) for r in surviving)
    return dict(own_scoring_seeds=own,opponent_scoring_seeds=other,
        passes_cross_policy_diagnostic=len(surviving)>=7 and len(own)>=2 and len(other)>=2 and two>=2)



def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=args.output/'audit.json'
        if path.exists():raise ValueError('Generalization audit exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/gae-consolidate-1998848/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==2131200 and review['formal_world_controls_total']==1874882 and review['original_complete_pass_streak']==2
        saved,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
        for _,opponent,sha in OPPONENTS:
            other,_=load_checked(opponent,sha);require_ppo_checkpoint(other);assert other['foundation_sha256']==saved['foundation_sha256']
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),source_sha256=SHA,
            maximum_evaluation_world_controls=65000,diagnostic_evaluation_world_controls=0,formal_evaluation_world_controls_before_job=1874882,
            diagnostic_ppo_retained=2131200,new_optimizer_updates=0,new_collection_world_controls=0,assistance=False,unchanged_physics=True,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,policy_adoption='none',promotion=False,
            original_complete_pass_streak_retained=2,cross_policy_checks=[],independent_groups=[],video_path=str(VIDEO),video_seed=29000003,
            cross_policy_diagnostic_criteria=dict(minimum_pair_survivors=7,minimum_own_scoring_seeds=2,minimum_opponent_scoring_seeds=2,minimum_reciprocal_episodes=2),
            formal_world_controls_ceiling=1939882,screen_milestone_retained=True,general_boxing_certified=False)
        atomic_json(path,state)
        def account():state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            for name,opponent,sha in OPPONENTS:
                for corner in (0,1):
                    folder=args.output/name/f'corner-{corner}';folder.mkdir(parents=True,exist_ok=False)
                    state.update(phase='cross_policy_evaluation',active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                    cmd=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),'--opponent-checkpoint',str(opponent),'--evaluated-corner',str(corner),
                        '--seeds',*[str(s) for s in range(31000003,31000007)],'--output',str(folder/'evaluation.json'),'--progress',str(folder/'evaluation-progress.json')]
                    with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:subprocess.run(cmd,stdout=out,stderr=err,check=True)
                    assert not (folder/'errors.log').read_text().strip()
                    r=json.loads((folder/'evaluation.json').read_text());rows=r['trials']
                    assert r['striking_sha256']==SHA and r['opponent_sha256']==sha and r['evaluated_corner']==corner
                    assert len(rows)==8 and {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in range(31000003,31000007) for w in (False,True)} and all(t['finite'] for t in rows)
                    survivors=[t for t in rows if t['survived']]
                    state['cross_policy_checks'].append(dict(opponent=name,opponent_sha256=sha,evaluated_corner=corner,pair_survivors=len(survivors),evaluated_corner_fallen=sum(t['fallen'][corner] for t in rows),opponent_corner_fallen=sum(t['fallen'][1-corner] for t in rows),surviving_events_by_corner=[sum(t['scored_by_corner'][i] for t in survivors) for i in (0,1)],surviving_two_way_episodes=sum(all(v>0 for v in t['scored_by_corner']) for t in survivors)))
                    state['cross_policy_checks'][-1].update(cross_policy_summary(rows,corner))
                    account();assert state['diagnostic_evaluation_world_controls']<=65000;atomic_json(path,state)
            for group in range(4):
                seeds=list(range(30000003+4*group,30000007+4*group));folder=args.output/f'group-{group}'
                state.update(phase='independent_full_group',active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                evaluate(SOURCE,folder,seeds,state,path);full,_=full_profiles(folder,seeds,SHA)
                state['independent_groups'].append(dict(group=group,seeds=seeds,full=full));account();assert state['diagnostic_evaluation_world_controls']<=65000;atomic_json(path,state)
            folder=args.output/'selected-replay';folder.mkdir(exist_ok=False);VIDEO.parent.mkdir(parents=True,exist_ok=True)
            state.update(phase='selected_video_replay',active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
            cmd=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),'--seeds','29000003',
                '--output',str(folder/'evaluation.json'),'--progress',str(folder/'evaluation-progress.json'),'--video',str(VIDEO)]
            with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:subprocess.run(cmd,stdout=out,stderr=err,check=True)
            assert not (folder/'errors.log').read_text().strip()
            r=json.loads((folder/'evaluation.json').read_text());original=json.loads(Path('docs/results/striking-pilot/gae-consolidate-1998848/final-fresh/selfplay.json').read_text())
            expected={(t['seed'],t['swap']):t for t in original['trials'] if t['seed']==29000003}
            assert len(r['trials'])==2 and r['striking_sha256']==SHA and VIDEO.is_file()
            for trial in r['trials']:assert all(trial.get(k)==v for k,v in expected[trial['seed'],trial['swap']].items()), 'Video replay changed physical fields'
            load_checked(SOURCE,SHA);account();assert state['diagnostic_evaluation_world_controls']<=65000
            state['selected_replay']=dict(seed=29000003,all_original_physical_fields_exact=True,video_path=str(VIDEO),
                note='First seed of the already evaluated final 29M mirror group, illustration only. Both swaps replayed; video shows the first arrangement. Not an untouched holdout or proof of general boxing.')
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise

if __name__=='__main__':main()
