"""Bounded unseen-reset full groups and independent learned-opponent diagnostics."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from tools.review_striking_credit_horizon import full_profiles

SOURCE=Path('checkpoints/striking-validation/fixed-repeat-1908736/gate-8192')
SHA='670bed08e371b30a4843a5c934ebcf16edb32a9a76147135c75ae52904b44aa5'
OPPONENTS=[('training_snapshot',Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192'),'6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a'),('prior_learner_snapshot',Path('checkpoints/striking-validation/fixed-opponent-continue-1875968/gate-32768'),'8003a7bfdf829141634ce63019dea82a700db79613d78afaae0974346130053b')]


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=args.output/'audit.json'
        if path.exists():raise ValueError('Generalization audit exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/fixed-repeat-1908736/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==1918208 and review['formal_world_controls_total']==1296474 and review['original_complete_pass_streak']==2
        saved,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
        for _,opponent,sha in OPPONENTS:
            other,_=load_checked(opponent,sha);require_ppo_checkpoint(other);assert other['foundation_sha256']==saved['foundation_sha256']
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),source_sha256=SHA,
            maximum_evaluation_world_controls=64000,diagnostic_evaluation_world_controls=0,formal_evaluation_world_controls_before_job=1296474,
            diagnostic_ppo_retained=1918208,new_optimizer_updates=0,new_collection_world_controls=0,assistance=False,unchanged_physics=True,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,policy_adoption='none',promotion=False,
            original_complete_pass_streak_retained=2,cross_policy_checks=[],independent_groups=[])
        atomic_json(path,state)
        def account():state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            for name,opponent,sha in OPPONENTS:
                for corner in (0,1):
                    folder=args.output/name/f'corner-{corner}';folder.mkdir(parents=True,exist_ok=False)
                    state.update(phase='cross_policy_evaluation',active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                    cmd=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),'--opponent-checkpoint',str(opponent),'--evaluated-corner',str(corner),
                        '--seeds',*[str(s) for s in range(22000003,22000007)],'--output',str(folder/'evaluation.json'),'--progress',str(folder/'evaluation-progress.json')]
                    with (folder/'evaluation.log').open('w') as out,(folder/'errors.log').open('w') as err:subprocess.run(cmd,stdout=out,stderr=err,check=True)
                    assert not (folder/'errors.log').read_text().strip()
                    r=json.loads((folder/'evaluation.json').read_text());rows=r['trials']
                    assert r['striking_sha256']==SHA and r['opponent_sha256']==sha and r['evaluated_corner']==corner
                    assert len(rows)==8 and {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in range(22000003,22000007) for w in (False,True)} and all(t['finite'] for t in rows)
                    survivors=[t for t in rows if t['survived']]
                    state['cross_policy_checks'].append(dict(opponent=name,opponent_sha256=sha,evaluated_corner=corner,pair_survivors=len(survivors),evaluated_corner_fallen=sum(t['fallen'][corner] for t in rows),opponent_corner_fallen=sum(t['fallen'][1-corner] for t in rows),surviving_events_by_corner=[sum(t['scored_by_corner'][i] for t in survivors) for i in (0,1)],surviving_two_way_episodes=sum(all(v>0 for v in t['scored_by_corner']) for t in survivors)))
                    account();assert state['diagnostic_evaluation_world_controls']<=64000;atomic_json(path,state)
            for group in range(4):
                seeds=list(range(21000003+4*group,21000007+4*group));folder=args.output/f'group-{group}'
                state.update(phase='independent_full_group',active_evaluation=str(folder),updated_utc=utc());atomic_json(path,state)
                evaluate(SOURCE,folder,seeds,state,path);full,_=full_profiles(folder,seeds,SHA)
                state['independent_groups'].append(dict(group=group,seeds=seeds,full=full));account();assert state['diagnostic_evaluation_world_controls']<=64000;atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise

if __name__=='__main__':main()
