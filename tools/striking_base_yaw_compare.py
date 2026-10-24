"""Bounded evaluation-only causal probe of automatic approach yaw."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192')
SHA='6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a'


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'comparison.json'
        if path.exists():raise ValueError('Yaw comparison exists; inspect output and process before recovery')
        review=json.loads(Path('docs/results/striking-pilot/mixed-failure-trace-1859584/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==1844480 and review['formal_world_controls_total']==1106474
        saved,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
        assert saved['consumed_steps']==saved['optimized_steps']==1859584 and saved['world_control_steps']==1236992
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),source_sha256=SHA,
            maximum_evaluation_world_controls=36000,diagnostic_evaluation_world_controls=0,
            formal_evaluation_world_controls_before_job=1106474,diagnostic_ppo_retained=1844480,
            new_optimizer_updates=0,new_collection_world_controls=0,assistance=False,unchanged_rewards=True,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            policy_adoption='none',promotion_eligible=False,checks=[],
            control_16m_reference='docs/results/striking-pilot/mixed-confirm-1851392/final-fresh')
        atomic_json(path,state)
        def account():
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
        try:
            for name,scale,seeds in [('modified-16m',.5,list(range(16000003,16000007))),('control-17m',1.,list(range(17000003,17000007))),('modified-17m',.5,list(range(17000003,17000007)))]:
                folder=args.output/name;folder.mkdir(exist_ok=False)
                for role in (0,1,None):
                    profile='selfplay' if role is None else f'neutral-{role}'
                    state.update(active_evaluation=str(folder/profile),updated_utc=utc());atomic_json(path,state)
                    command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),'--seeds',*[str(s) for s in seeds],
                        '--output',str(folder/(profile+'.json')),'--progress',str(folder/(profile+'-progress.json')),'--base-yaw-scale',str(scale)]
                    if role is not None:command+=['--neutral-partner',str(role)]
                    with (folder/(profile+'.log')).open('w') as out,(folder/(profile+'-errors.log')).open('w') as err:
                        subprocess.run(command,stdout=out,stderr=err,check=True)
                    assert not (folder/(profile+'-errors.log')).read_text().strip()
                    report=json.loads((folder/(profile+'.json')).read_text());rows=report['trials']
                    assert len(rows)==8 and {(r['seed'],r['swap']) for r in rows}=={(s,w) for s in seeds for w in (False,True)}
                    assert report['striking_sha256']==SHA and report['diagnostic_base_yaw_scale']==scale and all(r['finite'] for r in rows)
                    assert not report['sampled_policy'] and not report['assistance']
                    surviving=[r for r in rows if r['survived']]
                    state['checks'].append(dict(name=name,profile=profile,base_yaw_scale=scale,episodes=8,survivors=len(surviving),
                        surviving_events_by_corner=[sum(r['scored_by_corner'][i] for r in surviving) for i in (0,1)],
                        surviving_two_way_episodes=sum(all(v>0 for v in r['scored_by_corner']) for r in surviving)))
                    account();assert state['diagnostic_evaluation_world_controls']<=36000;atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise

if __name__=='__main__':main()
