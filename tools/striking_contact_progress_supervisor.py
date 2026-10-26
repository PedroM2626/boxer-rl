"""Persistent target-coupled contact-progress PPO with an immutable historical baseline."""
import argparse,fcntl,hashlib,json,os,pickle,shutil,signal,subprocess,sys
from contextlib import ExitStack
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import numpy as np
from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_contact_progress import SOURCE,SOURCE_SHA,PROFILE,prepare_progress_state as prepare_motion_state,progress_config as motion_config
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites
from tools.review_striking_credit_horizon import full_profiles

PPO_BEFORE=3212544
EVAL_BEFORE=2140974
MAX_PPO=2097152
MAX_EVAL=60000
OFFSET=3129344
WORLD_OFFSET=2506752
ARCHIVE=Path('docs/results/striking-pilot/contact-progress-v15')
CHECKPOINTS=Path('checkpoints/striking-validation/contact-progress-v15')
SEEDS=[7,23,31,47]
FRESH=[34000003,34000004,34000005,34000006]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_cycle(e):
    # Independent acceptance conjunction; either learned hand may execute a jab.
    return (e['hand'] in (0,1) and e['onset_relative_speed']>=1. and e['onset_forward_cosine']>=.7
        and e['pre_max_displacement_m']>=.12 and e['post_window_complete']
        and e['return_toward_pre_window_position_m'] is not None
        and e['return_toward_pre_window_position_m']>=.06 and e['post_contact_seconds']<=.10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--git-executable',default='git');parser.add_argument('--git-worktree')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    git_command=[args.git_executable]+(['-C',args.git_worktree] if args.git_worktree else [])
    assert os.environ['JAX_PLATFORMS']=='cpu'
    for identity in ('user.name','user.email'):
        value=subprocess.check_output(git_command+['config','--get',identity],text=True).strip()
        if not value:raise ValueError('configured Git author required before any new physical evaluation')
    with ExitStack() as stack:
        for path in (root/'.supervisor.lock',Path('runs/striking_diagnostic_supervisor.lock')):
            h=stack.enter_context(path.open('a'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Trainer owns this lock during collection; parent owns it for evaluations.
        with Path('runs/unassisted_boxing.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'audit.json';training=root/'training';cp=training/'checkpoint'
        source,manifest=load_checked(Path(SOURCE),SOURCE_SHA);require_ppo_checkpoint(source)
        prerequisites=verify_prerequisites()
        previous=Path('docs/results/striking-pilot/motion-cycle-v14/completed-review.json')
        assert json.loads(previous.read_text())['formal_world_controls_total']==EVAL_BEFORE
        if path.exists():
            if not args.resume:raise ValueError('existing job requires inspected ordinary --resume; never reinitialize')
            state=json.loads(path.read_text())
            assert state['source_sha256']==SOURCE_SHA and state['profile']==PROFILE and state['prerequisites']==prerequisites
            assert state['maximum_new_ppo_steps']==MAX_PPO and state['maximum_evaluation_controls']==MAX_EVAL
            assert state['source_final_review_sha256']==sha(previous)
        else:
            if args.resume:raise ValueError('no existing state to resume')
            prepared=prepare_motion_state(source,SOURCE_SHA)
            config=motion_config(json.loads((Path(SOURCE)/'config.json').read_text()))
            cp.mkdir(parents=True,exist_ok=False)
            payload=pickle.dumps(jax.device_get(prepared),protocol=pickle.HIGHEST_PROTOCOL)
            atomic_bytes(cp/'initial.pkl',payload)
            initial=dict(manifest,file='initial.pkl',sha256=hashlib.sha256(payload).hexdigest(),
                retention_loss_weight=0.,contact_progress_ppo=True,contact_progress_profile=PROFILE)
            atomic_json(cp/'latest.json',initial);atomic_json(training/'config.json',config)
            atomic_json(training/'progress.json',dict(consumed_steps=OFFSET,optimized_steps=OFFSET,
                world_control_steps=WORLD_OFFSET,neutral_credits_by_corner=source['neutral_credits_by_corner'],
                mixed_credits_by_mode_and_corner=source['mixed_credits_by_mode_and_corner']))
            state=dict(status='running',phase='initializing',source_sha256=SOURCE_SHA,profile=PROFILE,
                source_final_review_sha256=sha(previous),prerequisites=prerequisites,diagnostic_output=str(root),
                maximum_new_ppo_steps=MAX_PPO,maximum_evaluation_controls=MAX_EVAL,
                diagnostic_ppo_before_job=PPO_BEFORE,diagnostic_ppo_ceiling=PPO_BEFORE+MAX_PPO,
                formal_world_controls_before_job=EVAL_BEFORE,formal_world_controls_ceiling=EVAL_BEFORE+MAX_EVAL,
                main_project_consumed_retained=476041216,main_boxing_retained=1048576,
                cloned_queries_retained=6349,surface_queries_retained=960,render_forwards_retained=2065,
                standalone_bc_updates_retained=8000,optimizer_reset=False,assistance=False,main_adoption=False,
                started_utc=utc(),gates=[])
        state.update(status='running',pid=os.getpid(),updated_utc=utc());atomic_json(path,state)
        child=[None]
        def stop(signum,frame):
            if child[0] is not None and child[0].poll() is None:os.killpg(child[0].pid,signal.SIGTERM)
            raise InterruptedError('supervisor stopped; inspect ledgers and resume without rollback')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

        def account():
            progress=json.loads((training/'progress.json').read_text())
            used=progress['consumed_steps']-OFFSET
            controls=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in (root/'gates').rglob('*-progress.json') if p.parent.name!='baseline')
            assert 0<=used<=MAX_PPO and 0<=controls<=MAX_EVAL
            state.update(new_ppo_steps=used,new_collection_world_controls=progress['world_control_steps']-WORLD_OFFSET,
                optimized_new_steps=progress['optimized_steps']-OFFSET,diagnostic_ppo_total=PPO_BEFORE+used,
                diagnostic_evaluation_controls=controls,formal_world_controls_total=EVAL_BEFORE+controls,updated_utc=utc())
            atomic_json(path,state)

        def run(command,out,err):
            with out.open('w') as stdout,err.open('w') as stderr:
                child[0]=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
                state.update(active_child_pid=child[0].pid);atomic_json(path,state)
                code=child[0].wait();child[0]=None;account()
                if code:raise RuntimeError(f'child exited {code}; inspect {err}')
                if err.read_text().strip():raise RuntimeError(f'unexpected native stderr: {err}')

        def snapshot(offset,label):
            filename='initial.pkl' if offset==0 else f'step-{OFFSET+offset}.pkl'
            payload=(cp/filename).read_bytes();saved=pickle.loads(payload);require_ppo_checkpoint(saved)
            assert saved['consumed_steps']==saved['optimized_steps']==OFFSET+offset
            assert saved['world_control_steps']==WORLD_OFFSET+offset
            assert saved['contact_progress_transition']['source_sha256']==SOURCE_SHA
            location=CHECKPOINTS/label;location.mkdir(parents=True,exist_ok=True)
            atomic_bytes(location/'policy.pkl',payload)
            base=json.loads((cp/'latest.json').read_text())
            atomic_json(location/'latest.json',dict(base,file='policy.pkl',sha256=hashlib.sha256(payload).hexdigest(),
                consumed_steps=saved['consumed_steps'],optimized_steps=saved['optimized_steps'],world_control_steps=saved['world_control_steps']))
            atomic_json(location/'config.json',json.loads((training/'config.json').read_text()))
            return location,hashlib.sha256(payload).hexdigest()

        def publish(label):
            account();ARCHIVE.mkdir(parents=True,exist_ok=True)
            for folder in (root/'gates').iterdir():
                if folder.is_dir() and (folder/'motion-gate.json').exists():shutil.copytree(folder,ARCHIVE/'gates'/folder.name,dirs_exist_ok=True)
            atomic_json(ARCHIVE/'job-state.json',state)
            atomic_json(ARCHIVE/'training-progress.json',json.loads((training/'progress.json').read_text()))
            staged=subprocess.check_output(git_command+['diff','--cached','--name-only'],text=True).splitlines()
            prefixes=(str(ARCHIVE).replace('\\','/')+'/',str(CHECKPOINTS).replace('\\','/')+'/')
            if any(not name.startswith(prefixes) for name in staged):raise RuntimeError('unrelated staged changes block automatic publication')
            subprocess.run(git_command+['add','--',str(ARCHIVE),str(CHECKPOINTS)],check=True)
            logs=[str(p) for p in ARCHIVE.rglob('*-errors.log')]
            if logs:subprocess.run(git_command+['add','-f','--',*logs],check=True)
            if subprocess.run(git_command+['diff','--cached','--quiet']).returncode:
                subprocess.run(git_command+['commit','-m',f'Record contact-progress PPO {label}'],check=True)
                subprocess.run(git_command+['push'],check=True)

        def evaluate(offset,label,seeds):
            if any(g['label']==label for g in state['gates']):return
            checkpoint,digest=snapshot(offset,label);destination=root/'gates'/label;destination.mkdir(parents=True,exist_ok=True)
            account();assert state['diagnostic_evaluation_controls']+12000<=MAX_EVAL
            with Path('runs/unassisted_boxing.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                for role in (0,1,None):
                    name='selfplay' if role is None else f'neutral-{role}'
                    if (destination/(name+'.json')).exists():continue
                    if (destination/(name+'-progress.json')).exists():
                        raise RuntimeError('partial evaluation requires consumed-attempt review before retry; ledger must not be overwritten')
                    state.update(phase='independent_evaluation',active_gate=label,active_role=name);atomic_json(path,state)
                    cmd=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(checkpoint),
                        '--seeds',*[str(s) for s in seeds],'--output',str(destination/(name+'.json')),
                        '--progress',str(destination/(name+'-progress.json')),
                        '--punch-trace-directory',str(destination/name/'native'),
                        '--trace-directory',str(destination/name/'full')]
                    if role is not None:cmd+=['--neutral-partner',str(role)]
                    run(cmd,destination/(name+'.log'),destination/(name+'-errors.log'))
            original,controls=full_profiles(destination,seeds,digest);motion={};all_events=[]
            for role in (0,1,None):
                name='selfplay' if role is None else f'neutral-{role}'
                report=json.loads((destination/(name+'.json')).read_text());valid_seeds=set();reciprocal=set();total=np.zeros(2,int)
                for row in report['trials']:
                    native=Path(row['punch_trace_file']);assert sha(native)==row['punch_trace_sha256']
                    with np.load(native) as raw:
                        assert all(np.isfinite(raw[k]).all() for k in raw.files)
                        events=describe_events(raw);assert events==row['punch_measurements']
                    counts=np.zeros(2,int)
                    for e in events:
                        accepted=independent_cycle(e);all_events.append(dict(profile=name,seed=row['seed'],swap=row['swap'],**e,valid_cycle=accepted))
                        if accepted:counts[e['corner']]+=1
                    total+=counts
                    if row['survived']:
                        if role is not None and counts[role]>0:valid_seeds.add(row['seed'])
                        if role is None and (counts>0).all():reciprocal.add(row['seed'])
                motion[name]=dict(valid_cycles_by_corner=total.tolist(),own_surviving_valid_seeds=sorted(valid_seeds),
                    reciprocal_surviving_valid_seeds=sorted(reciprocal),passes=len(reciprocal if role is None else valid_seeds)>=2)
            result=dict(label=label,new_ppo_steps=offset,sha256=digest,seeds=seeds,controls=controls,
                original_contact_gate=original,motion=motion,events=all_events,
                passes_complete=original['passes_complete'] and all(m['passes'] for m in motion.values()),
                assistance=False,main_adoption=False,reviewed_utc=utc())
            atomic_json(destination/'motion-gate.json',result);state['gates'].append(result)
            assert verify_prerequisites()==prerequisites
            publish(label)

        def train(target,integration=False):
            progress=json.loads((training/'progress.json').read_text())
            if progress['optimized_steps']>=OFFSET+target:return
            if progress['consumed_steps']!=progress['optimized_steps']:
                raise RuntimeError('unoptimized consumed collection requires explicit accounting review before continuation')
            state.update(phase='ppo_training',target_new_ppo_steps=target,active_training_output=str(training));atomic_json(path,state)
            cmd=[sys.executable,'-u','-m','boxer.striking_train','--output',str(training),'--resume',
                '--contact-ppo','--demonstration-ppo','--neutral-contact-ppo','--credit-horizon-ppo',
                '--mixed-partner-ppo','--fixed-opponent-ppo','--long-gae-ppo','--motion-cycle-ppo','--contact-progress-ppo',
                '--training-gamma','.999','--retention-loss-weight','0','--ceiling',str(OFFSET+target),
                '--worlds','8','--unroll','128','--seed','53']
            if integration:cmd+=['--update-audit-directory',str(training/'integration-update-audit')]
            run(cmd,training/f'train-{target}.log',training/f'train-{target}-errors.log')
            latest=json.loads((cp/'latest.json').read_text());saved,_=load_checked(cp,latest['sha256']);require_ppo_checkpoint(saved)
            assert saved['consumed_steps']==saved['optimized_steps']==OFFSET+target
            assert int(saved['optimizer']['actor_adam'][0].count)==31368+target//1024*16
            assert int(saved['optimizer']['critic_adam'][0].count)==55040+target//1024*16
            if integration:
                audit=json.loads((training/'integration-update-audit/minibatch-measurements.json').read_text())
                assert len(audit['records'])==18 and audit['records'][0]['profiles']['all']['ratio_clip_fraction']==0.
                state['integration_verified']=True;atomic_json(path,state)

        try:
            snapshot(0,'initial-v15')
            if not any(g['label']=='baseline' for g in state['gates']):
                source_gate=Path('docs/results/striking-pilot/motion-cycle-v14/gates/step-1048576')
                old=json.loads((source_gate/'motion-gate.json').read_text())
                assert old['sha256']==SOURCE_SHA
                original,_=full_profiles(source_gate,SEEDS,SOURCE_SHA)
                assert original==old['original_contact_gate']
                baseline=dict(old,label='baseline',new_ppo_steps=0,historical_reference=str(source_gate),
                    historical_gate_sha256=sha(source_gate/'motion-gate.json'),new_evaluation_controls=0,
                    source_policy_equivalence='Transition adds reward metadata only; all prior saved fields remain exact.')
                destination=root/'gates/baseline';shutil.copytree(source_gate,destination,dirs_exist_ok=True)
                atomic_json(destination/'motion-gate.json',baseline);state['gates'].append(baseline)
                state['historical_baseline_reused']=True
                publish('historical-baseline')
            train(1024,integration=True)
            for target in (524288,1048576,1572864,2097152):
                train(target);evaluate(target,f'step-{target}',SEEDS)
                trained=[g for g in state['gates'] if g['label'].startswith('step-')]
                if len(trained)>=2 and all(g['passes_complete'] for g in trained[-2:]):break
            completed=json.loads((training/'progress.json').read_text())['optimized_steps']-OFFSET
            evaluate(completed,'fresh-final',FRESH)
            state.update(status='completed',phase='awaiting_interpretation',finished_utc=utc());publish('final')
        except BaseException as error:
            state.update(status='failed',phase='requires_investigation',error=str(error),updated_utc=utc())
            account();atomic_json(path,state);raise


if __name__=='__main__':main()
