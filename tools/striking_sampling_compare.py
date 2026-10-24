"""One bounded on-policy half-standard-deviation versus unchanged comparison."""
import argparse,hashlib,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
import jax,numpy as np
from boxer.checkpoint_io import atomic_bytes,atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_sampling_profile import prepare_sampling_state,sampling_config
from boxer.striking_fixed_opponent import fingerprint,FROZEN_FINGERPRINT
from tools.striking_objective_audit import SOURCE,SHA
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from tools.review_striking_credit_horizon import full_profiles

SEEDS=list(range(24000003,24000007))


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=args.output/'comparison.json'
        if path.exists():raise ValueError('Sampling comparison exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/objective-mean-1933312/analysis.json').read_text())
        assert review['status']=='reviewed' and review['formal_world_controls_total']==1487326 and review['diagnostic_ppo_total']==1950976
        source,manifest=load_checked(SOURCE,SHA);require_ppo_checkpoint(source)
        config=json.loads((SOURCE/'config.json').read_text());assert config['version']==8
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),source_sha256=SHA,
            source_consumed_offset=1933312,source_world_offset=1310720,maximum_new_ppo_steps=32768,new_ppo_steps_per_arm=16384,
            maximum_evaluation_world_controls=84000,diagnostic_consumed_steps=0,diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0,diagnostic_ppo_before_job=1950976,formal_evaluation_world_controls_before_job=1487326,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,new_separate_supervised_updates=0,
            unchanged_physics=True,unchanged_rewards=True,on_policy_sampling=True,assistance=False,policy_adoption='none',promotion=False,fresh_seeds=SEEDS,arms=[])
        atomic_json(path,state)
        def account():
            rows=[json.loads(p.read_text()) for p in (args.output/'half_noise/progress.json',args.output/'control/progress.json') if p.exists()]
            state['diagnostic_consumed_steps']=sum(p['consumed_steps']-1933312 for p in rows)
            state['diagnostic_collection_world_controls']=sum(p['world_control_steps']-1310720 for p in rows)
            state['diagnostic_evaluation_world_controls']=sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
            assert state['diagnostic_consumed_steps']<=32768 and state['diagnostic_evaluation_world_controls']<=84000
        try:
            state.update(phase='source_fresh_evaluation',updated_utc=utc());atomic_json(path,state)
            evaluate(SOURCE,args.output/'source-fresh',SEEDS,state,path)
            state['source_fresh'],_=full_profiles(args.output/'source-fresh',SEEDS,SHA)
            for name,cost in (('half_noise',True),('control',False)):
                folder=args.output/name;cp=folder/'checkpoint';cp.mkdir(parents=True,exist_ok=False)
                if cost:
                    initial=prepare_sampling_state(source,SHA);payload=pickle.dumps(initial,protocol=pickle.HIGHEST_PROTOCOL)
                    atomic_bytes(cp/'initial.pkl',payload)
                    atomic_json(cp/'latest.json',dict(manifest,file='initial.pkl',sha256=hashlib.sha256(payload).hexdigest(),reduced_noise_ppo=True,sampling_log_std_offset=initial['sampling_log_std_offset']))
                    atomic_json(folder/'config.json',sampling_config(config))
                    for k in source:
                        for a,b in zip(jax.tree_util.tree_leaves(source[k]),jax.tree_util.tree_leaves(initial[k])):np.testing.assert_equal(a,b)
                else:
                    for filename in (manifest['file'],'latest.json'):shutil.copy2(SOURCE/filename,cp/filename)
                    atomic_json(folder/'config.json',config)
                atomic_json(folder/'progress.json',dict(consumed_steps=1933312,optimized_steps=1933312,world_control_steps=1310720,
                    neutral_credits_by_corner=source['neutral_credits_by_corner'],mixed_credits_by_mode_and_corner=source['mixed_credits_by_mode_and_corner']))
                arm=dict(arm=name,initial_all_source_leaves_retained=True,gates=[]);state['arms'].append(arm)
                for target in (1024,8192,16384):
                    state.update(phase='training',active_arm=name,active_training_output=str(folder),updated_utc=utc());atomic_json(path,state)
                    cmd=[sys.executable,'-u','-m','boxer.striking_train','--output',str(folder),'--resume','--contact-ppo','--demonstration-ppo',
                        '--neutral-contact-ppo','--credit-horizon-ppo','--mixed-partner-ppo','--fixed-opponent-ppo','--training-gamma','.999',
                        '--retention-loss-weight','1','--ceiling',str(1933312+target),'--worlds','8','--unroll','128','--seed','53']
                    if cost:cmd+=['--reduced-noise-ppo']
                    if target==1024:cmd+=['--update-audit-directory',str(folder/'update-audit')]
                    with (folder/f'train-{target}.log').open('w') as out,(folder/f'train-{target}-errors.log').open('w') as err:subprocess.run(cmd,stdout=out,stderr=err,check=True)
                    assert not (folder/f'train-{target}-errors.log').read_text().strip();account()
                    m=json.loads((cp/'latest.json').read_text());saved,_=load_checked(cp,m['sha256']);require_ppo_checkpoint(saved)
                    assert saved['consumed_steps']==saved['optimized_steps']==1933312+target and saved['world_control_steps']==1310720+target
                    assert int(saved['optimizer']['actor_adam'][0].count)==12680+target//1024*16
                    assert int(saved['optimizer']['critic_adam'][0].count)==36352+target//1024*16
                    assert fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
                    assert saved.get('reduced_noise_ppo',False)==cost
                    np.testing.assert_array_equal(np.asarray(saved['mixed_credits_by_mode_and_corner']).sum(axis=1),[98304+target//2]*2)
                    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
                    if target==1024:
                        arm['integration_block_verified']=True
                        audit=json.loads((folder/'update-audit/minibatch-measurements.json').read_text())
                        assert len(audit['records'])==18
                        arm['initial_update_geometry']=audit['records'][-1]
                        atomic_json(path,state);continue
                    gate=folder/f'gate-{target}'/'checkpoint';gate.mkdir(parents=True,exist_ok=False)
                    for filename in (m['file'],'latest.json'):shutil.copy2(cp/filename,gate/filename)
                    shutil.copy2(folder/'config.json',gate/'config.json')
                    state.update(phase='screen_evaluation',updated_utc=utc());atomic_json(path,state)
                    evaluate(gate,folder/f'gate-{target}'/'screen',[7,23,31,47],state,path)
                    full,_=full_profiles(folder/f'gate-{target}'/'screen',[7,23,31,47],m['sha256'])
                    arm['gates'].append(dict(new_ppo_steps=target,sha256=m['sha256'],screen=full));account();atomic_json(path,state)
                    if min(p['pair_survivors'] for p in full['profiles'].values())<=3:
                        arm['early_stop_reason']='Severe survival regression: at most three paired survivors in a full profile';break
                arm['final_sha256']=m['sha256'];state.update(phase='final_fresh_evaluation',updated_utc=utc());atomic_json(path,state)
                evaluate(cp,folder/'fresh',SEEDS,state,path);arm['fresh'],_=full_profiles(folder/'fresh',SEEDS,m['sha256'])
                account();atomic_json(path,state)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            account();state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
