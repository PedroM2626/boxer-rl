"""Independent first-fall evaluation of the unassisted striking interface."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, load_policy_params, utc
from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, normalize


def arm_perturbation(noise, held, step, hold=1, scale=1.):
    """Change only arm covariance; command/body random draws remain untouched."""
    if step%hold==0:held=noise[:,3:17].copy()
    noise[:,3:17]=held*scale
    return noise,held


def ablate_action(action,zero_commands=False,zero_arms=False,arm_scale=1.):
    if not np.isfinite(arm_scale) or not 0. <= arm_scale <= 1.:
        raise ValueError('diagnostic arm scale must be finite in [0, 1]')
    if not zero_commands and not zero_arms and arm_scale==1.:return action
    result=action.copy()
    if zero_commands:result[:,:3]=0.
    if zero_arms:result[:,3:17]=0.
    else:result[:,3:17]*=arm_scale
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--opponent-checkpoint',type=Path,help='diagnostic learned opposing policy, independently normalized')
    parser.add_argument('--evaluated-corner',type=int,choices=(0,1),help='corner controlled by the main checkpoint in mixed-policy diagnostics')
    parser.add_argument('--output',type=Path,default=Path('runs/striking-validation/evaluation.json'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,31,47])
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--progress',type=Path,help='durable attempted controls for diagnostic recovery')
    parser.add_argument('--trace-directory',type=Path,help='passive observations/actions and first-contact timing')
    parser.add_argument('--punch-trace-directory',type=Path,help='passive native-substep punch kinematics; not punch certification')
    parser.add_argument('--base-yaw-scale',type=float,choices=(1.,.5),default=1.,help='diagnostic-only scale of automatic approach yaw; learned offsets remain intact')
    parser.add_argument('--sampled',action='store_true',
                        help='diagnostic saved-Gaussian policy; does not replace deterministic screening')
    parser.add_argument('--arm-noise-scale',type=float,default=1.,
                        help='sampled diagnostic only: scale arm noise, preserving command noise')
    parser.add_argument('--arm-noise-hold',type=int,default=1,
                        help='sampled diagnostic only: reuse arm perturbation across this many controls')
    parser.add_argument('--neutral-partner',type=int,choices=(0,1),metavar='LEARNER_CORNER',
                        help='diagnostic only: zero the other corner high-level commands; keep all physics')
    parser.add_argument('--zero-command-offsets',action='store_true',
                        help='diagnostic ablation: keep learned motor navigation, remove high-level command offsets')
    parser.add_argument('--zero-arm-residuals',action='store_true',
                        help='diagnostic ablation: zero only high-level fourteen arm residual outputs')
    parser.add_argument('--arm-residual-scale',type=float,default=1.,
                        help='diagnostic-only scale for both actors high-level arm outputs')
    parser.add_argument('--approach-gain',type=float,
                        help='diagnostic navigation proportional gain; original default is 0.8')
    parser.add_argument('--engagement-distance',type=float,
                        help='explicit diagnostic profile override; impact criteria remain fixed')
    args=parser.parse_args(argv)
    if (args.opponent_checkpoint is None)!=(args.evaluated_corner is None):
        raise ValueError('mixed-policy diagnostics require both opposing checkpoint and evaluated corner')
    if args.opponent_checkpoint is not None and (args.checkpoint is None or args.sampled or args.neutral_partner is not None or args.zero_command_offsets or args.zero_arm_residuals or args.arm_residual_scale!=1. or args.base_yaw_scale!=1. or args.approach_gain is not None or args.engagement_distance is not None):
        raise ValueError('mixed-policy diagnostics require deterministic unchanged contact control with no ablation')
    if not np.isfinite(args.arm_residual_scale) or not 0. <= args.arm_residual_scale <= 1.:
        raise ValueError('diagnostic arm residual scale must be finite in [0, 1]')
    if args.zero_arm_residuals and args.arm_residual_scale!=1.:
        raise ValueError('choose either zero arms or a diagnostic arm scale')
    if args.progress and args.progress.exists():raise ValueError('evaluation progress exists; inspect before rerun')
    if args.trace_directory:args.trace_directory.mkdir(parents=True,exist_ok=False)
    if args.punch_trace_directory:
        if args.checkpoint is None or args.progress is None or args.sampled:
            raise ValueError('passive punch trace requires a deterministic checkpoint and durable progress')
        args.punch_trace_directory.mkdir(parents=True,exist_ok=False)
    if args.sampled and not args.checkpoint:
        raise ValueError('sampled diagnostic requires a saved policy')
    if not np.isfinite(args.arm_noise_scale) or args.arm_noise_scale<=0 or args.arm_noise_hold<1:
        raise ValueError('positive finite arm noise scale and positive hold required')
    if not args.sampled and (args.arm_noise_scale!=1. or args.arm_noise_hold!=1):
        raise ValueError('arm noise overrides require explicit sampled diagnostic')
    if not np.isfinite(args.seconds) or args.seconds<=0:
        raise ValueError('positive finite duration required')
    foundation=UnassistedG1('approach',impl='jax')
    params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
    if contract(foundation)!=source['contract']:raise ValueError('motor contract mismatch')
    distance,reward_distance=.58,.6
    manifest=None
    config={}
    if args.checkpoint:
        manifest=json.loads((args.checkpoint/'latest.json').read_text())
        distance=manifest.get('engagement_distance',distance)
        reward_distance=manifest.get('engagement_reward_distance',reward_distance)
        config_path=args.checkpoint/'config.json'
        if not config_path.exists():config_path=args.checkpoint.parent/'config.json'
        if config_path.exists():
            config=json.loads(config_path.read_text())
            for key in ('engagement_distance','engagement_reward_distance','actor_inputs','actions','body_residual_radians','contact_history'):
                if key in manifest and key in config and manifest[key]!=config[key]:
                    raise ValueError('checkpoint and configuration engagement profiles disagree')
            distance=config.get('engagement_distance',.58)
            reward_distance=config.get('engagement_reward_distance',.6)
    if args.engagement_distance is not None:
        distance=reward_distance=args.engagement_distance
    inputs=config.get('actor_inputs',(manifest or {}).get('actor_inputs',146))
    actions=config.get('actions',(manifest or {}).get('actions',17))
    body_bound=config.get('body_residual_radians',(manifest or {}).get('body_residual_radians',0.))
    contact_history=config.get('contact_history',(manifest or {}).get('contact_history',False))
    if contact_history != ((inputs,actions)==(148,17)):
        raise ValueError('contact history and interface dimensions disagree')
    if args.base_yaw_scale!=1. and ((inputs,actions)!=(148,17) or args.checkpoint is None or args.sampled or args.zero_command_offsets or args.zero_arm_residuals):
        raise ValueError('base-yaw probe requires deterministic contact checkpoint and no other controller ablation')
    if (inputs,actions)==(146,17):
        if body_bound!=0.:raise ValueError('legacy interface cannot carry body residual authority')
        env=StrikingEnv(foundation,params,engagement_distance=distance,
                       engagement_reward_distance=reward_distance)
    elif (inputs,actions)==(148,17):
        if body_bound!=0.:raise ValueError('contact interface cannot carry body residual authority')
        from boxer.striking_contact import ContactAwareStrikingEnv
        if args.base_yaw_scale==1.:
            env=ContactAwareStrikingEnv(foundation,params,engagement_distance=distance,
                engagement_reward_distance=reward_distance)
        else:
            from boxer.striking_yaw_diagnostic import BaseYawDiagnosticEnv
            env=BaseYawDiagnosticEnv(foundation,params,engagement_distance=distance,
                engagement_reward_distance=reward_distance,base_yaw_scale=args.base_yaw_scale)
    elif (inputs,actions)==(161,32):
        from boxer.striking_body import WholeBodyStrikingEnv
        env=WholeBodyStrikingEnv(foundation,params,engagement_distance=distance,
            engagement_reward_distance=reward_distance,body_residual_radians=body_bound)
    else:
        raise ValueError('unsupported striking interface dimensions')
    if args.approach_gain is not None:
        if not np.isfinite(args.approach_gain) or args.approach_gain <= 0:
            raise ValueError('positive finite approach gain required')
        original_commands=env.pair.approach_commands
        def diagnostic_commands(stop_distance=.75):
            result=original_commands(stop_distance)
            roots=[env.pair.data.qpos[m['root']:m['root']+2] for m in env.pair.maps]
            distance=float(np.linalg.norm(roots[1]-roots[0]))
            original_speed=np.clip((distance-stop_distance)*.8,0.,.45)
            speed=np.clip((distance-stop_distance)*args.approach_gain,0.,.45)
            if original_speed>0:result[:,:2]*=speed/original_speed
            return result
        env.pair.approach_commands=diagnostic_commands
    residual_hash=None
    if args.checkpoint:
        payload=(args.checkpoint/manifest['file']).read_bytes()
        residual_hash=hashlib.sha256(payload).hexdigest()
        if residual_hash!=manifest['sha256']:raise ValueError('striking checksum mismatch')
        saved=pickle.loads(payload)
        if saved.get('training_kind')=='ppo' and saved.get('optimizer_kind')=='split_adam_v1':
            from boxer.striking_train import require_ppo_checkpoint
            require_ppo_checkpoint(saved)
        if saved.get('motion_cycle_ppo') or manifest.get('motion_cycle_ppo') or config.get('motion_cycle_ppo'):
            from boxer.striking_motion_reward import PROFILE as MOTION_PROFILE
            if (not all(x.get('motion_cycle_ppo') for x in (saved,manifest,config)) or config.get('version')!=(17 if saved.get('arm_effort_ppo') else 16 if saved.get('contact_rebalance_ppo') else 15 if saved.get('contact_progress_ppo') else 14)
                    or any(x.get('motion_reward_profile')!=MOTION_PROFILE for x in (saved,manifest,config))):
                raise ValueError('manifest, configuration and saved motion reward disagree')
        if saved.get('contact_progress_ppo') or manifest.get('contact_progress_ppo') or config.get('contact_progress_ppo'):
            from boxer.striking_contact_progress import PROFILE as PROGRESS_PROFILE
            if (not all(x.get('contact_progress_ppo') for x in (saved,manifest,config))
                    or any(x.get('contact_progress_profile')!=PROGRESS_PROFILE for x in (saved,manifest,config))):
                raise ValueError('manifest, configuration and saved contact-progress profile disagree')
        if saved.get('contact_rebalance_ppo') or manifest.get('contact_rebalance_ppo') or config.get('contact_rebalance_ppo'):
            from boxer.striking_contact_rebalance import PROFILE as REBALANCE_PROFILE
            if (not all(x.get('contact_rebalance_ppo') for x in (saved,manifest,config))
                    or any(x.get('contact_rebalance_profile')!=REBALANCE_PROFILE for x in (saved,manifest,config))):
                raise ValueError('manifest, configuration and saved contact-rebalance profile disagree')
        if saved.get('arm_effort_ppo') or manifest.get('arm_effort_ppo') or config.get('arm_effort_ppo'):
            from boxer.striking_arm_effort import PROFILE as ARM_EFFORT_PROFILE
            if (not all(x.get('arm_effort_ppo') for x in (saved,manifest,config))
                    or any(x.get('arm_effort_profile')!=ARM_EFFORT_PROFILE for x in (saved,manifest,config))):
                raise ValueError('manifest, configuration and saved arm-effort profile disagree')
        if manifest.get('kl_guard_ppo',False)!=saved.get('kl_guard_ppo',False) or manifest.get('kl_guard_target')!=saved.get('kl_guard_target'):
            raise ValueError('manifest and saved KL guard profile disagree')
        if manifest.get('long_gae_ppo',False)!=saved.get('long_gae_ppo',False) or manifest.get('training_gae_lambda')!=saved.get('training_gae_lambda'):
            raise ValueError('manifest and saved GAE profile disagree')
        if (manifest.get('pool_opponent_ppo',False)!=saved.get('pool_opponent_ppo',False)
                or (saved.get('pool_opponent_ppo') and (manifest.get('opponent_mode')!='mixed_pool_mean'
                    or config.get('version')!=13 or config.get('pool_opponent_ppo') is not True
                    or config.get('opponent_mode')!='mixed_pool_mean'))):
            raise ValueError('manifest, configuration and saved opponent pool disagree')
        for key,expected in (('actor_inputs',inputs),('actions',actions),('contact_history',contact_history)):
            if key in saved and saved[key]!=expected:raise ValueError('payload and declared interface disagree')
        if saved['foundation_sha256']!=source['sha256']:raise ValueError('striking motor mismatch')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('non-finite striking checkpoint')
        from boxer.striking_sampling_profile import sampling_offset
        offset=sampling_offset(saved)
        if manifest.get('sampling_log_std_offset',0.)!=offset or manifest.get('reduced_noise_ppo',False)!=saved.get('reduced_noise_ppo',False):
            raise ValueError('manifest and saved sampling profile disagree')
        actor=Actor(action_size=actions,sampling_log_std_offset=offset)
        policy=jax.jit(lambda observation:jp.tanh(actor.apply(saved['actor'],observation)[0]))
        distribution=jax.jit(lambda observation:actor.apply(saved['actor'],observation))
        modes=['sampled' if args.sampled else 'trained']
    else:
        modes=['neutral','exploration']
    opponent_hash=None
    if args.opponent_checkpoint is not None:
        if (inputs,actions)!=(148,17):raise ValueError('compatible contact-policy interface required')
        from boxer.striking_train import require_ppo_checkpoint
        from tools.striking_contact_compare import load_checked
        from boxer.striking_opponent_evaluation import route_opponent_action
        other_manifest=json.loads((args.opponent_checkpoint/'latest.json').read_text())
        opponent_hash=other_manifest['sha256'];opponent_saved,_=load_checked(args.opponent_checkpoint,opponent_hash)
        require_ppo_checkpoint(saved);require_ppo_checkpoint(opponent_saved)
        if opponent_saved['foundation_sha256']!=saved['foundation_sha256']:raise ValueError('opposing motor provenance mismatch')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(opponent_saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('nonfinite opposing checkpoint')
        opponent_policy=jax.jit(lambda observation:jp.tanh(actor.apply(opponent_saved['actor'],observation)[0]))
    results=[];transitions=0;frames=[];frame_times=[]
    for mode in modes:
        for seed in args.seeds:
            for swap in (False,True):
                obs=env.reset(seed,swap)
                punch_trace=None
                if args.punch_trace_directory:
                    from boxer.striking_punch_trace import PunchTrace,describe_events
                    punch_trace=PunchTrace();env.pair.diagnostic_punch_trace=punch_trace
                trace=None
                if args.trace_directory:
                    from boxer.striking_trace import StrikingTrace,terminal_details
                    trace=StrikingTrace(env)
                rng=np.random.default_rng(seed+int(swap)*1009)
                held_arm_noise=None
                hits=np.zeros(2);impulses=np.zeros(2)
                impact_events=[]
                minimum_hand_distance=env.hand_distances().copy()
                residual_activity=np.zeros(2)
                max_residual=np.zeros(2)
                for step in range(round(args.seconds/env.pair.dt)):
                    if mode=='trained':action=np.asarray(policy(jp.array(normalize(obs,saved['norm']))))
                    elif mode=='sampled':
                        mean,log_std=map(np.asarray,distribution(jp.array(normalize(obs,saved['norm']))))
                        noise=rng.normal(size=mean.shape)
                        # Draw every control in every profile to retain paired command
                        # noise streams. Only the fourteen arm noise channels vary.
                        noise,held_arm_noise=arm_perturbation(noise,held_arm_noise,step,
                            args.arm_noise_hold,args.arm_noise_scale)
                        action=np.tanh(mean+np.exp(log_std)*noise)
                    elif mode=='exploration':action=np.tanh(rng.normal(0.,np.exp(-1.9),(2,actions)))
                    else:action=np.zeros((2,actions))
                    if opponent_hash is not None:
                        other=1-args.evaluated_corner
                        opposing_action=np.asarray(opponent_policy(jp.array(normalize(obs,opponent_saved['norm']))))[other]
                        action=route_opponent_action(action,opposing_action,args.evaluated_corner)
                    if args.neutral_partner is not None:
                        action=action.copy()
                        action[1-args.neutral_partner]=0.
                    action=ablate_action(action,args.zero_command_offsets,args.zero_arm_residuals,args.arm_residual_scale)
                    previous_scores=env.pair.scorecard.last_score.copy()
                    if trace:trace.before(env,obs,action)
                    transitions+=1
                    if args.progress:
                        atomic_json(args.progress,dict(status='running',simulated_world_transitions=transitions,
                            seed=seed,swap=swap,control_step=step,striking_sha256=residual_hash,updated_utc=utc()))
                    obs,reward,done,outcome=env.step(action)
                    if trace:trace.after(env,obs,outcome)
                    hits+=outcome['scored'];impulses+=outcome['legal_contact_impulse']
                    if np.any(outcome['scored']):
                        recent=env.pair.scorecard.last_score > previous_scores
                        hands=[dict(corner=i,hand=h,
                                    score_time=float(env.pair.scorecard.last_score[i,h]),
                                    contact_window_impulse=float(env.pair.scorecard.impulse[i,h]))
                               for i in range(2) for h in range(2) if recent[i,h]]
                        impact_events.append(dict(control_step=step,elapsed=env.pair.elapsed,
                            scored_by_corner=outcome['scored'].tolist(),hands=hands,
                            fallen=outcome['fallen'].tolist()))
                    minimum_hand_distance=np.minimum(minimum_hand_distance,env.hand_distances())
                    residual_activity+=np.mean(np.abs(outcome['arm_residual']),axis=1)
                    max_residual=np.maximum(max_residual,np.max(np.abs(outcome['arm_residual']),axis=1))
                    if args.video and mode==modes[0] and seed==args.seeds[0] and not swap and step%2==0:
                        frames.append(env.pair.data.qpos.copy())
                        frame_times.append(env.pair.elapsed)
                    if done:break
                result=dict(mode=mode,seed=seed,swap=swap,survived=not env.pair.done,
                    elapsed=env.pair.elapsed,final_distance=outcome['distance'],
                    scored_by_corner=hits.tolist(),raw_legal_impulse=impulses.tolist(),
                    fallen=outcome['fallen'].tolist(),finite=bool(np.isfinite(obs).all()),
                    minimum_hand_target_distance=minimum_hand_distance.tolist(),
                    mean_abs_arm_residual=(residual_activity/(step+1)).tolist(),
                    max_abs_arm_residual=max_residual.tolist())
                result['impact_events']=impact_events
                if punch_trace is not None:
                    punch_file=args.punch_trace_directory/f'{mode}-seed-{seed}-swap-{int(swap)}.npz'
                    punch_trace.save(punch_file)
                    result.update(punch_trace_file=str(punch_file),
                        punch_trace_sha256=hashlib.sha256(punch_file.read_bytes()).hexdigest(),
                        punch_measurements=describe_events(punch_trace.arrays()))
                if trace:
                    trace_file=args.trace_directory/f'{mode}-seed-{seed}-swap-{int(swap)}.npz'
                    result['first_observed_contacts']=trace.save(trace_file,env)
                    result['trace_file']=str(trace_file)
                    result['trace_sha256']=hashlib.sha256(trace_file.read_bytes()).hexdigest()
                    result['first_fall_time']=env.pair.elapsed if env.pair.done else None
                    result['terminal_state']=terminal_details(env)
                results.append(result);print(json.dumps(result),flush=True)
    if frames:
        from tools.evaluate_paired_foundation import render
        caption=(f'Diagnostic body {body_bound:.2f} rad | {"Sampled" if args.sampled else "Mean"} policy | '
                 f'{manifest["consumed_steps"]:,} lineage steps') if manifest else 'Unassisted striking interface | Evaluation'
        render(env.pair,frames,args.video,caption=caption,frame_times=frame_times,
               impact_events=results[0]['impact_events'])
    atomic_json(args.output,dict(kind='striking_interface_evaluation',updated_utc=utc(),
        assistance=False,source_sha256=source['sha256'],striking_sha256=residual_hash,
        opponent_sha256=opponent_hash,evaluated_corner=args.evaluated_corner,
        diagnostic_base_yaw_scale=args.base_yaw_scale,
        simulated_world_transitions=transitions,optimization_transitions=0,
        scorecard_version=1,episode_seconds=args.seconds,trials=results,
        engagement_distance=distance,engagement_reward_distance=reward_distance,
        actor_inputs=inputs,actions=actions,body_residual_radians=body_bound,contact_history=contact_history,
        sampled_policy=args.sampled,
        policy_sampling_log_std_offset=offset if args.checkpoint else 0.,
        diagnostic_arm_noise_scale=args.arm_noise_scale,
        diagnostic_arm_noise_hold=args.arm_noise_hold,
        neutral_partner_learner_corner=args.neutral_partner,
        diagnostic_zero_command_offsets=args.zero_command_offsets,
        diagnostic_zero_arm_residuals=args.zero_arm_residuals,
        diagnostic_arm_residual_scale=args.arm_residual_scale,
        diagnostic_approach_gain=args.approach_gain,
        diagnostic_profile_override=args.engagement_distance is not None,
        certification=False))
    if args.progress:
        atomic_json(args.progress,dict(status='completed',simulated_world_transitions=transitions,
            striking_sha256=residual_hash,updated_utc=utc()))


if __name__=='__main__':
    main()
