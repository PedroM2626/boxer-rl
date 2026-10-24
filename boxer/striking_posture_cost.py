"""Training-only posture margin cost; no forces, actions or termination changes."""
import copy
import numpy as np

SOURCE_SHA='670bed08e371b30a4843a5c934ebcf16edb32a9a76147135c75ae52904b44aa5'
PROFILE=dict(pelvis_margin=.68,pelvis_fall=.55,upright_margin=.9,upright_fall=.6,weight_per_second=1.)
TRANSITION=dict(source_sha256=SOURCE_SHA,consumed_steps=1916928,world_control_steps=1294336,
    actor_count=12424,critic_count=36096,optimizer_reset=False,training_reward_only=True)


def margin_cost(height,upright,dt):
    if not all(np.isfinite(x) for x in (height,upright,dt)) or dt<=0:
        raise ValueError('finite posture and positive physical control interval required')
    h=np.clip((PROFILE['pelvis_margin']-height)/(PROFILE['pelvis_margin']-PROFILE['pelvis_fall']),0.,1.)
    u=np.clip((PROFILE['upright_margin']-upright)/(PROFILE['upright_margin']-PROFILE['upright_fall']),0.,1.)
    return float(dt*PROFILE['weight_per_second']*(h*h+u*u))


def learner_cost(env):
    pair=env.env.pair; m=pair.maps[env.learner_corner]
    return margin_cost(pair.data.qpos[m['root']+2],pair.data.sensor(m['side']+'upvector_torso').data[2],pair.dt)


def validate_posture_metadata(saved):
    if (saved.get('posture_cost_ppo') is not True or not saved.get('fixed_opponent_ppo')
            or saved.get('posture_cost_profile')!=PROFILE or saved.get('posture_cost_transition')!=TRANSITION
            or saved['consumed_steps']<TRANSITION['consumed_steps']):
        raise ValueError('explicit training-only posture cost and retained source required')


def prepare_posture_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('posture_cost_ppo') or saved['consumed_steps']!=saved['optimized_steps']
            or saved['consumed_steps']!=1916928 or saved['world_control_steps']!=1294336
            or int(saved['optimizer']['actor_adam'][0].count)!=12424
            or int(saved['optimizer']['critic_adam'][0].count)!=36096):
        raise ValueError('latest complete final required without optimizer reset or rollback')
    result=copy.deepcopy(saved)
    result.update(posture_cost_ppo=True,posture_cost_profile=copy.deepcopy(PROFILE),posture_cost_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result);return result


def posture_config(source):
    if source['version']!=8 or not source['fixed_opponent_ppo']:raise ValueError('inherited fixed-opponent profile required')
    return dict(source,version=9,posture_cost_ppo=True,posture_cost_profile=copy.deepcopy(PROFILE))
