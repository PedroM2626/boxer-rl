"""Explicit stationary active-opponent task with immutable archived policy provenance."""
import copy,hashlib,json
import jax,numpy as np
from boxer.striking_mixed_partner import ContactMixedStrikingEnv

SOURCE_SHA='6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a'
FROZEN_FINGERPRINT='158d7b3dbdccc0a0aadec27d7d39b714677191d35b1d8a68759f12e5472873b0'


class ContactFixedMixedStrikingEnv(ContactMixedStrikingEnv):
    opponent_mode='mixed_frozen_mean'


def fingerprint(frozen):
    digest=hashlib.sha256()
    for path,x in jax.tree_util.tree_flatten_with_path(frozen)[0]:
        a=np.asarray(x)
        if not np.isfinite(a).all():raise ValueError('finite fixed opponent required')
        digest.update(json.dumps([str(path),a.dtype.str,list(a.shape)]).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def validate_fixed_metadata(saved):
    transition=dict(source_sha256=SOURCE_SHA,consumed_steps=1859584,world_control_steps=1236992,actor_count=11528,critic_count=35200,optimizer_reset=False,opponent_policy='frozen_source_mean')
    if (saved.get('fixed_opponent_ppo') is not True or not saved.get('mixed_partner_ppo') or saved.get('conservative_actor_ppo')
            or saved.get('opponent_mode')!=('mixed_pool_mean' if saved.get('pool_opponent_ppo') else 'mixed_frozen_mean') or saved.get('fixed_opponent_transition')!=transition
            or saved['consumed_steps']<1859584 or saved['optimized_steps']<1859584
            or saved.get('fixed_opponent_fingerprint')!=FROZEN_FINGERPRINT
            or fingerprint(saved.get('frozen_opponent',{}))!=FROZEN_FINGERPRINT):
        raise ValueError('explicit fixed-source opponent and retained optimizer prefixes required')


def prepare_fixed_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('fixed_opponent_ppo') or saved.get('conservative_actor_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=1859584
            or saved['world_control_steps']!=1236992 or int(saved['optimizer']['actor_adam'][0].count)!=11528
            or int(saved['optimizer']['critic_adam'][0].count)!=35200):
        raise ValueError('latest unchanged full final required for fixed-opponent transition')
    result=copy.deepcopy(saved);frozen=copy.deepcopy(dict(actor=saved['actor'],norm=saved['norm']))
    assert fingerprint(frozen)==FROZEN_FINGERPRINT
    result.update(fixed_opponent_ppo=True,opponent_mode='mixed_frozen_mean',frozen_opponent=frozen,
        fixed_opponent_fingerprint=FROZEN_FINGERPRINT,fixed_opponent_transition=dict(source_sha256=sha,
            consumed_steps=1859584,world_control_steps=1236992,actor_count=11528,critic_count=35200,
            optimizer_reset=False,opponent_policy='frozen_source_mean'))
    require_ppo_checkpoint(result);return result


def fixed_config(source):
    if source['version']!=6 or source['opponent_mode']!='mixed_current_mean':raise ValueError('original version-6 profile required')
    return dict(source,version=8,fixed_opponent_ppo=True,opponent_mode='mixed_frozen_mean',fixed_opponent_fingerprint=FROZEN_FINGERPRINT)


def opponent_state(saved,actor,norm):
    """Only the active adversary uses frozen parameters; learner/critic remain trainable."""
    if saved.get('fixed_opponent_ppo'):
        return saved['frozen_opponent']['actor'],saved['frozen_opponent']['norm']
    return actor,norm
