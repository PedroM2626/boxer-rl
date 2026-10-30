"""Small durable attempted-step frontier with a separate full audit snapshot.

Hot progress never substitutes for checkpoint provenance or native child
ledgers. A crash leaving unoptimized attempted data must require investigation,
never a rewind to the older full audit file.
"""
import copy
from boxer.checkpoint_io import atomic_json

IDENTITY = ('source_sha256', 'maximum_new_ppo_steps', 'diagnostic_ppo_before',
            'diagnostic_ppo_ceiling', 'formal_before', 'formal_ceiling', 'main_retained')
COUNTERS = ('consumed_steps', 'optimized_steps', 'world_control_steps', 'new_ppo_steps',
            'diagnostic_ppo_total', 'formal_new', 'formal_total')
LIVE = ('status', 'phase', 'pid', 'updated_utc', 'resumes', 'engine_ledger',
        'physics_health', 'latest_sha256', 'active_gate', 'active_role',
        'active_child_pid', 'error', 'publication_error')


def hot_progress(state):
    result = {key: copy.deepcopy(state[key]) for key in (*IDENTITY, *COUNTERS, *LIVE) if key in state}
    result.update(kind='batched_gpu_hot_progress_v1', full_audit_file='state.json')
    return result


def write_progress(root, state, full=False):
    if full: atomic_json(root/'state.json', state)
    atomic_json(root/'progress.json', hot_progress(state))


def reconcile_prefix(audit, hot, checkpoint):
    """Read both durable frontiers. Reject any unoptimized or ambiguous prefix."""
    result = copy.deepcopy(audit)
    if hot is not None:
        if hot.get('kind') != 'batched_gpu_hot_progress_v1' or any(hot.get(k) != audit.get(k) for k in IDENTITY):
            raise ValueError('hot/audit job identity or ceiling changed; inspect without resetting counters')
        for key in COUNTERS:
            if key in hot: result[key] = max(audit.get(key, 0), hot[key])
        older, newer = audit.get('engine_ledger', {}), hot.get('engine_ledger', {})
        result['engine_ledger'] = {key: max(older.get(key, 0), newer.get(key, 0)) for key in older.keys() | newer.keys()}
    if (result['consumed_steps'] != result['optimized_steps']
            or result['optimized_steps'] != checkpoint['optimized_steps']
            or result['consumed_steps'] != checkpoint['consumed_steps']
            or result['world_control_steps'] != checkpoint['world_control_steps']):
        raise ValueError('attempted/unoptimized or ambiguous checkpoint frontier; retain hot counters and investigate')
    return result
