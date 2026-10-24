"""Versioned per-event motion evidence, separate from legacy contact scoring."""
from types import MappingProxyType

import numpy as np

PROFILE = MappingProxyType(dict(version=1, kind='strike_motion_reference_evidence', measurement_only=True,
    speed_reference_m_s=1., forward_cosine_reference=.7,
    pre_window_seconds=.25, post_window_seconds=.5,
    legacy_scorecard_unchanged=True, punch_certification_supported=False))


def motion_evidence(event):
    """Do not borrow speed/direction across hands or imply a universal punch gate."""
    required=('onset_relative_speed','onset_forward_cosine','pre_max_displacement_m',
        'pre_path_length_m','pre_elbow_excursion_rad','event_window_impulse_ns',
        'post_window_seconds','post_contact_seconds')
    values=np.array([event[k] for k in required],dtype=float)
    if not np.isfinite(values).all() or np.any(values[[0,2,3,4,5,6,7]]<0):
        raise ValueError('finite nonnegative strike-motion measurements required')
    if abs(event['onset_forward_cosine'])>1+1e-9:
        raise ValueError('direction cosine outside physical range')
    recovered=event.get('return_toward_pre_window_position_m')
    if recovered is not None and not np.isfinite(recovered):
        raise ValueError('finite or unavailable recovery measurement required')
    speed=bool(event['onset_relative_speed']>=PROFILE['speed_reference_m_s'])
    forward=bool(event['onset_forward_cosine']>=PROFILE['forward_cosine_reference'])
    complete=event.get('post_window_complete')
    if not isinstance(complete,(bool,np.bool_)):
        raise ValueError('explicit post-window censorship required')
    return dict(profile_version=1,corner=int(event['corner']),hand=int(event['hand']),
        speed_reference_met=speed,forward_reference_met=forward,
        both_reference_components_met=speed and forward,
        pre_max_displacement_m=float(event['pre_max_displacement_m']),
        pre_path_length_m=float(event['pre_path_length_m']),
        pre_elbow_excursion_rad=float(event['pre_elbow_excursion_rad']),
        event_window_impulse_ns=float(event['event_window_impulse_ns']),
        recovery_observation_complete=bool(complete),
        return_toward_reference_m=float(recovered) if complete and recovered is not None else None,
        guard_recovery_certified=False,punch_certified=False)


def evaluate_motion(events):
    rows=[motion_evidence(e) for e in events]
    return dict(profile=dict(PROFILE),events=rows,legacy_event_count=len(rows),
        both_reference_components_count=sum(r['both_reference_components_met'] for r in rows),
        censored_recovery_count=sum(not r['recovery_observation_complete'] for r in rows),
        certified_punch_count=None,certification=False)
