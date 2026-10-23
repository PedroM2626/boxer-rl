"""Verified motor-preserving candidate data and explicit BC continuation checks."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_anchor_candidates():
    folder=Path('docs/results/striking-pilot/precontact-anchors-bc4000')
    manifest=json.loads((folder/'dataset.json').read_text())
    payload=(folder/manifest['file']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['sha256'] or not manifest['contexts_deduplicated']:
        raise ValueError('anchor integrity mismatch')
    with np.load(folder/manifest['file']) as archive:data={k:archive[k] for k in archive.files}
    n=len(data['observation'])
    if (data['observation'].shape!=(n,148) or data['action'].shape!=(n,17) or n!=manifest['rows'] or
            data['observation'].dtype!=np.float32 or data['action'].dtype!=np.float32 or
            not all(np.isfinite(x).all() for x in data.values()) or np.any(data['action']) or
            np.any(data['observation'][:,146:]) or set(data['role'])!={0,1} or
            not set(data['seed']).issubset(set(range(3000003,3000011)))):
        raise ValueError('invalid precontact candidates')
    if len(np.unique(data['trajectory_id']))!=manifest['unique_role_trajectories']:
        raise ValueError('deduplication accounting mismatch')
    keys=[]
    for trajectory in np.unique(data['trajectory_id']):
        selected=data['trajectory_id']==trajectory
        if selected.sum()!=50:raise ValueError('complete fifty-control role trajectory required')
        values=[]
        for name in ('seed','swap','role'):
            unique=np.unique(data[name][selected])
            if len(unique)!=1:raise ValueError('mixed trajectory identity')
            values.append(unique[0].item())
        keys.append(tuple(values))
    if len(set(keys))!=len(keys):raise ValueError('duplicate physical role prefixes')
    return data,manifest


def require_anchor_continuation(saved):
    if (saved.get('training_kind')!='supervised_contact_state' or saved.get('supervised_updates')!=4000 or
            saved.get('actor_inputs')!=148 or saved.get('actions')!=17 or
            saved.get('ppo_optimizer_compatible') is not False or 'sampler_state' not in saved or
            int(saved['optimizer'][1][0].count)!=4000):
        raise ValueError('latest BC actor, optimizer and sampler required for anchoring')


def mixed_indices(rng,demo_rows,anchor_rows):
    return rng.integers(0,demo_rows,192),rng.integers(0,anchor_rows,64)
