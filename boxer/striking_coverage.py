"""Descriptive distances to successful demonstrations, never physical gates."""
import numpy as np
from scipy.spatial import cKDTree

BLOCKS = dict(motor=(0,103),opponent=(103,129),previous_action=(129,146),contact_history=(146,148))


def block_distances(reference,query):
    """Exact nearest-neighbor RMS in each already-normalized input block."""
    result={}
    for name,(start,end) in BLOCKS.items():
        distance,_=cKDTree(reference[:,start:end]).query(query[:,start:end],k=1,workers=1)
        result[name]=distance/np.sqrt(end-start)
    return result


def leave_trajectory_out_baselines(data,roles,trajectory_ids):
    """Avoid self-neighbor zeros and adjacent states from the same trajectory."""
    distances={name:np.empty(len(data)) for name in BLOCKS}
    for trajectory in np.unique(trajectory_ids):
        query=trajectory_ids==trajectory
        role=roles[query][0]
        reference=(roles==role)&~query
        if not reference.any():raise ValueError('at least two trajectories per role required')
        for name,values in block_distances(data[reference],data[query]).items():distances[name][query]=values
    return {str(role):{name:dict(q50=float(np.quantile(values[roles==role],.5)),
        q95=float(np.quantile(values[roles==role],.95)),q99=float(np.quantile(values[roles==role],.99)))
        for name,values in distances.items()} for role in (0,1)}
