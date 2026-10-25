"""Read-only signed geometry with explicit budgets; not physical impact evidence."""
import mujoco
import numpy as np
from boxer.scene import UPPER_TARGET_GEOMS


class SurfaceBudget:
    def __init__(self,forwards,distances,record=None):
        self.forward_limit=forwards;self.distance_limit=distances
        self.forwards=0;self.distances=0;self.record=record

    def consume(self,kind):
        attr,limit=('forwards',self.forward_limit) if kind=='forward' else ('distances',self.distance_limit)
        if kind not in ('forward','distance'):raise ValueError('unknown geometry query')
        if getattr(self,attr)>=limit:raise RuntimeError('surface audit query ceiling reached')
        setattr(self,attr,getattr(self,attr)+1)
        if self.record:self.record(dict(cloned_forward_evaluations=self.forwards,surface_distance_queries=self.distances))


def signed_gap(model,data,first,second,budget):
    budget.consume('distance');segment=np.zeros(6)
    gap=float(mujoco.mj_geomDistance(model,data,int(first),int(second),5.,segment))
    if not np.isfinite(gap) or not np.isfinite(segment).all():raise FloatingPointError('nonfinite surface query')
    if gap>=5.:raise ValueError('capped or unsupported surface distance; not an exact gap')
    return dict(signed_gap_m=gap,nearest_segment=segment.tolist(),censored=False)


def pose_surfaces(pair,qpos,corner,budget):
    qpos=np.asarray(qpos)
    if qpos.shape!=(pair.model.nq,) or not np.isfinite(qpos).all():raise ValueError('finite source qpos required')
    clone=mujoco.MjData(pair.model);clone.qpos[:]=qpos
    budget.consume('forward');mujoco.mj_kinematics(pair.model,clone);mujoco.mj_comPos(pair.model,clone)
    fist=int(pair.fists[corner,0]);other=pair.maps[1-corner]['side'];rows=[]
    pairs={
        frozenset((int(a),int(b))) for a,b in zip(pair.model.pair_geom1,pair.model.pair_geom2)}
    for name in UPPER_TARGET_GEOMS:
        target=pair.model.geom(other+name).id
        if frozenset((fist,target)) not in pairs:raise ValueError('surface pair is absent from the physical model')
        rows.append(dict(target=other+name,legal=name in ('head_collision','torso_collision'),
            collision_pair_declared=True,**signed_gap(pair.model,clone,fist,target,budget)))
    return rows
