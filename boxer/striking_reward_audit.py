"""Read-only decomposition of the existing striking reward, with no new shaping."""
import numpy as np


def reward_terms(scored,fallen,progress,facing,distance,residual,dt,reward_distance,both_up):
    scored,fallen,progress,facing,residual=map(np.asarray,(scored,fallen,progress,facing,residual))
    if (any(x.shape!=(2,) for x in (scored,fallen,progress,facing)) or residual.shape!=(2,14) or
            not all(np.isfinite(x).all() for x in (scored,fallen,progress,facing,residual)) or
            not all(np.isfinite(x) and x>0 for x in (dt,reward_distance)) or not np.isfinite(distance)):
        raise ValueError('finite original two-corner reward inputs required')
    terms=dict(scored=2.*scored,received=-2.*scored[::-1],fall=-5.*fallen.astype(float))
    terms.update(progress=.5*progress if both_up else np.zeros(2),
        facing=dt*.1*facing if both_up else np.zeros(2),
        engagement=np.full(2,dt*.15*np.exp(-((distance-reward_distance)/.2)**2)) if both_up else np.zeros(2),
        residual=-dt*.03*np.mean(residual**2,axis=1) if both_up else np.zeros(2))
    return terms
