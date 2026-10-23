"""Passive control observations and native first-contact diagnostics."""
import io

import numpy as np

from boxer.checkpoint_io import atomic_bytes


class StrikingTrace:
    def __init__(self, env):
        self.rows=[]
        env.pair.diagnostic_contact_times={}

    def before(self, env, observation, action):
        self.pending=dict(observation=observation.copy(),action=action.copy(),
            time_before=env.pair.elapsed,commands=env.commands(action).copy())

    def after(self, env, observation, outcome):
        pair=env.pair
        self.pending.update(next_observation=observation.copy(),qpos=pair.data.qpos.copy(),
            qvel=pair.data.qvel.copy(),time_after=pair.elapsed,
            applied_actions=outcome['applied_actions'].copy(),arm_residual=outcome['arm_residual'].copy(),
            scored=outcome['scored'].copy(),fallen=outcome['fallen'].copy(),
            hand_distance=env.hand_distances().copy(),
            pelvis_height=np.array([pair.data.qpos[m['root']+2] for m in pair.maps]),
            upright=np.array([pair.data.sensor(m['side']+'upvector_torso').data[2] for m in pair.maps]))
        self.rows.append(self.pending);del self.pending

    def save(self, path, env):
        arrays={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise FloatingPointError('nonfinite diagnostic trace')
        data=io.BytesIO();np.savez_compressed(data,**arrays);atomic_bytes(path,data.getvalue())
        return [dict(geoms=[env.pair.model.geom(g).name for g in geoms],elapsed=time)
            for geoms,time in sorted(env.pair.diagnostic_contact_times.items(),key=lambda item:item[1])]


def terminal_details(env):
    pair=env.pair
    contacts=[[pair.model.geom(int(g)).name for g in contact.geom] for contact in pair.data.contact]
    corners=[]
    for m in pair.maps:
        height=float(pair.data.qpos[m['root']+2])
        upright=float(pair.data.sensor(m['side']+'upvector_torso').data[2])
        local=[{name[len(m['side']):] for name in names} for names in contacts
            if all(name.startswith(m['side']) for name in names)]
        leg_contact=any(names in ({'left_foot','right_foot'},{'left_foot','right_shin'},
            {'right_foot','left_shin'}) for names in local)
        corners.append(dict(pelvis_height=height,upright=upright,low_pelvis=height<.55,
            tilted=upright<.6,self_leg_contact=leg_contact))
    return dict(corners=corners,contact_pairs=contacts)
