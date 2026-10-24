"""Evaluation-only automatic approach-yaw probe; no learned action is masked."""
import numpy as np
from boxer.striking_contact import ContactAwareStrikingEnv


class BaseYawDiagnosticEnv(ContactAwareStrikingEnv):
    def __init__(self,*args,base_yaw_scale=1.,**kwargs):
        if base_yaw_scale not in (1.,.5):raise ValueError('explicit diagnostic yaw scale 1 or 0.5 required')
        self.base_yaw_scale=base_yaw_scale
        super().__init__(*args,**kwargs)

    def commands(self,actions=None):
        if self.base_yaw_scale==1.:return super().commands(actions)
        command=self.pair.approach_commands(stop_distance=self.engagement_distance)
        command[:,2]*=self.base_yaw_scale
        if actions is not None:command+=actions[:,:3]*[.15,.15,.25]
        return np.clip(command,[-.35,-.25,-.5],[.55,.25,.5])
