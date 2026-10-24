"""Yaw probe changes only automatic yaw and cannot certify an ordinary full gate."""
import json
from pathlib import Path
import numpy as np
import pytest
from boxer.striking_yaw_diagnostic import BaseYawDiagnosticEnv
from boxer.striking_env import StrikingEnv
from tools.review_striking_credit_horizon import full_profiles


class Pair:
    def approach_commands(self,stop_distance):return np.array([[.4,.4,.5],[-.1,-.3,-.5]])


def test_only_base_yaw_changes_before_original_action_addition_and_clipping():
    env=BaseYawDiagnosticEnv.__new__(BaseYawDiagnosticEnv);env.pair=Pair();env.engagement_distance=.45
    actions=np.arange(34,dtype=float).reshape(2,17)/50-.3;original=actions.copy()
    env.base_yaw_scale=1.
    np.testing.assert_array_equal(env.commands(actions),StrikingEnv.commands(env,actions))
    env.base_yaw_scale=.5
    expected=env.pair.approach_commands(.45);expected[:,2]*=.5;expected+=actions[:,:3]*[.15,.15,.25]
    np.testing.assert_array_equal(env.commands(actions),np.clip(expected,[-.35,-.25,-.5],[.55,.25,.5]))
    np.testing.assert_array_equal(actions,original)
    np.testing.assert_array_equal(env.commands(actions)[:,:2],StrikingEnv.commands(env,actions)[:,:2])


def test_modified_yaw_report_is_rejected_by_unchanged_promotion_gate(tmp_path):
    source=Path('docs/results/striking-pilot/mixed-confirm-1851392/final-fresh')
    for fn in ('neutral-0.json','neutral-0-progress.json'):
        (tmp_path/fn).write_bytes((source/fn).read_bytes())
    p=tmp_path/'neutral-0.json';r=json.loads(p.read_text());r['diagnostic_base_yaw_scale']=.5;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError,match='diagnostic'):
        full_profiles(tmp_path,list(range(16000003,16000007)),r['striking_sha256'])
