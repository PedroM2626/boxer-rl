"""Rebuild a run's environment consistently for training, scoring and video."""
from __future__ import annotations

from boxer.warp_env import WarpBoxing


def make_env(config, nworld, **overrides):
    kind = config.get("environment", "direct")
    if kind not in ("direct", "skills"):
        raise ValueError(f"unknown environment {kind!r}")
    keys = ("n_frames", "spawn_distance", "action_scale", "round_steps",
            "damage_cap", "stun_steps", "guard_pose")
    kwargs = {k: config[k] for k in keys if k in config}
    kwargs.update(overrides)
    kwargs["nworld"] = nworld
    if kind == "skills":
        from boxer.skills_env import SkillsBoxing
        return SkillsBoxing(**kwargs)
    return WarpBoxing(**kwargs)
