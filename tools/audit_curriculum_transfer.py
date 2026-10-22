"""Read-only CPU audit of the curriculum contract against the raw boxing scene.

This does not transplant a policy, run a learner, or certify physical transfer.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.scene import build_model, stance, SIDES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/curriculum/walk_easy"))
    parser.add_argument("--output", type=Path, default=Path("runs/curriculum-transfer-audit.json"))
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint / "latest.json").read_text())
    foundation = metadata["contract"]
    config_path = args.checkpoint / "config.json"
    configuration = json.loads(config_path.read_text()) if config_path.exists() else {}
    model = build_model(spawn_distance=2.0, visual=False)
    qpos, _ = stance(model, guard_pose="boxing")
    sides = {}
    for side in SIDES:
        ids = [i for i in range(model.nu) if model.actuator(i).name.startswith(side)]
        names = [model.actuator(i).name[len(side):] for i in ids]
        gains = model.actuator_gainprm[ids, 0]
        joints = model.actuator_trnid[ids, 0]
        pose = qpos[model.jnt_qposadr[joints]]
        same_order = names == foundation["actuators"]
        sides[side] = dict(actuator_count=len(ids), actuator_order_matches=same_order,
                           position_gains=gains.tolist(), default_pose=pose.tolist())
        if same_order:
            gain_delta = gains - foundation["position_gains"]
            pose_delta = pose - foundation["default_pose"]
            sides[side].update(
                max_gain_difference=float(np.max(np.abs(gain_delta))),
                max_default_pose_difference=float(np.max(np.abs(pose_delta))),
                gain_differences={n: float(d) for n, d in zip(names, gain_delta) if abs(d) > 1e-6},
                pose_differences={n: float(d) for n, d in zip(names, pose_delta) if abs(d) > 1e-6})
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    result = dict(scope="Raw composed boxing scene only; environment wrappers and policy observations are not certified",
                  foundation_checkpoint_sha256=metadata["sha256"],
                  foundation_stage_steps=metadata["stage_steps"],
                  foundation_control_dt=foundation["dt"],
                  foundation_physics_dt=configuration.get("environment", {}).get("sim_dt"),
                  foundation_action_scale=foundation["action_scale"],
                  foundation_observation_size=foundation["observation_size"],
                  boxing_physics_dt=model.opt.timestep,
                  boxing_free_joint_count=len(free), equality_constraints=model.neq,
                  gravity_compensated_bodies=int(np.count_nonzero(model.body_gravcomp)),
                  initial_free_roots=[model.qpos0[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3].tolist()
                                      for j in free],
                  explicit_contact_pairs=model.npair, sides=sides,
                  verdict="Adapter required; physical policy transfer remains unvalidated")
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
