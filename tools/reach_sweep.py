"""Is there any static arm pose that reaches a scoring surface?

`tools/hit_trace.py` settled that the fifth arm's 0.000 is reach: over 8000
steps the learner's strike geoms never once crossed into a scoring target's
contact margin -- 72% of those steps glove-on-glove at 0.72 m, and the rearm
counter never even reset. The next question separates two worlds that look the
same from that trace:

* the opponent's guard seals the scoring surfaces, and no arm pose of this
  robot gets around it -- then no reward written on this scene pays a punch
  against a guarding opponent, and the moves are the guard, the target set, or
  a force-based score;
* the guard leaves lanes and the policy never found them -- then the moves are
  imitation or shaping that show the arm where the lanes are.

This measures it: red's arm joints swept uniformly inside their model ranges
(plus the stance itself and the arm-at-zero pose), blue either holding the
guard or with its arms at zero, at several separations, every pose run through
the env's own ``pipeline_init`` and ``_contact_damage`` so "landed" is exactly
the quantity the gate reads. The arms-down half is the control and the preflight:
an extended arm against arms-down at close range must re-find the torso contact
that `tools/fist_probe.py` measured dynamically, or the sweep is not measuring
what it claims and stops before reporting anything.

Static is the floor, not the ceiling -- a real punch carries torso rotation and
translation on top of the arm pose, so a pose that fails statically may still
land dynamically. But a target that no pose in the whole joint box reaches
against a guard is a wall, not a skill gap.

Only the swept arm's own contact slots are counted, because the other arm stays
in guard and the guards of two boxers at fighting range are already hand-on-hand
(measured with `tools/margin_check.py`: red's guard hand sits at -13 mm inside
blue's at 0.9 m, in every pose). An unrestricted count would read the standing
guard's touch as the swept arm's, and at clinch range it would even score the
guard fist resting on the opponent's torso.

    JAX_PLATFORMS=cuda,cpu python tools/reach_sweep.py [--poses 8192]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import scene
from boxer.env import BoxingEnv

SEPARATIONS = (0.9, 0.75, 0.65, 0.55, 0.45)


def arm_map(level, side: str) -> dict:
    """Qpos addresses and ranges of one corner's arm joints, by scene name."""
    model = level._model
    addresses, ranges = {}, {}
    for half in ("left", "right"):
        for joint in scene.ARM_JOINTS:
            name = f"{side}{half}_{joint}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            addresses[f"{half}_{joint}"] = int(model.jnt_qposadr[jid])
            if model.jnt_limited[jid]:
                ranges[f"{half}_{joint}"] = (float(model.jnt_range[jid][0]),
                                              float(model.jnt_range[jid][1]))
            else:
                ranges[f"{half}_{joint}"] = (-2.5, 2.5)
    return addresses, ranges


def target_group(blue_name: str) -> str:
    bare = blue_name.split("/")[-1]
    if bare == "head_collision":
        return "head"
    if bare == "torso_collision":
        return "torso"
    return "lower"  # pelvis, hips, thighs -- the 0.35-weight band


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", type=int, default=8192,
                        help="uniform random arm poses sampled per (arm, config)")
    parser.add_argument("--chunk", type=int, default=1024,
                        help="worlds per pipeline batch; the vmapped solver needs "
                             "a few hundred MB per thousand worlds on a 6 GB card")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--separations", type=float, nargs="*", default=list(SEPARATIONS))
    args = parser.parse_args()

    level = BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    model = level._model
    base = np.asarray(level._stance_qpos).astype(np.float32).copy()
    qvel0 = jp.zeros(level.sys.nv)
    ctrl0 = level._stance_ctrl
    blue_free = level._sides[scene.BLUE]["free_adr"]

    red_adr, red_rng = arm_map(level, scene.RED)
    blue_adr, _ = arm_map(level, scene.BLUE)
    blue_arms = np.array([blue_adr[f"{h}_{j}"] for h in ("left", "right")
                          for j in scene.ARM_JOINTS])

    init_one = lambda q: level.pipeline_init(q, qvel0, None, ctrl0)
    batch_init = jax.jit(jax.vmap(init_one))

    def run_batch(qpos: np.ndarray) -> np.ndarray:
        """Every pose through the env's own init, in memory-sized chunks: one
        vmapped pipeline solve holds a few f32[worlds, 802, 70] intermediates,
        and 8k worlds of that is 1.8 GB each on a 6 GB card."""
        dists = []
        for start in range(0, len(qpos), args.chunk):
            block = qpos[start:start + args.chunk]
            dists.append(np.asarray(batch_init(jp.array(block)).contact.dist))
        return np.concatenate(dists)

    def name(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<{gid}>"

    reference = init_one(level._stance_qpos)
    slot_pairs = np.asarray(reference.contact.geom)
    slot_margin = np.asarray(reference.contact.includemargin)
    slot_landed = np.asarray(level._slot["red"]["landed"]) > 0
    slot_scoring = np.asarray(level._slot["red"]["scoring"]) > 0

    def slot_str(slot: int) -> str:
        return f"{name(slot_pairs[slot, 0])} x {name(slot_pairs[slot, 1])}"

    def blue_side(slot: int) -> str:
        pair = (name(slot_pairs[slot, 0]), name(slot_pairs[slot, 1]))
        return pair[1] if pair[1].startswith("blue") else pair[0]

    def slots_of(half: str) -> list:
        """The swept arm's own slots: this half's hand/wrist geoms on any blue
        target. Everything else stays out of the counts -- the other arm is
        parked in a guard that is already hand-on-hand with the opponent's at
        fighting range, and at clinch range that guard fist even rests on the
        opponent's torso."""
        mine = {f"red/{half}_hand_collision", f"red/{half}_wrist_collision"}
        out = []
        for slot in range(len(slot_pairs)):
            if slot_landed[slot] and {name(slot_pairs[slot, 0]),
                                      name(slot_pairs[slot, 1])} & mine:
                out.append(slot)
        return out

    # Preflight: the MJX buffer must agree with host MuJoCo on the same posed
    # qpos, because the whole sweep reads "landed" through pipeline_init. Host
    # mj_forward is the ground truth (contacts appear there with dist < 0); if
    # the two disagree, every number below measures an execution path rather
    # than geometry, so the tool stops. The probe pose is an extended left arm
    # against arms-down at 0.45 m -- and note what the host says about it, since
    # the earlier draft of this preflight asserted it lands, and that assertion
    # was wrong: the contact was the *other* arm's guard fist resting on the
    # torso at clinch range, not the extended one.
    probe = base.copy()
    probe[blue_free] -= 0.85
    probe[blue_arms] = 0.0
    probe[red_adr["left_shoulder_pitch"]] = -1.5
    probe[red_adr["left_elbow"]] = 0.0

    data = mujoco.MjData(model)
    data.qpos[:] = probe
    mujoco.mj_forward(model, data)
    host_pairs = {
        frozenset((name(data.contact[i].geom1), name(data.contact[i].geom2)))
        for i in range(data.ncon)
    }
    probe_inside = np.asarray(init_one(jp.array(probe)).contact.dist) < slot_margin
    mjx_pairs = {
        frozenset((name(slot_pairs[s, 0]), name(slot_pairs[s, 1])))
        for s in np.flatnonzero(probe_inside)
    }
    cross = sorted(
        " x ".join(sorted(pair)) for pair in host_pairs
        if len(pair) == 2 and {n.split("/")[0] for n in pair} == {"red", "blue"}
    )
    print("preflight probe: extended left arm vs arms-down at 0.45 m")
    print(f"  host cross-corner contacts: {cross or 'none'}")
    for half in ("left", "right"):
        mine = [s for s in slots_of(half) if slot_scoring[s] and probe_inside[s]]
        print(f"  its own scoring slots inside the margin: {half} {len(mine)}")
    if host_pairs != mjx_pairs:
        only_host = host_pairs - mjx_pairs
        only_mjx = mjx_pairs - host_pairs
        raise SystemExit(
            "preflight failed: the MJX contact buffer disagrees with host MuJoCo "
            f"on the same pose (host-only: {sorted(map(sorted, only_host))}, "
            f"mjx-only: {sorted(map(sorted, only_mjx))}) -- pipeline_init is not "
            "populating the buffer for a posed qpos, so the sweep would measure "
            "its own blind spot."
        )
    print("  host and MJX agree on every penetrating pair -- the buffer is live\n")

    rng = np.random.default_rng(args.seed)
    print(f"guard pose: {level.config()['guard_pose']}")
    print(f"pose space: {args.poses} uniform samples per arm inside the model ranges, "
          f"plus the stance and the arm-at-zero pose; the other arm stays in guard")
    print(f"devices: {jax.devices()}\n")

    summary = {}
    for separation in args.separations:
        gap = 1.3 - separation
        for blue_mode in ("guard", "down"):
            key = (separation, blue_mode)
            scored_counts = {}
            for half in ("left", "right"):
                adr = np.array([red_adr[f"{half}_{j}"] for j in scene.ARM_JOINTS])
                lo = np.array([red_rng[f"{half}_{j}"][0] for j in scene.ARM_JOINTS])
                hi = np.array([red_rng[f"{half}_{j}"][1] for j in scene.ARM_JOINTS])
                poses = np.vstack([
                    rng.uniform(lo, hi, (args.poses, len(scene.ARM_JOINTS))),
                    base[adr][None], np.zeros((1, len(scene.ARM_JOINTS))),
                ]).astype(np.float32)

                qpos = np.tile(base, (len(poses), 1))
                qpos[:, adr] = poses
                qpos[:, blue_free] -= gap
                if blue_mode == "down":
                    qpos[:, blue_arms] = 0.0

                dist = run_batch(qpos)
                inside = dist < slot_margin[None, :]
                arm = slots_of(half)
                my_scored = [s for s in arm if slot_scoring[s]]
                my_blockers = [s for s in arm if not slot_scoring[s]]
                scored = (inside[:, my_scored].any(axis=1)
                          if my_scored else np.zeros(len(poses), dtype=bool))

                groups = {}
                for group in ("head", "torso", "lower"):
                    slots = [s for s in my_scored
                             if target_group(blue_side(s)) == group]
                    groups[group] = int(inside[:, slots].any(axis=1).sum()) if slots else 0
                scored_counts[half] = (int(scored.sum()), len(poses), groups, poses,
                                       dist, inside, scored, my_scored, my_blockers)

            print(f"=== {separation:.2f} m apart, blue "
                  f"{'holding the guard' if blue_mode == 'guard' else 'with arms down'} ===")
            for half in ("left", "right"):
                (count, total, groups, poses, dist, inside, scored,
                 my_scored, my_blockers) = scored_counts[half]
                share = 100.0 * count / total
                print(f"  {half:5s} arm, {total} poses: scored on {count} ({share:.2f}%) "
                      f"[torso {groups['torso']}, head {groups['head']}, "
                      f"lower {groups['lower']}]")
                if count:
                    index = int(np.flatnonzero(scored)[0])
                    joints = ", ".join(f"{j} {poses[index, k]:+.2f}"
                                       for k, j in enumerate(scene.ARM_JOINTS))
                    print(f"      a scoring pose: {joints}")
                elif my_scored:
                    best = min(my_scored, key=lambda s: dist[:, s].min())
                    print(f"      closest approach: {slot_str(best)} at "
                          f"{dist[:, best].min():+.3f} m (inside below "
                          f"{slot_margin[best]:+.3f})")
                blockers = [(s, float(inside[:, s].mean())) for s in my_blockers]
                blockers = [b for b in blockers if b[1] > 0.0]
                if blockers:
                    blockers.sort(key=lambda b: -b[1])
                    top = ", ".join(f"{blue_side(s).split('/')[-1]} {100.0 * f:.0f}%"
                                    for s, f in blockers[:4])
                    print(f"      what the swept arm meets instead: {top}")
            print()
            summary[key] = max(scored_counts[half][0] / scored_counts[half][1]
                               for half in ("left", "right"))

    print("=== summary ===")
    for mode in ("guard", "down"):
        row = ", ".join(f"{sep:.2f} m {100.0 * summary[(sep, mode)]:.2f}%"
                        for sep in args.separations)
        label = "the guard" if mode == "guard" else "arms down"
        print(f"  scored poses against {label}: {row}")
    print(f"  the reward pays in_range at {level._target_range:.2f} m and starts "
          f"penalising the clinch inside {level._punch_reach:.2f} m")
    opens = [sep for sep in args.separations
             if summary[(sep, "guard")] > 0 or summary[(sep, "down")] > 0]
    if not opens:
        print("\nverdict: no static pose in the arm's joint box reaches a scoring "
              "surface at any tested separation -- the wall is geometric, and no "
              "policy trained on this scene can be paid for a punch.")
    elif max(opens) < level._target_range:
        guard_best = max(summary[(s, "guard")] for s in opens)
        down_best = max(summary[(s, "down")] for s in opens)
        print(f"\nverdict: static scoring opens only at {max(opens):.2f} m and below "
              f"(guard {100.0 * guard_best:.2f}% of poses, arms down "
              f"{100.0 * down_best:.2f}%) -- arm length, not the guard, is what "
              f"closes the target at the {level._target_range:.2f} m range the "
              f"reward pays to stand at, and the lanes that do exist sit inside "
              f"the clinch band the reward penalises.")
    else:
        print("\nverdict: static poses score at the reward's own fighting range -- "
              "the fifth arm's zero is a search failure, not a wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
