"""Audit of the MJX contact buffer that hit attribution is read from.

Answers three questions with the buffer itself rather than with a guess:

1. what is in it -- how many slots, and what pair of geoms each one holds;
2. does a cross-body fist-on-head slot exist at all;
3. does the slot -> pair mapping change between states.

Run on CPU: CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python tools/slot_audit.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import scene
from boxer.env import BoxingEnv, NUM_AGENTS


def main() -> None:
    level = BoxingEnv()
    model = level._model
    print(f"model: ngeom={model.ngeom} npair={model.npair}")

    state = level.reset(jax.random.PRNGKey(9))
    before = np.asarray(state.pipeline_state.contact.geom)
    print(f"buffer at reset: {before.shape[0]} slots, dtype={before.dtype}")
    print(f"  min geom id={before.min()} max={before.max()} (ngeom={model.ngeom})")

    def name(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<unnamed {gid}>"

    kinds = Counter()
    cross: list[tuple[int, str, str]] = []
    for slot, pair in enumerate(before):
        n1, n2 = name(pair[0]), name(pair[1])
        c1, c2 = n1.split("/")[0] if "/" in n1 else n1, n2.split("/")[0] if "/" in n2 else n2
        if n1 == scene.FLOOR or n2 == scene.FLOOR:
            kinds["floor"] += 1
        elif c1 == c2:
            kinds[f"same-corner ({c1})"] += 1
        else:
            kinds["cross-corner"] += 1
            cross.append((slot, n1, n2))
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")

    strikers = set(scene.HAND_GEOMS + scene.FOREARM_GEOMS)
    bare = lambda n: n.split("/", 1)[1] if "/" in n else n
    fist_to_head = [(s, a, b) for s, a, b in cross
                    if (bare(a) in strikers and bare(b) == "head_collision")
                    or (bare(b) in strikers and bare(a) == "head_collision")]
    fist_to_glove = [(s, a, b) for s, a, b in cross
                     if bare(a) in strikers and bare(b) in strikers
                     and a.split("/")[0] != b.split("/")[0]]
    print(f"  cross-corner fist-on-head slots: {len(fist_to_head)}")
    for slot, a, b in fist_to_head[:6]:
        print(f"    {slot}: {a} <-> {b}")
    print(f"  cross-corner fist-on-fist slots: {len(fist_to_glove)}")
    for slot, a, b in fist_to_glove[:4]:
        print(f"    {slot}: {a} <-> {b}")

    declared = set()
    for i in range(model.npair):
        declared.add(frozenset((
            name(model.pair_geom1[i]), name(model.pair_geom2[i])
        )))
    in_buffer = {frozenset((name(a), name(b))) for a, b in before}
    missing = sorted(p for p in declared - in_buffer)
    print(f"declared pairs: {len(declared)}, distinct pairs in buffer: {len(in_buffer)}")
    print(f"declared but absent from the buffer: {len(missing)}")
    for pair in missing[:10]:
        print(f"    {sorted(pair)}")

    action = jax.random.uniform(
        jax.random.PRNGKey(10), (NUM_AGENTS, level.action_size), minval=-1.0, maxval=1.0
    )
    stepped = jax.jit(level.step)(state, action)
    after = np.asarray(stepped.pipeline_state.contact.geom)
    print(f"buffer after one step: {after.shape[0]} slots, "
          f"identical to reset: {after.shape == before.shape and np.array_equal(after, before)}")
    if after.shape == before.shape and not np.array_equal(after, before):
        differ = int(np.sum(np.any(after != before, axis=1)))
        print(f"  slots whose pair changed: {differ}")

    # And with the corners overlapping, which is where a punch actually happens.
    pipeline = stepped.pipeline_state
    blue_free = level._sides[scene.BLUE]["free_adr"]
    overlapped = level.reset(jax.random.PRNGKey(11)).replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[blue_free].add(-0.50))
    )
    touched = jax.jit(level.step)(overlapped, jp.zeros((NUM_AGENTS, level.action_size)))
    close = np.asarray(touched.pipeline_state.contact.geom)
    print(f"buffer with the corners 0.5 m closer: {close.shape[0]} slots, "
          f"identical to reset: {close.shape == before.shape and np.array_equal(close, before)}")
    dist = np.asarray(touched.pipeline_state.contact.dist)
    active = dist < np.asarray(touched.pipeline_state.contact.includemargin)
    print(f"  slots inside the margin: {int(active.sum())}")
    for slot in np.flatnonzero(active)[:8]:
        print(f"    {slot}: {name(close[slot, 0])} <-> {name(close[slot, 1])} dist={dist[slot]:+.4f}")

    # Do the folded slot tables agree with the names? This is the check the glove
    # test makes, and it is the one that failed while reading the wrong model.
    arm_targets = set(scene.UPPER_TARGET_GEOMS) - {"head_collision", "torso_collision"}
    glove_bad, head_bad, checked = [], [], 0
    for slot, pair in enumerate(before):
        n1, n2 = name(pair[0]), name(pair[1])
        if "/" not in n1 or "/" not in n2 or n1.split("/")[0] == n2.split("/")[0]:
            continue
        (s1, b1), (s2, b2) = n1.split("/", 1), n2.split("/", 1)
        if b1 in strikers:
            striker, victim, side = b1, b2, s1
        elif b2 in strikers:
            striker, victim, side = b2, b1, s2
        else:
            continue
        checked += 1
        weight = float(np.asarray(level._slot[side]["weight"])[slot])
        scoring = float(np.asarray(level._slot[side]["scoring"])[slot])
        head = float(np.asarray(level._slot[side]["head"])[slot])
        if victim in arm_targets and (weight != 0.0 or scoring != 0.0):
            glove_bad.append((slot, side, n1, n2, weight, scoring))
        if victim == "head_collision" and (weight != 1.0 or scoring != 1.0 or head != 1.0):
            head_bad.append((slot, side, n1, n2, weight, scoring, head))
    print(f"cross-corner slots with a striker: {checked}")
    print(f"  glove slots scoring anything: {len(glove_bad)}")
    for row in glove_bad[:6]:
        print(f"    {row}")
    print(f"  head slots not scoring 1.0: {len(head_bad)}")
    for row in head_bad[:6]:
        print(f"    {row}")

    stance_geom = np.asarray(level.pipeline_init(
        level._stance_qpos, jp.zeros(level.sys.nv), None, level._stance_ctrl
    ).contact.geom)
    print(f"the buffer _slot was built from equals the reset buffer: "
          f"{np.array_equal(stance_geom, before)}")

    counts = Counter(frozenset((name(a), name(b))) for a, b in before)
    repeated = {k: v for k, v in counts.items() if v > 1}
    print(f"pairs holding more than one slot: {len(repeated)} "
          f"(extra slots: {sum(repeated.values()) - len(repeated)})")
    for pair, count in list(sorted(repeated.items(), key=lambda kv: -kv[1]))[:5]:
        print(f"    {count}x {sorted(pair)}")

    # A pair in two condim groups gets two slots, so one touch could be summed
    # twice. That is harmless for the hit count, which is an indicator, but not
    # for damage -- so report which repeated pairs carry a weight.
    weighted = []
    for slot, pair in enumerate(before):
        n1, n2 = name(pair[0]), name(pair[1])
        key = frozenset((n1, n2))
        if key not in repeated:
            continue
        total = sum(float(np.asarray(level._slot[side]["weight"])[slot]) for side in ("red", "blue"))
        if total > 0.0:
            weighted.append((slot, sorted(key), total))
    print(f"repeated-pair slots carrying a weight: {len(weighted)}")
    for row in weighted[:10]:
        print(f"    {row}")


if __name__ == "__main__":
    main()
