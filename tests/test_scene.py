"""Checks that the composed scene really holds two independent boxers."""

import mujoco
import numpy as np
import pytest

from boxer import scene


@pytest.fixture(scope="session")
def model():
    return scene.build_model()


def test_two_robots_are_attached(model):
    for side in scene.SIDES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "torso_link") >= 0
    assert model.nu == 58
    assert model.nq == 72


def test_boxers_face_each_other(model):
    data = mujoco.MjData(model)
    qpos, ctrl = scene.stance(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    red = data.site(scene.RED + "torso").xpos
    blue = data.site(scene.BLUE + "torso").xpos
    assert red[0] < 0 < blue[0]
    assert np.linalg.norm(red - blue) == pytest.approx(1.3, abs=0.15)
    # Facing is the +x axis of the torso frame; the two must point at each other.
    red_forward = data.site(scene.RED + "torso").xmat.reshape(3, 3)[:, 0]
    blue_forward = data.site(scene.BLUE + "torso").xmat.reshape(3, 3)[:, 0]
    assert np.dot(red_forward, blue - red) > 0
    assert np.dot(blue_forward, red - blue) > 0


def test_stance_holds_the_guard(model):
    qpos, ctrl = scene.stance(model)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    for side in scene.SIDES:
        torso = data.site(side + "torso")
        to_local = torso.xmat.reshape(3, 3).T
        for hand in ("left", "right"):
            palm = to_local @ (data.site(f"{side}{hand}_palm").xpos - torso.xpos)
            # Fists in front of the chest, held toward the centre line: a guard, not
            # a relaxed hang and not a reach. The upstream home keyframe fails the
            # first of these. The height threshold used to be > 0.10, which this
            # robot only satisfies by breaking its wrist or straightening its elbow,
            # so it was testing the fit's fantasy rather than the pose's shape --
            # +0.05 m above torso centre is the measured straight-wrist ceiling.
            assert palm[0] > 0.15, palm
            assert palm[2] > -0.02, palm
            assert abs(palm[1]) < 0.20, palm

        # The two shapes the position-only fit got wrong: a guard bends the elbow and
        # keeps the wrist nearly straight.
        names = ("left_elbow_joint", "right_elbow_joint", "left_wrist_pitch_joint",
                 "right_wrist_pitch_joint", "left_wrist_yaw_joint",
                 "right_wrist_yaw_joint")
        joint = {}
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}{name}")
            joint[name] = qpos[model.jnt_qposadr[jid]]
        assert joint["left_elbow_joint"] > 0.4 and joint["right_elbow_joint"] > 0.4, joint
        for wrist in names[2:]:
            assert abs(joint[wrist]) < 0.25, f"{wrist} at {joint[wrist]:+.2f} rad"

    for _ in range(250):
        mujoco.mj_step(model, data)
    for side in scene.SIDES:
        # Still standing after a second of holding the pose, guard still up.
        assert data.site(side + "torso").xpos[2] > 0.5
        torso = data.site(side + "torso")
        to_local = torso.xmat.reshape(3, 3).T
        palm = to_local @ (data.site(side + "left_palm").xpos - torso.xpos)
        assert palm[0] > 0.10 and palm[2] > -0.05


def test_the_legacy_pose_stays_reachable_by_name(model):
    """Checkpoints trained before the stance was recorded must still be playable.

    The pose is part of the observation, so a checkpoint is only the same agent in
    the pose it learned. That makes the broken-wrist guard a thing evaluation has to
    be able to ask for, not just a mistake to delete.
    """
    natural = scene.stance(model)
    legacy = scene.stance(model, "broken-wrist")
    assert not np.array_equal(natural[0], legacy[0])

    adr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, scene.RED + "left_wrist_yaw_joint")]
    assert legacy[0][adr] == pytest.approx(-1.610, abs=1e-3), "at its joint limit"
    assert natural[0][adr] == pytest.approx(-0.043, abs=1e-3)

    with pytest.raises(SystemExit):
        scene.stance(model, "chin-height")


def test_no_duplicate_contact_pairs(model):
    pairs = set()
    for i in range(model.npair):
        key = frozenset((model.pair_geom1[i], model.pair_geom2[i]))
        assert key not in pairs, "the same unordered geom pair is declared twice"
        assert len(key) == 2, "a pair must reference two distinct geoms"
        pairs.add(key)


def test_striking_pairs_exist_both_ways(model):
    """A fist must be able to reach the opponent's head, in both directions."""
    geom = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    pairs = {
        frozenset((model.pair_geom1[i], model.pair_geom2[i])) for i in range(model.npair)
    }
    for striker, target in ((scene.RED, scene.BLUE), (scene.BLUE, scene.RED)):
        key = frozenset((geom(striker + "right_hand_collision"), geom(target + "head_collision")))
        assert key in pairs


def test_sensors_are_prefixed_per_boxer(model):
    for side in scene.SIDES:
        for name in ("torso_pos", "head_pos", "left_palm_linvel", "torso_quat"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, side + name) >= 0


def test_mjx_solver_settings_survive_the_attach(model):
    """The root spec wins option conflicts, so these must be restated."""
    assert model.opt.timestep == pytest.approx(0.004)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.iterations == 5


def test_the_closed_fist_is_a_render_and_not_a_change_of_robot(model):
    """The grafted hand must reach the camera and nothing else.

    Watched back, the third arm had open hands: the robot that trains carries one
    rigid rubber-hand mesh. The articulated-hand variant of the same G1 would fix
    that at the source and cost a rewrite of every contact pair plus a solver that
    ships at 2 ms and 100 iterations, so the fist is grafted instead -- the hand's
    own meshes, welded shut, as visual geoms on the wrist.

    What must not happen is the physics moving with it. The trainer compiles the
    scene without visuals, and a graft that survived that build would change the
    mass at the end of both arms and the contact list, which is to say the meaning
    of every checkpoint already trained. So the two builds are held side by side:
    identical in the degrees of freedom, the pairs and the mass, and the fingers
    present only in the one that renders.
    """
    plain = scene.build_model(visual=False)
    assert (plain.nq, plain.nv, plain.nu, plain.npair) == \
           (model.nq, model.nv, model.nu, model.npair)
    assert float(plain.body_mass.sum()) == pytest.approx(float(model.body_mass.sum())), \
        "a visual graft must not add mass to the end of the arm"

    def meshes(built):
        return [
            mujoco.mj_id2name(built, mujoco.mjtObj.mjOBJ_MESH, built.geom_dataid[g]) or ""
            for g in range(built.ngeom)
            if built.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ]

    hands = lambda built: {m for m in meshes(built) if "_hand_" in m}
    assert hands(plain) == set(), "the trainer's model must not carry the graft"
    assert len(hands(model)) == 2 * len(scene.SIDES) * scene.FIST_MESHES_PER_HAND
    assert not any("rubber_hand" in m for m in meshes(model)), \
        "the open hand has to go, or the fist grows out of it"

    # On the wrist, and inert: a geom has no degrees of freedom, which is what
    # welds the fingers, and contype 0 with no pair is what makes it visual.
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              scene.RED + "left_wrist_yaw_link")

    def mesh_of(g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) or ""

    fingers = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == wrist
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and "_hand_" in mesh_of(g)          # the wrist link's own mesh is not a finger
    ]
    assert len(fingers) == scene.FIST_MESHES_PER_HAND
    for g in fingers:
        assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        assert model.geom_group[g] == 2, "group 2 is what the visual build renders"


def test_the_grafted_fingers_are_curled_and_not_splayed(model):
    """A fist, checked geometrically, because a render can also be argued with.

    The pose is written as joint angles of the articulated model and then welded,
    so nothing else verifies that the result is closed -- and the two hands take
    opposite signs, because the right hand's joint ranges are the negation of the
    left's. If a sign is wrong the fingers stay straight and the boxer renders
    karate-chopping, which is exactly the mistake the mirrored ranges invite.
    """
    data = mujoco.MjData(model)
    qpos, ctrl = scene.stance(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    distal = ("index_1_link", "middle_1_link", "thumb_2_link")
    for side in scene.SIDES:
        for hand in ("left", "right"):
            wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      f"{side}{hand}_wrist_yaw_link")
            palm, tips = None, []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != wrist:
                    continue
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                         model.geom_dataid[g]) or ""
                if name.endswith("palm_link"):
                    palm = data.geom_xpos[g].copy()
                elif name.endswith(distal):
                    tips.append(data.geom_xpos[g].copy())
            assert palm is not None and len(tips) == 3, f"{side}{hand} graft incomplete"
            reach = [float(np.linalg.norm(tip - palm)) for tip in tips]
            assert max(reach) < 0.08, (
                f"{side}{hand}: the distal links sit {max(reach):.3f} m from the palm "
                "mesh. Measured on the articulated model: 0.056-0.061 m at these joint "
                "angles, 0.094-0.111 m with the fingers straight, so 0.08 is the line "
                "between a fist and a karate chop -- and a wrong sign on one hand "
                "leaves it straight."
            )

