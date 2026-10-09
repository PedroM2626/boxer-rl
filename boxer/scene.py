"""Two-boxer MuJoCo scene composed from the Menagerie Unitree G1 model.

The scene is built with ``mujoco.MjSpec`` so that the robot description is never
hand-written: two copies of the upstream ``g1_mjx.xml`` are attached to one root
spec under the ``red/`` and ``blue/`` name prefixes, facing each other.

Contacts are declared explicitly (``<pair>``) rather than left to the automatic
contype/conaffinity filter, which is how the upstream MJX scene is set up. That
keeps the candidate pair count -- and therefore the MJX contact buffer -- small,
and it makes every striking contact addressable by geom id.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_menagerie as menagerie
import numpy as np

RED = "red/"
BLUE = "blue/"
SIDES = (RED, BLUE)
OPPONENT = {RED: BLUE, BLUE: RED}

FLOOR = "floor"
RING_RADIUS = 2.0

# Geoms that score when they land on the opponent. The hand capsule spans the
# fist; the wrist capsule stands in for the forearm, which is what a boxer
# actually blocks with.
HAND_GEOMS = ("left_hand_collision", "right_hand_collision")
FOREARM_GEOMS = ("left_wrist_collision", "right_wrist_collision")
# Geoms that can be punched. Head and torso are the scoring targets; the arm
# geoms are there so that blocking is a real physical event, not a pass-through.
UPPER_TARGET_GEOMS = (
    "head_collision",
    "torso_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
    "left_wrist_collision",
    "right_wrist_collision",
    "left_hand_collision",
    "right_hand_collision",
)
BODY_TARGET_GEOMS = (
    "pelvis_collision",
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
)
STRIKE_TARGETS = (
    (HAND_GEOMS, UPPER_TARGET_GEOMS + BODY_TARGET_GEOMS),
    (FOREARM_GEOMS, UPPER_TARGET_GEOMS),
)

# Leg contacts between the two robots: footwork, trips and the clinch. Only
# like-for-like groups plus foot-against-shin are worth the pair budget.
CLINCH_GROUPS = (
    ("left_thigh_collision", "right_thigh_collision"),
    ("left_shin_collision", "right_shin_collision"),
    ("left_foot_box_collision", "right_foot_box_collision"),
    ("left_hip_collision", "right_hip_collision"),
)
CLINCH_CROSS_GROUPS = ((CLINCH_GROUPS[2], CLINCH_GROUPS[1]),)
CLINCH_TRUNK_PAIRS = (("torso_collision", "torso_collision"), ("pelvis_collision", "pelvis_collision"))

# Sites added on top of the upstream model, per robot.
EXTRA_SITES = {
    "torso_link": (("torso", (0.0, 0.0, 0.0)), ("head", (0.0, 0.0, 0.43))),
}
POS_SITES = ("torso", "head", "left_palm", "right_palm")
VEL_SITES = ("torso", "left_palm", "right_palm")

# Upstream pairs reused as parameter templates for the cross-robot contacts.
SLIP_TEMPLATE = "left_hand_right_hand"
GRIP_TEMPLATE = "left_foot1_floor"

ARM_JOINTS = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)
# Arm targets for the guard, fitted by random search against MuJoCo forward
# kinematics (tools/fit_guard.py). The first version of this fit minimised fist
# *position* alone and landed on elbows locked straight with both wrists cranked
# ~90 degrees, because the arm is redundant and the palm site sits 8 cm down the
# forearm from the wrist yaw joint -- breaking the wrist moves the fist without
# moving the arm, so it was the cheapest solution to a target the robot cannot
# reach naturally.
#
# The replacement was fitted with the wrists inside 0.15 rad and a floor on elbow
# flexion, and the interesting result is what it cannot do: at elbow floors of 0.6,
# 0.9 and 1.2 rad the fist tops out at +0.05 m above torso centre every time, and
# asking for chin height (+0.20 m) only makes the search pay for it by straightening
# the arm or pinning the wrist at its cap. A high elbow flexion buys reach forward,
# not height. So this is the measured envelope of a straight-wristed G1 guard: hands
# (0.36, +-0.12, +0.05) m in the torso frame, elbows flexed 34 degrees, 24 cm apart.
# It reads as hands at the sternum rather than at the chin, and that is the robot,
# not the fit.
GUARD_ARMS = {
    "left": (-0.446, 0.099, -0.068, 0.600, -0.008, -0.150, -0.043),
    "right": (-0.446, -0.100, 0.068, 0.600, 0.009, -0.150, 0.043),
}
# The pose every checkpoint in runs/ before 2026-10-08 was trained in: fists at
# chin height, elbows locked, wrists cranked to their stops. It is kept only so
# those checkpoints can still be played in the environment they learned. The
# stance enters the observation -- 6 guard-error dims and 29 `qpos - stance`
# dims -- so reloading one of them into the fitted pose above measures a shifted
# input distribution, not the same agent. Measured: the 204M arm went from 0.938
# wins against a statue to 0.139, and from 0.754 against the 12.45M arm to 0.051.
GUARD_ARMS_BROKEN_WRIST = {
    "left": (-0.676, 0.679, 0.236, -0.007, -0.573, 1.491, -1.610),
    "right": (-0.781, -0.930, 0.076, -0.310, 0.430, 1.614, 1.419),
}

#: Names a run can record so an evaluation rebuilds the pose it trained with.
GUARD_POSES = {"natural": GUARD_ARMS, "broken-wrist": GUARD_ARMS_BROKEN_WRIST}


def guard_arms(name: str) -> dict:
    try:
        return GUARD_POSES[name]
    except KeyError:
        raise SystemExit(f"unknown guard pose {name!r}; choose from {sorted(GUARD_POSES)}")


def robot_xml_path() -> Path:
    """Path to the upstream MJX-ready Unitree G1 description."""
    return menagerie.get("unitree_g1").xml("g1_mjx")


def scene_xml_path() -> Path:
    """Path to the upstream single-robot MJX scene (floor + contact pairs)."""
    return menagerie.get("unitree_g1").xml("scene_mjx")


def upstream_scene() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(scene_xml_path()))


def home_pose() -> tuple[np.ndarray, np.ndarray]:
    """The upstream ``home`` keyframe: standing, elbows bent into a guard."""
    spec = upstream_scene()
    key = next(k for k in spec.keys if k.name == "home")
    return np.asarray(key.qpos, dtype=np.float64), np.asarray(key.ctrl, dtype=np.float64)


def _copy_pair(spec: mujoco.MjSpec, geom1: str, geom2: str, template) -> None:
    """Adds a pair whose solver parameters mirror an upstream pair."""
    pair = spec.add_pair()
    pair.geomname1 = geom1
    pair.geomname2 = geom2
    pair.condim = template.condim
    pair.friction = np.asarray(template.friction).copy()
    pair.solref = np.asarray(template.solref).copy()
    pair.solimp = np.asarray(template.solimp).copy()
    pair.solreffriction = np.asarray(template.solreffriction).copy()
    pair.margin = template.margin
    pair.gap = template.gap


def _add_arena(spec: mujoco.MjSpec, visual: bool) -> None:
    checker = spec.add_texture()
    checker.name = "canvas"
    checker.type = mujoco.mjtTexture.mjTEXTURE_2D
    checker.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    checker.rgb1 = [0.22, 0.26, 0.32]
    checker.rgb2 = [0.14, 0.17, 0.21]
    checker.width, checker.height = 300, 300

    canvas = spec.add_material()
    canvas.name = "canvas"
    canvas.textures = ["canvas"] + [""] * 9
    canvas.texrepeat = [8, 8]
    canvas.reflectance = 0.15

    if visual:
        sky = spec.add_texture()
        sky.name = "skybox"
        sky.type = mujoco.mjtTexture.mjTEXTURE_CUBE
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1 = [0.3, 0.5, 0.7]
        sky.rgb2 = [0.05, 0.05, 0.1]
        sky.width, sky.height = 512, 3072

    # contype/conaffinity stay 0: every contact in this scene is an explicit pair.
    floor = spec.worldbody.add_geom()
    floor.name = FLOOR
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "canvas"
    floor.contype = 0
    floor.conaffinity = 0

    for pos, direction, diffuse in (
        ([0.0, 0.0, 4.0], [0.0, 0.0, -1.0], [0.75, 0.75, 0.75]),
        ([-3.0, -3.0, 3.0], [0.4, 0.4, -0.8], [0.35, 0.35, 0.35]),
    ):
        light = spec.worldbody.add_light(pos=pos, dir=direction, diffuse=diffuse)
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL


def _add_robot(spec: mujoco.MjSpec, prefix: str, x: float, face_away: bool) -> None:
    """Attaches one G1 at ``x`` on the ring's centre line.

    The upstream model faces +x, so the boxer on the positive side is turned
    180 degrees about z to look at its opponent.
    """
    robot = mujoco.MjSpec.from_file(str(robot_xml_path()))

    for sensor in list(robot.sensors):
        robot.delete(sensor)
    for body_name, sites in EXTRA_SITES.items():
        body = robot.body(body_name)
        for site_name, pos in sites:
            site = body.add_site()
            site.name = site_name
            site.pos = list(pos)
            site.size = np.array([[0.01], [0.0], [0.0]])
            site.group = 5

    quat = [0.0, 0.0, 0.0, 1.0] if face_away else [1.0, 0.0, 0.0, 0.0]
    frame = spec.worldbody.add_frame(pos=[x, 0.0, 0.0], quat=quat)
    spec.attach(robot, prefix=prefix, frame=frame)


def _clinch_pairs() -> list[tuple[str, str]]:
    pairs = list(CLINCH_TRUNK_PAIRS)
    for group in CLINCH_GROUPS:
        pairs += [(a, b) for a in group for b in group]
    for group1, group2 in CLINCH_CROSS_GROUPS:
        pairs += [(a, b) for a in group1 for b in group2]
    return pairs


def _add_contacts(spec: mujoco.MjSpec) -> None:
    upstream = upstream_scene()
    templates = {p.name: p for p in upstream.pairs}
    slip, grip = templates[SLIP_TEMPLATE], templates[GRIP_TEMPLATE]

    wanted: list[tuple[str, str, object]] = []
    for side in SIDES:
        for source in upstream.pairs:
            wanted.append((
                FLOOR if source.geomname1 == FLOOR else side + source.geomname1,
                FLOOR if source.geomname2 == FLOOR else side + source.geomname2,
                source,
            ))

    for side in SIDES:
        other = OPPONENT[side]
        for strikers, targets in STRIKE_TARGETS:
            for striker in strikers:
                for target in targets:
                    wanted.append((side + striker, other + target, slip))
        for geom1, geom2 in _clinch_pairs():
            wanted.append((side + geom1, other + geom2, grip))

    # A pair is unordered in MuJoCo: declaring both "red fist on blue head" and
    # "blue fist on red head" would allocate two contact slots for one collision
    # and score the same punch twice. The geom role mask in the env already tells
    # the two directions apart.
    seen: set[frozenset[str]] = set()
    for geom1, geom2, template in wanted:
        key = frozenset((geom1, geom2))
        if len(key) == 1 or key in seen:
            continue
        seen.add(key)
        _copy_pair(spec, geom1, geom2, template)


def _add_sensors(spec: mujoco.MjSpec) -> None:
    readings = [(site, "_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS) for site in POS_SITES]
    readings += [(site, "_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL) for site in VEL_SITES]
    readings += [("torso", "_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT)]
    readings += [("imu_in_torso", "_gyro", mujoco.mjtSensor.mjSENS_GYRO)]

    for side in SIDES:
        for site, suffix, sensor_type in readings:
            sensor = spec.add_sensor()
            sensor.name = f"{side}{site}{suffix}"
            sensor.type = sensor_type
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = side + site


def build_spec(spawn_distance: float = 1.3, visual: bool = True) -> mujoco.MjSpec:
    """Composes the full two-boxer scene spec."""
    spec = mujoco.MjSpec()
    # The root spec wins option conflicts on attach, so the upstream G1 solver
    # settings have to be restated here or the scene silently reverts to defaults.
    spec.option.timestep = 0.004
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 5
    spec.option.ls_iterations = 8
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    spec.compiler.degree = False
    spec.compiler.discardvisual = not visual
    # mujoco.Renderer refuses to draw larger than the model's offscreen
    # framebuffer, whose default is 640x480, so size it for rollout renders.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    _add_arena(spec, visual)
    half = spawn_distance / 2.0
    _add_robot(spec, RED, -half, face_away=False)
    _add_robot(spec, BLUE, +half, face_away=True)
    _add_contacts(spec)
    _add_sensors(spec)
    return spec


def build_model(spawn_distance: float = 1.3, visual: bool = True) -> mujoco.MjModel:
    return build_spec(spawn_distance, visual=visual).compile()


def stance(model: mujoco.MjModel, guard_pose: str = "natural") -> tuple[np.ndarray, np.ndarray]:
    """Guard pose for both boxers, as ``(qpos, ctrl)`` for ``model``.

    Free-joint values come from the model defaults, which the frame attachment
    already placed on the ring centre line facing the opponent; the legs and
    waist take the upstream home keyframe and the arms take the named pose in
    :data:`GUARD_POSES`.
    """
    arms = guard_arms(guard_pose)
    home_qpos, home_ctrl = home_pose()
    robot = mujoco.MjModel.from_xml_path(str(robot_xml_path()))

    qpos = model.qpos0.copy()
    ctrl = np.zeros(model.nu)
    for side in SIDES:
        for j in range(robot.njnt):
            if robot.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_JOINT, j)
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, side + name)
            qpos[model.jnt_qposadr[target]] = home_qpos[robot.jnt_qposadr[j]]
        for a in range(robot.nu):
            name = mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, side + name)] = home_ctrl[a]

        for half, values in arms.items():
            for joint, value in zip(ARM_JOINTS, values):
                name = f"{side}{half}_{joint}_joint"
                qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
                ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = value
    return qpos, ctrl
