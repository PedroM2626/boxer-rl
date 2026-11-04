"""Native MuJoCo transfer harness for two unassisted foundation actors.

The learned feet, dynamics and 103-input interface come from the foundation.
Opponent collision geometry is additional physics, never a stabilizing force.
This harness is deliberately separate from legacy assisted boxing controllers.
"""
from __future__ import annotations

import mujoco
import numpy as np

from boxer import scene
from boxer.impact_scorecard import ImpactScorecard


STRIKE_SENSOR_TARGETS = ('head_collision', 'torso_collision')
# Roped ring, scaled to the 1.32 m G1 (0.73 x a human ring): 4 m between the
# ropes, three ropes at 0.35 / 0.60 / 0.85 m (human: 0.46 / 0.81 / 1.17 m).
RING_HALF = 2.0
ROPE_HEIGHTS = (.35, .60, .85)
ROPE_RADIUS = .02
# Ropes give like stiff springs (50 ms time constant) rather than walls.
ROPE_SOLREF = (.05, 1.)
# Robot geoms that collide with the ropes.
ROPE_CONTACTS = (scene.HAND_GEOMS + scene.FOREARM_GEOMS + scene.UPPER_TARGET_GEOMS
                 + scene.BODY_TARGET_GEOMS + ('left_shin', 'right_shin', 'left_foot', 'right_foot'))


def add_ring(spec, robot_geoms):
    """Four corner posts and three ropes per side, colliding with each robot."""
    h = RING_HALF + ROPE_RADIUS
    corners = [(h, h), (-h, h), (-h, -h), (h, -h)]
    ropes = []
    for k, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
        for level, z in enumerate(ROPE_HEIGHTS):
            rope = spec.worldbody.add_geom(
                name=f'rope_{k}_{level}', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[a[0], a[1], z, b[0], b[1], z], size=[ROPE_RADIUS, 0, 0],
                contype=0, conaffinity=0)
            rope.rgba = [.85, .85, .85, 1]
            ropes.append(rope.name)
    for k, (x, y) in enumerate(corners):
        post = spec.worldbody.add_geom(
            name=f'post_{k}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x * 1.02, y * 1.02, .5], size=[.04, .5, 0], contype=0, conaffinity=0)
        post.rgba = [.55, .55, .55, 1]
    for rope in ropes:
        for side in scene.SIDES:
            for name in ROPE_CONTACTS:
                if name in robot_geoms:
                    pair = spec.add_pair(geomname1=rope, geomname2=side + name, condim=3,
                                         friction=[.3, .3, .005, .0001, .0001])
                    pair.solref = list(ROPE_SOLREF)


def strike_sensor_name(side, hand, target):
    """Contact sensor: ``side``'s ``hand`` fist touching the opponent's ``target``."""
    return f'{side}{hand}_on_{target}'


def build_pair(foundation, separation=2.0, *, visual_fists=False, diagnostic_excluded_pairs=(),
               strike_sensors=False, ring=False):
    """Both robots on a shared floor; ``ring`` adds the roped ring (``add_ring``)."""
    if not np.isfinite(separation) or separation < 1.8:
        raise ValueError("transfer spawns must be separated by at least 1.8 m")
    source = foundation.mj_model
    spec = mujoco.MjSpec()
    for name in ("timestep", "integrator", "iterations", "ls_iterations", "cone",
                 "disableflags", "solver", "impratio", "gravity"):
        setattr(spec.option, name, getattr(source.opt, name))
    spec.compiler.degree = False
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[0, 0, .01], contype=1, conaffinity=0)
    floor.rgba = [.2, .25, .3, 1]
    spec.worldbody.add_light(pos=[0, 0, 4])
    # Read measured target geometry from the existing scene; retain the learned
    # robot rather than replacing its foot shapes or joint dynamics with it.
    targets = scene.build_model(separation, visual=False)
    for index, side in enumerate(scene.SIDES):
        robot = mujoco.MjSpec.from_file(foundation.xml_path, assets=foundation._model_assets)
        if visual_fists:
            # Reuse the original closed-hand visual graft only in replay models.
            # The live learner keeps its exact compiled model and geom ordering.
            scene._graft_fists(robot, scene.FIST_COLORS[side])
        for sensor in list(robot.sensors):
            if sensor.type == mujoco.mjtSensor.mjSENS_CONTACT:
                robot.delete(sensor)
        floor_pairs = []
        for pair in list(robot.pairs):
            if "floor" in (pair.geomname1, pair.geomname2):
                floor_pairs.append((pair.geomname1, pair.geomname2, pair.condim,
                                    np.array(pair.friction), np.array(pair.solref),
                                    np.array(pair.solimp)))
                robot.delete(pair)
        robot.delete(robot.geom("floor"))
        for key in list(robot.keys):
            robot.delete(key)
        existing = {g.name for g in robot.geoms}
        # Head, torso and blocking surfaces are absent from feet-only contacts.
        # Copy massless collision surfaces only: body inertials remain explicit.
        for name in set(scene.UPPER_TARGET_GEOMS + scene.BODY_TARGET_GEOMS):
            if name in existing:
                continue
            reference = targets.geom(scene.RED + name)
            body_name = targets.body(reference.bodyid[0]).name[len(scene.RED):]
            geom = robot.body(body_name).add_geom(name=name, type=reference.type[0],
                size=reference.size.copy(), pos=reference.pos.copy(), quat=reference.quat.copy(),
                contype=0, conaffinity=0, mass=0, group=3)
            geom.rgba = [.7, .2, .2, .25] if index == 0 else [.2, .3, .8, .25]
        frame = spec.worldbody.add_frame(pos=[(-.5 if index == 0 else .5) * separation, 0, 0],
                                         quat=[1, 0, 0, 0] if index == 0 else [0, 0, 0, 1])
        robot_geoms = {g.name for g in robot.geoms}
        spec.attach(robot, prefix=side, frame=frame)
        for first, second, dim, friction, solref, solimp in floor_pairs:
            pair = spec.add_pair(geomname1="floor" if first == "floor" else side + first,
                                 geomname2="floor" if second == "floor" else side + second)
            pair.condim, pair.friction, pair.solref, pair.solimp = dim, friction, solref, solimp
    wanted = set()
    for side in scene.SIDES:
        other = scene.OPPONENT[side]
        for strikers, surfaces in scene.STRIKE_TARGETS:
            for fist in strikers:
                for target in surfaces:
                    wanted.add(tuple(sorted((side + fist, other + target))))
        for limb in ("left_foot", "right_foot", "left_shin", "right_shin",
                     "left_thigh", "right_thigh", "torso_collision", "pelvis_collision"):
            for target in ("left_foot", "right_foot", "left_shin", "right_shin",
                           "left_thigh", "right_thigh", "torso_collision", "pelvis_collision"):
                wanted.add(tuple(sorted((side + limb, other + target))))
    excluded={tuple(sorted(pair)) for pair in diagnostic_excluded_pairs}
    if not excluded <= wanted:
        raise ValueError('diagnostic exclusion must name existing cross-robot pairs')
    for first, second in sorted(wanted-excluded):
        spec.add_pair(geomname1=first, geomname2=second, condim=3, friction=[.6, .6, .005, .0001, .0001])
    if ring:
        add_ring(spec, robot_geoms)
    if strike_sensors:
        # Read-only "found" flags for batched training: fist-on-target and the
        # chest-to-chest clinch. They add no force and no contact pairs.
        def found(name, first, second):
            sensor = spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_CONTACT)
            sensor.objtype = sensor.reftype = mujoco.mjtObj.mjOBJ_GEOM
            sensor.objname, sensor.refname = first, second
            sensor.intprm[0] = 1      # data: found
            sensor.intprm[1] = 1      # reduce: mindist
            sensor.intprm[2] = 1      # num
        for side in scene.SIDES:
            other = scene.OPPONENT[side]
            for hand in scene.HAND_GEOMS:
                for target in STRIKE_SENSOR_TARGETS:
                    found(strike_sensor_name(side, hand, target), side + hand, other + target)
        found('clinch', scene.RED + 'torso_collision', scene.BLUE + 'torso_collision')
    model = spec.compile()
    if model.neq or np.any(model.body_gravcomp):
        raise ValueError("paired transfer forbids constraints and gravity compensation")
    # Contract checks fail before any action if composition changes learned DOFs.
    for side in scene.SIDES:
        ids = [model.actuator(side + source.actuator(i).name).id for i in range(source.nu)]
        np.testing.assert_allclose(model.actuator_gainprm[ids], source.actuator_gainprm)
        np.testing.assert_allclose(model.actuator_biasprm[ids], source.actuator_biasprm)
    return model


class PairedFoundation:
    """Both policies act on shared physics; the first fall ends the round."""

    dt = .02

    def __init__(self, foundation, separation=2.0, *, diagnostic_excluded_pairs=(), model=None,
                 ring=False):
        """``model``: an already built ``build_pair`` model (e.g. loaded in a worker)."""
        self.foundation = foundation
        self.model = model if model is not None else build_pair(
            foundation, separation, diagnostic_excluded_pairs=diagnostic_excluded_pairs, ring=ring)
        # A loaded model (native fine-tuning workers) carries its own ring.
        self.ring = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'rope_0_0') >= 0
        self.data = mujoco.MjData(self.model)
        self.pose = np.asarray(foundation._default_pose)
        self.init_height32 = np.float32(foundation._init_q[2])
        self.action_scale = float(foundation._config.action_scale)
        self.substeps = round(self.dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.dt):
            raise ValueError("physics/control interval mismatch")
        self.maps = []
        for side in scene.SIDES:
            actuators = np.array([self.model.actuator(side + foundation.mj_model.actuator(i).name).id
                                  for i in range(29)])
            joints = self.model.actuator_trnid[actuators, 0]
            root = next(j for j in range(self.model.njnt)
                        if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        and self.model.joint(j).name.startswith(side))
            self.maps.append(dict(act=actuators, q=self.model.jnt_qposadr[joints],
                                  v=self.model.jnt_dofadr[joints], root=int(self.model.jnt_qposadr[root]),
                                  rootv=int(self.model.jnt_dofadr[root]), side=side))
        self.fists = np.array([[self.model.geom(side + hand).id for hand in scene.HAND_GEOMS]
                               for side in scene.SIDES])
        self.targets = [{self.model.geom(side + target).id
                         for target in ('head_collision', 'torso_collision')}
                        for side in scene.SIDES]
        self.scorecard = ImpactScorecard()
        # The foundation also ends a trial when a robot's foot touches its own
        # other foot or shin. Knockdown rules may count only real falls.
        self.self_trip_falls = True
        self._index_geoms()
        self.reset()

    def _index_geoms(self):
        """Integer lookups that replace per-substep name and set work in ``step``."""
        model = self.model
        self._tracked = sorted(set(self.fists.ravel()) | self.targets[0] | self.targets[1])
        self._torso_sites = [model.site(m['side'] + 'imu_in_torso').id for m in self.maps]
        self._aim_geoms = [(model.geom(side + 'head_collision').id, model.geom(side + 'torso_collision').id)
                           for side in scene.SIDES]
        self._warnings = [w for w in (getattr(mujoco.mjtWarning, name, None) for name in (
            'mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC',
            'mjWARN_CNSTRFULL', 'mjWARN_CONTACTFULL')) if w is not None]
        # owner[g] = (corner, hand) for fists; target_of[g] = corner whose target g is.
        self._fist_owner = {int(g): (i, hand) for i in range(2) for hand, g in enumerate(self.fists[i])}
        self._target_of = {int(g): i for i in range(2) for g in self.targets[i]}
        # Self-contacts that count as a fall (foot on foot or foot on the other shin).
        trips = ({'left_foot', 'right_foot'}, {'left_foot', 'right_shin'}, {'right_foot', 'left_shin'})
        self._trip = {}
        names = [model.geom(g).name for g in range(model.ngeom)]
        self._ropes = {g for g, n in enumerate(names) if n.startswith('rope_')}
        self._corner_of = {g: i for i, side in enumerate(scene.SIDES)
                           for g, n in enumerate(names) if n.startswith(side)}
        legs = ('foot', 'shin', 'thigh', 'thigh_collision', 'hip_collision')
        self._legs = {g for g in self._corner_of if names[g].endswith(legs)}
        for i, side in enumerate(scene.SIDES):
            own = {n[len(side):]: g for g, n in enumerate(names) if n.startswith(side)}
            for pair in trips:
                a, b = sorted(pair)
                if a in own and b in own:
                    self._trip[frozenset((own[a], own[b]))] = i

    def reset(self, seed=0, swap=False, randomized=True):
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.phase = np.tile([0., np.pi], (2, 1))
        self.frequency = rng.uniform(1.25, 1.5, (2, 1))
        self.last_action = np.zeros((2, 29))
        self.done = False
        self.scorecard.reset()
        self.elapsed = 0.
        for index, mapping in enumerate(self.maps):
            q, rv, side = mapping['root'], mapping['rootv'], mapping['side']
            self.data.qpos[mapping['q']] = self.pose * (rng.uniform(.5, 1.5, 29) if randomized else 1.)
            self.data.qpos[q + 2] = float(self.foundation._init_q[2])
            if swap:
                self.data.qpos[q:q+2] *= -1
                self.data.qpos[q+3:q+7] = [0, 0, 0, 1] if index == 0 else [1, 0, 0, 0]
            if randomized:
                self.data.qpos[q:q+2] += rng.uniform(-.05, .05, 2)
                self.data.qvel[rv:rv+6] = rng.uniform(-.5, .5, 6)
            self.data.ctrl[mapping['act']] = self.data.qpos[mapping['q']]
        mujoco.mj_forward(self.model, self.data)
        return self.observation(np.zeros((2, 3)))

    def observation(self, commands):
        commands = np.asarray(commands)
        if commands.shape != (2, 3) or not np.isfinite(commands).all():
            raise ValueError("commands must be finite with shape (2, 3)")
        rows = []
        for i, mapping in enumerate(self.maps):
            side = mapping['side']
            gravity = self.data.site(side + 'imu_in_pelvis').xmat.reshape(3, 3).T @ [0, 0, -1]
            rows.append(np.concatenate([
                self.data.sensor(side + 'local_linvel_pelvis').data,
                self.data.sensor(side + 'gyro_pelvis').data, gravity, commands[i],
                self.data.qpos[mapping['q']] - self.pose, self.data.qvel[mapping['v']],
                self.last_action[i], np.cos(self.phase[i]), np.sin(self.phase[i])]))
        return np.asarray(rows, dtype=np.float32)

    def approach_commands(self, stop_distance=.75):
        roots = [self.data.qpos[m['root']:m['root']+2] for m in self.maps]
        delta = roots[1] - roots[0]
        distance = np.linalg.norm(delta)
        speed = np.clip((distance - stop_distance) * .8, 0., .45)
        commands = []
        for i, mapping in enumerate(self.maps):
            rotation = self.data.site(mapping['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)
            local = rotation[:2, :2].T @ (delta if i == 0 else -delta)
            commands.append([*(speed * local / max(distance, 1e-6)),
                             np.clip(np.arctan2(local[1], local[0]), -.5, .5)])
        return np.asarray(commands)

    def on_ropes(self):
        """Per corner: is any part of the robot touching a rope now?"""
        touching = np.zeros(2, bool)
        if self._ropes:
            for g1, g2 in self.data.contact.geom:
                g1, g2 = int(g1), int(g2)
                if g1 in self._ropes or g2 in self._ropes:
                    touching[self._corner_of.get(g2 if g1 in self._ropes else g1, 0)] = True
        return touching

    def legs_touching(self):
        """Are the two robots' legs (hips to feet) in contact with each other now?"""
        for g1, g2 in self.data.contact.geom:
            g1, g2 = int(g1), int(g2)
            if g1 in self._legs and g2 in self._legs and self._corner_of[g1] != self._corner_of[g2]:
                return True
        return False

    def fallen(self):
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        flags = []
        for m in self.maps:
            flags.append(not finite or self.data.qpos[m['root'] + 2] < .55
                         or self.data.sensor(m['side'] + 'upvector_torso').data[2] < .6)
        # Keep the foundation's self-foot/leg contact termination rule.
        if not self.self_trip_falls:
            return np.array(flags)
        for g1, g2 in self.data.contact.geom:
            corner = self._trip.get(frozenset((int(g1), int(g2))))
            if corner is not None:
                flags[corner] = True
        return np.array(flags)

    def step(self, actions):
        if self.done:
            raise RuntimeError("first-fall termination requires an explicit reset")
        actions = np.asarray(actions)
        if actions.shape != (2, 29) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite with shape (2, 29)")
        impulses = np.zeros(2)
        scored = np.zeros(2)
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        for i, m in enumerate(self.maps):
            self.data.ctrl[m['act']] = self.pose + self.action_scale * actions[i]
        for _ in range(self.substeps):
            # Cache velocities before the constraint solver changes them on
            # impact. A motionless fist cannot score when an opponent falls in.
            velocities = {}
            for geom in self._tracked:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM,
                                         int(geom), velocity, 0)
                velocities[geom] = velocity[3:].copy()
            positions = self.data.geom_xpos.copy()
            torso_motion = []
            for site in self._torso_sites:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                         site, velocity, 0)
                torso_motion.append((self.data.site_xpos[site].copy(), velocity))
            punch_trace = getattr(self, 'diagnostic_punch_trace', None)
            if punch_trace is not None:
                punch_trace.before_substep(self, positions, velocities, torso_motion)
            self._physics_step()
            for warning in self._warnings:
                if self.data.warning[warning].number:
                    # Native MuJoCo can reset an invalid state internally.
                    # Such a reset must never become apparent trial survival.
                    raise FloatingPointError(f'Invalid paired physics: {warning.name}')
            present = np.zeros((2, 2), bool)
            per_hand = np.zeros((2, 2))
            closing = np.zeros((2, 2))
            advancing = np.zeros((2, 2))
            contact_times = getattr(self, 'diagnostic_contact_times', None)
            for k, (g1, g2) in enumerate(self.data.contact.geom):
                g1, g2 = int(g1), int(g2)
                # Optional passive diagnostic: first positive normal force at
                # native substep resolution; no physics/scoring modification.
                if contact_times is not None:
                    pair_key = tuple(sorted({g1, g2}))
                    if pair_key not in contact_times:
                        diagnostic_force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, k, diagnostic_force)
                        if diagnostic_force[0] > 0:
                            contact_times[pair_key] = float(self.data.time)
                # A legal hit: one geom a fist, the other a target of the
                # fist owner's opponent (corner 0 before corner 1, as before).
                hits = sorted(self._fist_owner[f] + (t,) for f, t in ((g1, g2), (g2, g1))
                              if f in self._fist_owner and self._target_of.get(t) == 1 - self._fist_owner[f][0])
                for i, hand, target in hits:
                    fist = self.fists[i][hand]
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, k, force)
                    per_hand[i, hand] += max(0., force[0]) * self.model.opt.timestep
                    present[i, hand] = True
                    direction = positions[target] - positions[fist]
                    direction /= max(np.linalg.norm(direction), 1e-8)
                    closing[i, hand] = max(closing[i, hand],
                        float((velocities[fist] - velocities[target]) @ direction))
                    origin, velocity = torso_motion[i]
                    torso_at_fist = velocity[3:] + np.cross(velocity[:3], positions[fist] - origin)
                    advancing[i, hand] = max(advancing[i, hand],
                        float((velocities[fist] - torso_at_fist) @ direction))
            impulses += per_hand.sum(axis=1)
            fallen = self.fallen()
            scored += self.scorecard.update(present, per_hand, closing, advancing,
                self.data.time, self.model.opt.timestep, both_up=not fallen.any())
            if punch_trace is not None:
                punch_trace.after_substep(self, present, per_hand, closing, advancing)
            if fallen.any():
                self.done = True
                break
        self.last_action = actions.copy()
        self.phase = (self.phase + 2 * np.pi * self.dt * self.frequency + np.pi) % (2*np.pi) - np.pi
        self.elapsed = float(self.data.time)
        return dict(done=self.done, fallen=self.fallen(), legal_contact_impulse=impulses,
                    scored=scored)

    def _physics_step(self):
        """Native production dynamics; experimental backends override this hook."""
        mujoco.mj_step(self.model, self.data)
