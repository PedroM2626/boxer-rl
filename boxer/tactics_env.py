"""Batched GPU boxing for learning tactics on top of the sparring motor.

One MJX/Warp world holds both free G1s. Each control step (50 Hz):

1. each corner's *tactics* (7 numbers) choose footwork, punches (straight or
   hook) and the guard height: the
   learner's from PPO, the opponent's from a frozen snapshot of an earlier
   learner or from the scripted ring brain;
2. ``punch_arms`` (same primitives as the ``spar`` stage) turns a launch into
   arm joint targets;
3. the frozen ``spar`` motor turns footwork commands into leg/waist actions,
   exactly as in native ``PairedFoundation`` play;
4. physics runs five 4 ms substeps. Contact sensors flag fist-on-head/torso
   and chest-to-chest touches.

A *hit* is the first fist-on-target touch of a punch, during extension, with
the fist closing on the target at ≥ 0.5 m/s. This proxy is only the training
signal. Results are reported with the native impact scorecard
(``boxer.spar_match``).

The learner is always red. Spawns are randomized in distance, heading and
global yaw, so red/blue carries no positional advantage.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from boxer import scene
from boxer import moves
from boxer.curriculum_env import LEFT_ARM, UnassistedG1
from boxer.curriculum_train import inference
from boxer.paired_foundation import (STRIKE_SENSOR_TARGETS, build_pair,
                                     strike_sensor_name)

# left punch, right punch, aim, advance, lateral, hook (vs straight), high guard
ACTION_SIZE = 7
OBS_SIZE = 30
# Checkpoints before hooks and guards (2026-10-30/31) see the first 26 inputs
# and output the first 5 actions; ``tactics_policy`` adapts them.
LEGACY_ACTION_SIZE, LEGACY_OBS_SIZE = 5, 26
HEAD_VALUE, TORSO_VALUE = 1.0, .6
REACH = .62              # root distance where straight punches land


def tactics_network_factory():
    import functools
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=(128, 128),
                             value_hidden_layer_sizes=(256, 256))


def policy_sizes(params):
    """(observation, action) sizes of a saved tactics policy."""
    layers = params[1]["params"]
    last = layers[sorted(layers, key=lambda name: int(name.split("_")[1]))[-1]]
    return int(layers["hidden_0"]["kernel"].shape[0]), int(last["bias"].shape[0]) // 2


def tactics_policy(params, deterministic=False):
    """Inference function for a saved tactics policy (normalizer, policy, value).

    Accepts current (30 → 7) and legacy (26 → 5) policies: a legacy policy sees
    the first 26 observations, and its actions are padded with "straight, normal
    guard".
    """
    obs_size, action_size = policy_sizes(params)
    net = tactics_network_factory()(obs_size, action_size,
                                    preprocess_observations_fn=running_statistics.normalize)
    infer = ppo_networks.make_inference_fn(net)(params, deterministic=deterministic)
    if (obs_size, action_size) == (OBS_SIZE, ACTION_SIZE):
        return infer

    def legacy(obs, key):
        action, extras = infer(obs[..., :obs_size], key)
        pad = -jp.ones(action.shape[:-1] + (ACTION_SIZE - action_size,))
        return jp.concatenate([action, pad], axis=-1), extras
    return legacy


def widen_legacy(params, new_scale=-1.):
    """A 30 → 7 policy that acts exactly like a legacy 26 → 5 one.

    New inputs get zero weights (normalized by mean 0, std 1), new action means
    start at 0 ("straight, normal guard", since only positive values switch), and
    new action scales start at softplus(``new_scale``) + 0.001 for exploration.
    """
    norm, policy, value = params
    obs_size, action_size = policy_sizes(params)
    extra_obs, extra_act = OBS_SIZE - obs_size, ACTION_SIZE - action_size

    def widen_input(net):
        net = jax.tree.map(lambda x: x, net)
        kernel = net["params"]["hidden_0"]["kernel"]
        net["params"]["hidden_0"]["kernel"] = jp.concatenate(
            [kernel, jp.zeros((extra_obs, kernel.shape[1]), kernel.dtype)])
        return net

    policy, value = widen_input(policy), widen_input(value)
    layers = policy["params"]
    last = sorted(layers, key=lambda name: int(name.split("_")[1]))[-1]
    kernel, bias = layers[last]["kernel"], layers[last]["bias"]
    zeros = jp.zeros((kernel.shape[0], extra_act), kernel.dtype)
    layers[last]["kernel"] = jp.concatenate(
        [kernel[:, :action_size], zeros, kernel[:, action_size:], zeros], axis=1)
    layers[last]["bias"] = jp.concatenate(
        [bias[:action_size], jp.zeros(extra_act), bias[action_size:], jp.full(extra_act, new_scale)])
    pad = lambda x, v: jp.concatenate([x, jp.full((extra_obs,), v, x.dtype)])
    # The normalizer is frozen during fine-tuning; only mean and std are used.
    norm = norm.replace(mean=pad(norm.mean, 0.), std=pad(norm.std, 1.),
                        summed_variance=pad(norm.summed_variance, 0.))
    return norm, policy, value


def decode(tactics, distance, local):
    """Maps tactics in [-1, 1] to footwork, a punch request, punch type and guard.

    Turning toward the opponent is automatic; the policy chooses range and
    angle. Works for NumPy and JAX inputs (``local``: opponent in pelvis frame).
    Returns (command, want, hand, aim, move, high_guard); legacy 5-number
    tactics mean straight punches and the normal guard.
    """
    xp = jp if isinstance(tactics, jax.Array) else np
    tactics = xp.clip(tactics, -1, 1)
    if tactics.shape[-1] == LEGACY_ACTION_SIZE:
        tactics = xp.concatenate([tactics, -xp.ones(ACTION_SIZE - LEGACY_ACTION_SIZE)])
    direction = local / xp.maximum(distance, 1e-6)
    advance = xp.where(tactics[3] > 0, .4 * tactics[3], .3 * tactics[3])
    tangent = xp.stack([-direction[1], direction[0]])
    # Circling can turn the bearing faster than the motor yaws; fade it out
    # between 35 and 55 deg off, or the pair can drift until they face away
    # for good (docs/tactics.md, long rounds).
    bearing = xp.arctan2(local[1], local[0])
    circling = xp.clip((.96 - xp.abs(bearing)) / .35, 0., 1.)
    planar = advance * direction + .2 * tactics[4] * circling * tangent
    heading = xp.clip(bearing, -.5, .5)
    command = xp.clip(xp.concatenate([planar, heading[None]]),
                      xp.array([-.35, -.25, -.5]), xp.array([.55, .25, .5]))
    want = xp.maximum(tactics[0], tactics[1]) > 0
    hand = xp.where(tactics[1] > tactics[0], 1, 0)
    aim = (tactics[2] + 1) / 2
    move = xp.where(tactics[5] > 0, moves.HOOK, moves.STRAIGHT)
    return command, want, hand, aim, move, tactics[6] > 0


def observe(xp, own, opp, own_arm, opp_arm, clock):
    """Tactics observation for one corner (30 floats), NumPy or JAX.

    ``own``/``opp``: dict(pos (2,), rot (2, 2) pelvis->world, vel (2,),
    up, height). ``*_arm``: (age fraction, active, hand sign, aim, hook,
    high guard). The hook/guard flags come last, so the first 26 entries keep
    the legacy layout.
    """
    delta = opp["pos"] - own["pos"]
    distance = xp.linalg.norm(delta)
    local = own["rot"].T @ delta
    opp_facing = opp["rot"].T @ (-delta) / xp.maximum(distance, 1e-6)
    return xp.concatenate([
        local, distance[None], (distance - REACH)[None], opp_facing,
        own["rot"].T @ own["vel"], own["rot"].T @ opp["vel"],
        xp.stack([own["up"], own["height"], opp["up"], opp["height"]]),
        own_arm[:4], opp_arm[:4], xp.stack([clock, 1 - clock]),
        xp.stack([xp.minimum(distance, 3.0) ** 2, xp.sin(xp.arctan2(local[1], local[0]))]),
        own_arm[4:6], opp_arm[4:6],
    ])


class TacticsEnv(mjx_env.MjxEnv):
    """Single-agent view (red) of two-robot boxing with a frozen opponent."""

    def __init__(self, motor_params, opponent_params=None, *, scripted_fraction=1.0,
                 episode_seconds=20.0, worlds_hint=512, impl="warp",
                 opponent_deterministic=False, native_reset_fraction=0.0, scoring="punch",
                 fall_penalty=10.0):
        """``opponent_params``: one tactics checkpoint's params or a list (a pool).

        Each bout draws its opponent at reset: the scripted brain with
        ``scripted_fraction``, otherwise a uniformly chosen pool member.
        ``native_reset_fraction`` of bouts start like ``PairedFoundation.reset``
        (2 m apart, joints scaled by U(0.5, 1.5), random root velocity), the
        states native evaluation starts from. ``scoring``: "punch" (first touch
        of an extending punch, head 1.0 / torso 0.6; generations 0-3) or
        "native" (the ImpactScorecard rules, see ``step``).
        """
        if scoring not in ("punch", "native"):
            raise ValueError(f"unknown scoring {scoring!r}")
        self.scoring = scoring
        self.fall_penalty = float(fall_penalty)
        super().__init__(config_dict.create(ctrl_dt=.02, sim_dt=.004))
        self.foundation = UnassistedG1("spar")
        self._mj_model = build_pair(self.foundation, strike_sensors=True)
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=impl)
        self._impl = impl
        self.naconmax, self.njmax = worlds_hint * 64, 600
        self.motor = inference(self.foundation, motor_params)
        if opponent_params is not None and not isinstance(opponent_params, list):
            opponent_params = [opponent_params]
        self.opponents = [tactics_policy(p, opponent_deterministic) for p in (opponent_params or [])]
        self.scripted_fraction = 1.0 if not self.opponents else float(scripted_fraction)
        self.native_reset_fraction = float(native_reset_fraction)
        self.episode_steps = int(round(episode_seconds / self.dt))
        self.pose = jp.asarray(self.foundation._default_pose)
        self.scale = float(self.foundation._config.action_scale)
        self.stand_height = float(self.foundation._init_q[2])
        m = self._mj_model
        self.act, self.q, self.v, self.root, self.rootv = [], [], [], [], []
        self.linvel, self.gyro, self.upvec, self.imu, self.fists, self.targets = [], [], [], [], [], []
        hits = []
        for i, side in enumerate(scene.SIDES):
            names = [self.foundation.mj_model.actuator(k).name for k in range(29)]
            act = np.array([m.actuator(side + n).id for n in names])
            joints = m.actuator_trnid[act, 0]
            root = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        and m.joint(j).name.startswith(side))
            self.act.append(act)
            self.q.append(m.jnt_qposadr[joints])
            self.v.append(m.jnt_dofadr[joints])
            self.root.append(int(m.jnt_qposadr[root]))
            self.rootv.append(int(m.jnt_dofadr[root]))
            adr = lambda name: int(m.sensor_adr[m.sensor(side + name).id])
            self.linvel.append(adr("local_linvel_pelvis"))
            self.gyro.append(adr("gyro_pelvis"))
            self.upvec.append(adr("upvector_torso"))
            self.imu.append(m.site(side + "imu_in_pelvis").id)
            self.fists.append([m.geom(side + h).id for h in scene.HAND_GEOMS])
            other = scene.OPPONENT[side]
            self.targets.append([m.geom(other + t).id for t in STRIKE_SENSOR_TARGETS])
            hits.append([[int(m.sensor_adr[m.sensor(strike_sensor_name(side, h, t)).id])
                          for t in STRIKE_SENSOR_TARGETS] for h in scene.HAND_GEOMS])
        self.act, self.q, self.v = map(np.stack, (self.act, self.q, self.v))
        self.fists, self.targets = np.array(self.fists), np.array(self.targets)
        self.hit_adr = np.array(hits)                     # (corner, hand, target)
        self.clinch_adr = int(m.sensor_adr[m.sensor("clinch").id])

    # --- MjxEnv contract -------------------------------------------------
    @property
    def xml_path(self):
        return self.foundation.xml_path

    @property
    def action_size(self):
        return ACTION_SIZE

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def observation_size(self):
        return OBS_SIZE

    # --- domain randomization ---------------------------------------------
    def randomizer(self, strength=1.0):
        """``randomization_fn(model, rng)`` for ``wrap_for_brax_training``.

        Per world: floor friction U(0.4, 1.0) and robot-robot friction
        U(0.3, 0.9); contact time constants ×U(0.7, 1.5); joint friction
        ×U(0.5, 2) and armature ×U(1, 1.05); link masses ×U(0.9, 1.1) and
        torso +U(-1, 1) kg; position-actuator stiffness ×U(0.9, 1.1). The
        ranges follow Playground's G1 randomizer and widen to cover contact,
        where batched and native physics disagree. ``strength`` scales
        every interval around its nominal value.
        """
        m = self._mj_model
        floor = m.geom("floor").id
        floor_pairs = np.array([k for k in range(m.npair)
                                if floor in (m.pair_geom1[k], m.pair_geom2[k])])
        robot_pairs = np.array([k for k in range(m.npair) if k not in set(floor_pairs)])
        joint_dofs = np.concatenate(self.v)
        torsos = np.array([m.body(side + "torso_link").id for side in scene.SIDES])
        actuators = np.concatenate(self.act)

        def span(low, high, nominal=1.0):
            return nominal + strength * (low - nominal), nominal + strength * (high - nominal)

        def randomize(model, rng):
            @jax.vmap
            def draw(rng):
                keys = jax.random.split(rng, 8)
                friction = model.pair_friction
                lo, hi = span(.4, 1., .7)
                friction = friction.at[floor_pairs, :2].set(
                    jax.random.uniform(keys[0], (len(floor_pairs), 1), minval=lo, maxval=hi))
                lo, hi = span(.3, .9, .6)
                friction = friction.at[robot_pairs, :2].set(
                    jax.random.uniform(keys[1], (len(robot_pairs), 1), minval=lo, maxval=hi))
                lo, hi = span(.7, 1.5)
                solref = model.pair_solref.at[:, 0].multiply(
                    jax.random.uniform(keys[2], (m.npair,), minval=lo, maxval=hi))
                lo, hi = span(.5, 2.)
                frictionloss = model.dof_frictionloss.at[joint_dofs].multiply(
                    jax.random.uniform(keys[3], (len(joint_dofs),), minval=lo, maxval=hi))
                lo, hi = span(1., 1.05)
                armature = model.dof_armature.at[joint_dofs].multiply(
                    jax.random.uniform(keys[4], (len(joint_dofs),), minval=lo, maxval=hi))
                lo, hi = span(.9, 1.1)
                mass = model.body_mass * jax.random.uniform(keys[5], (m.nbody,), minval=lo, maxval=hi)
                lo, hi = span(-1., 1., 0.)
                mass = mass.at[torsos].add(jax.random.uniform(keys[6], (2,), minval=lo, maxval=hi))
                lo, hi = span(.9, 1.1)
                stiffness = jax.random.uniform(keys[7], (len(actuators),), minval=lo, maxval=hi)
                gain = model.actuator_gainprm.at[actuators, 0].multiply(stiffness)
                bias = model.actuator_biasprm.at[actuators, 1].multiply(stiffness)
                return friction, solref, frictionloss, armature, mass, gain, bias

            fields = ("pair_friction", "pair_solref", "dof_frictionloss", "dof_armature",
                      "body_mass", "actuator_gainprm", "actuator_biasprm")
            values = draw(rng)
            in_axes = jax.tree_util.tree_map(lambda _: None, model)
            in_axes = in_axes.tree_replace({name: 0 for name in fields})
            return model.tree_replace(dict(zip(fields, values))), in_axes

        return randomize

    # --- helpers ---------------------------------------------------------
    def _body(self, data, i):
        r = self.root[i]
        rot = data.site_xmat[self.imu[i]][:2, :2]
        rot = rot / jp.maximum(jp.linalg.norm(rot[:, 0]), 1e-6)
        return dict(pos=data.qpos[r:r + 2], rot=rot, vel=data.qvel[self.rootv[i]:self.rootv[i] + 2],
                    up=data.sensordata[self.upvec[i] + 2], height=data.qpos[r + 2] - self.stand_height)

    def _fallen(self, data):
        return jp.stack([(data.qpos[self.root[i] + 2] < .55) | (data.sensordata[self.upvec[i] + 2] < .6)
                         | ~jp.isfinite(data.qpos).all() for i in range(2)])

    def _arm_features(self, arm, i):
        active = arm["age"][i] >= 0
        extend, retract = moves.durations(jp, arm["move"][i], self.dt)
        return jp.stack([jp.maximum(arm["age"][i], 0) / (extend + retract),
                         active.astype(jp.float32),
                         jp.where(active, 2. * arm["hand"][i] - 1, 0.), arm["aim"][i] * active,
                         (active & (arm["move"][i] == moves.HOOK)).astype(jp.float32),
                         arm["high"][i].astype(jp.float32)])

    def _observations(self, data, info):
        bodies = [self._body(data, i) for i in range(2)]
        clock = info["step"] / self.episode_steps
        arm = info["arm"]
        return [observe(jp, bodies[i], bodies[1 - i], self._arm_features(arm, i),
                        self._arm_features(arm, 1 - i), clock) for i in range(2)]

    def _motor_obs(self, data, info, commands):
        rows = []
        for i in range(2):
            gravity = data.site_xmat[self.imu[i]].T @ jp.array([0., 0., -1.])
            rows.append(jp.concatenate([
                data.sensordata[self.linvel[i]:self.linvel[i] + 3],
                data.sensordata[self.gyro[i]:self.gyro[i] + 3], gravity, commands[i],
                data.qpos[self.q[i]] - self.pose, data.qvel[self.v[i]],
                info["last_motor"][i], jp.cos(info["phase"][i]), jp.sin(info["phase"][i])]))
        return jp.stack(rows)

    def _scripted(self, obs, rng, wait):
        """JAX twin of ``spar_match.RingBrain`` in tactics space."""
        gap = obs[3]
        advance = jp.where(gap > .08, jp.clip(gap * 2, .1, 1), jp.where(gap < -.12, -.5, 0.))
        aim_rng, hand_rng, wait_rng, pause_rng = jax.random.split(rng, 4)
        ready = (obs[15] == 0) & (wait <= 0) & (gap < .1) & (jp.abs(obs[25]) < .35)
        hand = jax.random.bernoulli(hand_rng)
        punch = jp.where(ready, 1., -1.)
        tactics = jp.stack([jp.where(hand, -1., punch), jp.where(hand, punch, -1.),
                            jp.where(jax.random.bernoulli(aim_rng), 1., -1.), advance, 0., -1., -1.])
        pause = jp.where(jax.random.bernoulli(pause_rng, .6),
                         jax.random.randint(wait_rng, (), 2, 12),
                         jax.random.randint(wait_rng, (), 20, 50))
        return tactics, jp.where(ready, pause, wait - 1)

    def _arms(self, arm, i, want, hand, aim, move, high, rng):
        """Launches/advances corner ``i``'s punch and returns its 14 arm actions.

        Both arms rest in the normal or high guard (``high``, chosen every
        step); the punching arm follows ``moves.punch_pose`` from and back to it.
        """
        jitter_rng, speed_rng = jax.random.split(rng)
        launch = want & (arm["age"][i] < 0)
        move = jp.where(launch, move, arm["move"][i])
        aim = jp.where(launch, aim, arm["aim"][i])
        target = moves.strike_pose(jp, move, aim,
                                   jax.random.uniform(jitter_rng, (4,), minval=-.06, maxval=.06))
        hand = jp.where(launch, hand, arm["hand"][i])
        target = jp.where(launch, target, arm["target"][i])
        speed = jp.where(launch, jax.random.uniform(speed_rng, minval=.9, maxval=1.15),
                         arm["speed"][i])
        age = jp.where(launch, 0., arm["age"][i])
        actions = []
        for side in range(2):
            guard = self.pose[LEFT_ARM + 7 * side:LEFT_ARM + 7 * side + 4]
            rest = moves.rest_pose(jp, moves.mirror(jp, guard, side), high)
            swing = moves.punch_pose(jp, move, aim, target, age, rest, self.dt)
            pose = moves.mirror(jp, jp.where((age >= 0) & (hand == side), swing, rest), side)
            actions.append(jp.concatenate([(pose - guard) / self.scale, jp.zeros(3)]))
        extend, _ = moves.durations(jp, move, self.dt)
        extending = (age >= 0) & (age < extend + 2)
        new_age = jp.where(age >= 0, age + speed, age)
        new_age = jp.where(moves.finished(jp, move, new_age, self.dt), -1., new_age)
        state = dict(age=new_age, hand=hand, target=target, speed=speed, aim=aim, move=move,
                     high=jp.asarray(high))
        return jp.concatenate(actions), state, launch, extending

    # --- reset / step ----------------------------------------------------
    def reset(self, rng):
        (rng, place_rng, yaw_rng, center_rng, opp_rng, freq_rng, wait_rng, native_rng, pool_rng,
         joint_rng, vel_rng) = jax.random.split(rng, 11)
        native = jax.random.bernoulli(native_rng, self.native_reset_fraction)
        distance = jp.where(native, 2.0, jax.random.uniform(place_rng, minval=.9, maxval=1.7))
        theta = jax.random.uniform(yaw_rng, minval=-jp.pi, maxval=jp.pi)
        center = jax.random.uniform(center_rng, (2,), minval=-.5, maxval=.5)
        noise = jax.random.uniform(yaw_rng, (2,), minval=-.25, maxval=.25) * ~native
        axis = jp.array([jp.cos(theta), jp.sin(theta)])
        qpos = jp.asarray(self._mj_model.qpos0)
        for i in range(2):
            r = self.root[i]
            pos = center + (-.5 if i == 0 else .5) * distance * axis
            yaw = theta + (0. if i == 0 else jp.pi) + noise[i]
            qpos = qpos.at[r:r + 3].set(jp.concatenate([pos, jp.array([self.stand_height])]))
            qpos = qpos.at[r + 3:r + 7].set(jp.array([jp.cos(yaw / 2), 0., 0., jp.sin(yaw / 2)]))
            scale = jp.where(native, jax.random.uniform(jax.random.fold_in(joint_rng, i), (29,),
                                                        minval=.5, maxval=1.5), 1.)
            qpos = qpos.at[self.q[i]].set(self.pose * scale)
        qvel = jp.zeros(self._mj_model.nv)
        for i in range(2):
            kick = jax.random.uniform(jax.random.fold_in(vel_rng, i), (6,), minval=-.5, maxval=.5)
            qvel = qvel.at[self.rootv[i]:self.rootv[i] + 6].set(kick * native)
        ctrl = jp.zeros(self._mj_model.nu)
        for i in range(2):
            ctrl = ctrl.at[self.act[i]].set(qpos[self.q[i]])
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel,
                                 ctrl=ctrl, impl=self._impl, naconmax=self.naconmax,
                                 njmax=self.njmax)
        data = mjx.forward(self._mjx_model, data)
        arm = dict(age=-jp.ones(2), hand=jp.zeros(2, jp.int32), target=jp.zeros((2, 4)),
                   speed=jp.ones(2), aim=jp.zeros(2), move=jp.zeros(2, jp.int32),
                   high=jp.zeros(2, bool))
        info = dict(rng=rng, step=jp.array(0), arm=arm, last_motor=jp.zeros((2, 29)),
                    phase=jp.array([[0., jp.pi], [0., jp.pi]]),
                    frequency=jax.random.uniform(freq_rng, (2, 1), minval=1.25, maxval=1.5),
                    landed=jp.zeros(2, bool), engaged=jp.zeros((2, 2), bool),
                    absent=jp.full((2, 2), .1), since_score=jp.full((2, 2), 1.), scripted_wait=jax.random.randint(wait_rng, (), 5, 30),
                    scripted=jax.random.bernoulli(opp_rng, self.scripted_fraction),
                    opponent=jax.random.randint(pool_rng, (), 0, max(len(self.opponents), 1)),
                    previous_fists=data.geom_xpos[self.fists], previous_targets=data.geom_xpos[self.targets])
        obs = self._observations(data, info)
        metrics = {k: jp.zeros(()) for k in (
            "hits", "hits_taken", "head_hits", "punches", "fell", "opponent_fell", "clinch", "distance")}
        return mjx_env.State(data, obs[0], jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        info, data = dict(state.info), state.data
        info["rng"], opp_rng, script_rng, arm_rng0, arm_rng1 = jax.random.split(info["rng"], 5)
        obs = self._observations(data, info)
        if self.opponents:
            pool = jp.stack([policy(obs[1], opp_rng)[0] for policy in self.opponents])
            learned = pool[info["opponent"]]
        else:
            learned = jp.zeros(ACTION_SIZE)
        scripted, wait = self._scripted(obs[1], script_rng, info["scripted_wait"])
        info["scripted_wait"] = wait
        opponent = jp.where(info["scripted"], scripted, learned)
        bodies = [self._body(data, i) for i in range(2)]
        commands, arm_actions, arm_states, launches, extending = [], [], [], [], []
        for i, (tactics, rng) in enumerate(((action, arm_rng0), (opponent, arm_rng1))):
            delta = bodies[1 - i]["pos"] - bodies[i]["pos"]
            distance = jp.linalg.norm(delta)
            command, want, hand, aim, move, high = decode(tactics, distance, bodies[i]["rot"].T @ delta)
            actions, arm_state, launch, ext = self._arms(info["arm"], i, want, hand, aim, move,
                                                         high, rng)
            commands.append(command)
            arm_actions.append(actions)
            arm_states.append(arm_state)
            launches.append(launch)
            extending.append(ext)
        info["arm"] = jax.tree.map(lambda a, b: jp.stack([a, b]), *arm_states)
        motor, _ = self.motor({"state": self._motor_obs(data, info, commands),
                               "privileged_state": jp.zeros((2, 216))}, jax.random.PRNGKey(0))
        motor = motor.at[:, LEFT_ARM:].set(jp.stack(arm_actions))
        ctrl = data.ctrl
        for i in range(2):
            ctrl = ctrl.at[self.act[i]].set(self.pose + self.scale * motor[i])

        def substep(carry, _):
            data, found, clinch = carry
            data = mjx.step(self._mjx_model, data.replace(ctrl=ctrl))
            found = jp.maximum(found, data.sensordata[self.hit_adr])
            return (data, found, jp.maximum(clinch, data.sensordata[self.clinch_adr])), None

        zeros = jp.zeros(self.hit_adr.shape)
        (data, found, clinch), _ = jax.lax.scan(substep, (data, zeros, jp.zeros(())), (), 5)

        # Clean hits: first touch of a punch, while extending, closing >= 0.5 m/s.
        fists, targets = data.geom_xpos[self.fists], data.geom_xpos[self.targets]
        fist_motion = (fists - info["previous_fists"]) / self.dt              # (2, hand, 3)
        target_motion = (targets - info["previous_targets"]) / self.dt        # (2, target, 3)
        towards = targets[:, None] - info["previous_fists"][:, :, None]       # (2, hand, target, 3)
        towards = towards / jp.maximum(jp.linalg.norm(towards, axis=-1, keepdims=True), 1e-6)
        closing = jp.sum((fist_motion[:, :, None] - target_motion[:, None]) * towards, axis=-1)
        if self.scoring == "native":
            # Mirrors ImpactScorecard: one event per hand contact onset, closing
            # >= 0.3 m/s on the target and >= 0.2 m/s relative to the own torso,
            # rearmed after 0.1 s apart, 0.2 s cooldown. Every event is worth 1.
            # (The 0.5 N s impulse test has no batched equivalent.)
            touching = (found > 0).any(axis=2)                                # (2, hand)
            torso = jp.stack([data.qvel[self.rootv[i]:self.rootv[i] + 3] for i in range(2)])
            own = jp.sum((fist_motion[:, :, None] - torso[:, None, None]) * towards, axis=-1)
            touched = found > 0
            closing_ok = jp.max(jp.where(touched, closing, -jp.inf), axis=2) >= .3
            advancing_ok = jp.max(jp.where(touched, own, -jp.inf), axis=2) >= .2
            absent = jp.where(touching, 0., info["absent"] + self.dt)
            engaged = info["engaged"] & ~(~touching & (absent >= .1))
            onset = touching & ~engaged
            events = onset & closing_ok & advancing_ok & (info["since_score"] >= .2) & ~self._fallen(data).any()
            info["engaged"] = engaged | touching
            info["absent"] = absent
            info["since_score"] = jp.where(events, 0., info["since_score"] + self.dt)
            scored = events.sum(axis=1).astype(jp.float32)
        else:
            punching_hand = jp.stack([jp.arange(2) == info["arm"]["hand"][i] for i in range(2)])
            clean = (found > 0) & (closing >= .5) & punching_hand[:, :, None]
            clean = clean & jp.stack(extending)[:, None, None]
            value = jp.array([HEAD_VALUE, TORSO_VALUE])
            best = jp.max(clean * value, axis=(1, 2))                         # (2,)
            landed = info["landed"] & (info["arm"]["age"] >= 0)               # reset per punch
            scored = jp.where(landed, 0., best)
            info["landed"] = landed | (best > 0)
        info["previous_fists"], info["previous_targets"] = fists, targets

        info["last_motor"] = motor
        info["phase"] = jp.fmod(info["phase"] + 2 * jp.pi * self.dt * info["frequency"] + jp.pi,
                                2 * jp.pi) - jp.pi
        info["step"] = info["step"] + 1
        bodies = [self._body(data, i) for i in range(2)]
        fell = self._fallen(data)
        distance = jp.linalg.norm(bodies[1]["pos"] - bodies[0]["pos"])
        in_range = jp.exp(-((distance - REACH) / .15) ** 2)
        reward = (2. * scored[0] - 1.5 * scored[1] - self.fall_penalty * fell[0] + 5. * fell[1] * ~fell[0]
                  + self.dt * (.3 * in_range - 1. * clinch) - .02 * launches[0])
        done = fell.any() | (info["step"] >= self.episode_steps)
        metrics = dict(hits=scored[0], hits_taken=scored[1], head_hits=(scored[0] >= HEAD_VALUE) * 1.,
                       punches=launches[0] * 1., fell=fell[0] * 1., opponent_fell=fell[1] * 1.,
                       clinch=clinch, distance=distance)
        obs = self._observations(data, info)
        reward = jp.where(jp.isfinite(reward), reward, -10.)
        return state.replace(data=data, obs=obs[0], reward=reward, done=done.astype(jp.float32),
                             metrics=metrics, info=info)
