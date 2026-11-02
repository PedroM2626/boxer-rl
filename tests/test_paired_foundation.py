"""Physical and observation contracts for the unassisted paired transfer."""
from types import SimpleNamespace

import jax
import mujoco
import numpy as np
import pytest

from boxer.curriculum_env import UnassistedG1, stage_config
from boxer.paired_foundation import PairedFoundation, build_pair


def test_closed_fist_replay_preserves_physics_and_coordinate_contract(foundation, pair):
    visual = build_pair(foundation, visual_fists=True)
    plain = pair.model
    assert (visual.nq, visual.nv, visual.nu, visual.npair, visual.neq) == (
        plain.nq, plain.nv, plain.nu, plain.npair, plain.neq)
    for field in ('body_mass', 'body_inertia', 'dof_damping', 'dof_armature',
                  'actuator_gainprm', 'actuator_biasprm', 'jnt_qposadr', 'qpos0'):
        np.testing.assert_array_equal(getattr(visual, field), getattr(plain, field))
    def contacts(model):
        return [(model.geom(a).name, model.geom(b).name)
                for a, b in zip(model.pair_geom1, model.pair_geom2)]
    assert contacts(visual) == contacts(plain)
    data = mujoco.MjData(visual)
    mujoco.mj_forward(visual, data)
    for side in ('red/', 'blue/'):
        for hand in ('left', 'right'):
            wrist = visual.body(f'{side}{hand}_wrist_yaw_link').id
            meshes = {}
            for g in range(visual.ngeom):
                if visual.geom_bodyid[g] != wrist or visual.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                name = visual.mesh(visual.geom_dataid[g]).name
                assert 'rubber_hand' not in name
                if '_hand_' in name:
                    assert visual.geom_contype[g] == visual.geom_conaffinity[g] == 0
                    meshes[name.split('_hand_')[-1]] = data.geom_xpos[g]
            assert len(meshes) == 8
            for tip in ('index_1_link', 'middle_1_link', 'thumb_2_link'):
                assert np.linalg.norm(meshes[tip] - meshes['palm_link']) < .08


@pytest.fixture(scope='module')
def foundation():
    config = stage_config('approach', impl='jax')
    config.noise_config.level = 0.
    return UnassistedG1('approach', config=config)


@pytest.fixture(scope='module')
def pair(foundation):
    return PairedFoundation(foundation)


def test_free_roots_and_learned_dynamics(pair, foundation):
    model = pair.model
    assert model.nu == 58 and model.nq == 72
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 2
    assert model.neq == 0 and not np.any(model.body_gravcomp)
    assert pair.substeps == 10 and model.opt.timestep == .002
    for mapping in pair.maps:
        np.testing.assert_allclose(model.actuator_gainprm[mapping['act']], foundation.mj_model.actuator_gainprm)
        np.testing.assert_allclose(model.dof_damping[mapping['v']], foundation.mj_model.dof_damping[6:])
        np.testing.assert_allclose(model.dof_armature[mapping['v']], foundation.mj_model.dof_armature[6:])
        for body in range(1, foundation.mj_model.nbody):
            name = foundation.mj_model.body(body).name
            target = model.body(mapping['side'] + name).id
            np.testing.assert_allclose(model.body_mass[target], foundation.mj_model.body_mass[body])
            np.testing.assert_allclose(model.body_inertia[target], foundation.mj_model.body_inertia[body])


def test_actor_observation_matches_upstream_noiseless_state(pair, foundation):
    pair.reset(seed=31, randomized=False)
    reference = mujoco.MjData(foundation.mj_model)
    reference.qpos[:] = np.asarray(foundation._init_q)
    reference.qpos[7:] += np.linspace(-.02, .02, 29)
    reference.qvel[:] = np.linspace(-.1, .1, foundation.mj_model.nv)
    reference.ctrl[:] = np.asarray(foundation._default_pose)
    mujoco.mj_forward(foundation.mj_model, reference)
    command = np.array([.1, -.05, .2])
    previous = np.linspace(-.1, .1, 29)
    phase = np.array([.3, .3 + np.pi])
    proxy = SimpleNamespace(qpos=reference.qpos, qvel=reference.qvel,
        sensordata=reference.sensordata, site_xmat=reference.site_xmat.reshape(-1,3,3),
        actuator_force=reference.actuator_force)
    info = dict(command=command, last_act=previous, phase=phase,
                rng=jax.random.PRNGKey(0), feet_air_time=np.zeros(2))
    expected = np.asarray(foundation._get_obs(proxy, info, np.zeros(2))['state'])
    for mapping in pair.maps:
        q, v = mapping['root'], mapping['rootv']
        pair.data.qpos[q:q+7] = reference.qpos[:7]
        pair.data.qpos[mapping['q']] = reference.qpos[7:]
        pair.data.qvel[v:v+6] = reference.qvel[:6]
        pair.data.qvel[mapping['v']] = reference.qvel[6:]
    pair.last_action[:] = previous
    pair.phase[:] = phase
    mujoco.mj_forward(pair.model, pair.data)
    actual = pair.observation(np.tile(command,(2,1)))
    assert actual.shape == (2,103)
    np.testing.assert_allclose(actual, np.tile(expected,(2,1)), atol=2e-6)


def test_first_fall_stops_both_and_no_external_support(pair):
    pair.reset(randomized=False)
    pair.data.qpos[pair.maps[0]['root']+2] = .4
    mujoco.mj_forward(pair.model,pair.data)
    result=pair.step(np.zeros((2,29)))
    assert result['done'] and result['fallen'][0]
    assert not np.any(pair.data.xfrc_applied) and not np.any(pair.data.qfrc_applied)
    with pytest.raises(RuntimeError,match='first-fall'):
        pair.step(np.zeros((2,29)))


def test_reset_swap_spawns_and_invalid_actions(pair):
    for swap in (False,True):
        pair.reset(seed=23,swap=swap,randomized=False)
        roots=[pair.data.qpos[m['root']:m['root']+2] for m in pair.maps]
        assert np.linalg.norm(roots[1]-roots[0]) == 2.
        commands=pair.approach_commands()
        assert (commands[:,0] > 0).all()
        assert not pair.fallen().any()
        np.testing.assert_allclose(pair.observation(commands)[:,70:99],0.)
    with pytest.raises(ValueError,match='finite'):
        pair.step(np.full((2,29),np.nan))


def test_real_fist_head_contact_has_solver_force(pair):
    pair.reset(randomized=False)
    fist=pair.model.geom('red/left_hand_collision').id
    head=pair.model.geom('blue/head_collision').id
    delta=pair.data.geom_xpos[fist]-pair.data.geom_xpos[head]
    q=pair.maps[1]['root']
    pair.data.qpos[q:q+3] += delta
    mujoco.mj_forward(pair.model,pair.data)
    forces=[]
    for index,contact in enumerate(pair.data.contact):
        if set(contact.geom)=={fist,head}:
            force=np.zeros(6)
            mujoco.mj_contactForce(pair.model,pair.data,index,force)
            forces.append(force[0])
    assert forces and max(forces)>0


def test_native_numerical_warning_invalidates_round(pair):
    pair.reset(randomized=False)
    pair.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    with pytest.raises(FloatingPointError,match='BADQACC'):
        pair.step(np.zeros((2,29)))
    pair.reset(randomized=False)


def test_active_fist_impact_scores_but_passive_contact_does_not(pair):
    pair.reset(randomized=False)
    fist=pair.model.geom('red/left_hand_collision').id
    head=pair.model.geom('blue/head_collision').id
    root=pair.maps[1]['root']
    pair.data.qpos[root:root+3] += (pair.data.geom_xpos[fist]+[.05,0,0]
                                   - pair.data.geom_xpos[head])
    mujoco.mj_forward(pair.model,pair.data)
    jacobian=np.zeros((3,pair.model.nv));rotation=np.zeros_like(jacobian)
    mujoco.mj_jacGeom(pair.model,pair.data,jacobian,rotation,fist)
    arm_dofs=pair.maps[0]['v'][15:]
    weights=jacobian[0,arm_dofs]
    pair.data.qvel[arm_dofs]=weights/max(weights@weights,1e-6)
    mujoco.mj_forward(pair.model,pair.data)
    assert not pair.fallen().any()
    outcome=pair.step(np.zeros((2,29)))
    assert not outcome['done']
    np.testing.assert_array_equal(outcome['scored'],[1.,0.])
    assert outcome['legal_contact_impulse'][0]>.5


def test_legs_touching_sees_only_leg_on_leg_contact_between_the_robots(pair):
    pair.reset(seed=0, randomized=False)
    assert not pair.legs_touching()                 # spawned 2 m apart
    red, blue = (m['root'] for m in pair.maps)
    # Blue stands with its pelvis 0.25 m in front of red's: the legs overlap.
    pair.data.qpos[blue:blue + 2] = pair.data.qpos[red:red + 2] + [.25, 0.]
    mujoco.mj_forward(pair.model, pair.data)
    assert pair.legs_touching()
    names = {pair.model.geom(g).name.split('/', 1)[1] for g in pair._legs}
    assert names == {f'{s}_{p}' for s in ('left', 'right')
                     for p in ('foot', 'shin', 'thigh', 'thigh_collision', 'hip_collision')}
