"""The arm skill starts as a no-op and steers only the punching arm."""
import jax
import numpy as np

from boxer.arm_skill import ARM_ACTION_SIZE, ARM_OBS_SIZE, ArmSkill, initial_arm_params
from boxer.spar_match import PunchArm


def test_fresh_arm_skill_outputs_zero_residual():
    skill = ArmSkill(initial_arm_params(jax.random.PRNGKey(0)))
    obs = np.random.default_rng(0).normal(size=ARM_OBS_SIZE).astype(np.float32)
    assert np.allclose(np.asarray(skill.act(obs, skill.key)), np.zeros(ARM_ACTION_SIZE))


def test_residual_moves_only_the_punching_arm_mirrored():
    pose = np.zeros(29)
    plain, steered = PunchArm(pose, 1., .02), PunchArm(pose, 1., .02)
    for arm in (plain, steered):
        arm.launch(1, .5, np.random.default_rng(1))
    steered.residual = np.array([.1, .2, .3, .4])
    a, b = plain.actions(), steered.actions()
    np.testing.assert_allclose(b[:7], a[:7])                       # left arm at guard
    np.testing.assert_allclose(b[7:11] - a[7:11], [.1, -.2, -.3, .4])   # right arm, mirrored
