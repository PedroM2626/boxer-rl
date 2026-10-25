"""Signed surface separation must distinguish gaps from overlap without dynamics."""
import mujoco
import numpy as np
import pytest
from boxer.surface_audit import SurfaceBudget,signed_gap


def test_signed_surface_gap_and_overlap_are_not_center_distances():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/>'
        '<body pos=".5 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    before=data.qpos.copy();velocity=data.qvel.copy();budget=SurfaceBudget(0,2)
    result=signed_gap(model,data,0,1,budget)
    np.testing.assert_allclose(result['signed_gap_m'],.3,atol=1e-14)
    np.testing.assert_array_equal(data.qpos,before);np.testing.assert_array_equal(data.qvel,velocity)
    data.qpos[0]=.15;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    np.testing.assert_allclose(signed_gap(model,data,0,1,budget)['signed_gap_m'],-.05,atol=1e-14)
    with pytest.raises(RuntimeError):signed_gap(model,data,0,1,budget)
