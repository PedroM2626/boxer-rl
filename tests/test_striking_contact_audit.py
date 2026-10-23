"""A contact-force intervention must suppress exactly its intended solver pairs."""
import mujoco
import numpy as np

from boxer.paired_foundation import PairedFoundation
from tools.striking_contact_audit import incoming_legal_pairs
from test_paired_foundation import foundation


def test_incoming_force_intervention_is_local_and_removes_solver_force(foundation):
    pair=PairedFoundation(foundation);pair.reset(randomized=False)
    fist=pair.model.geom('blue/left_hand_collision').id
    head=pair.model.geom('red/head_collision').id
    pair.data.qpos[pair.maps[0]['root']:pair.maps[0]['root']+3] += (
        pair.data.geom_xpos[fist]-pair.data.geom_xpos[head])
    def force(subject):
        mujoco.mj_forward(subject.model,subject.data)
        total=0.
        for i,contact in enumerate(subject.data.contact):
            if set(contact.geom)=={fist,head}:
                vector=np.zeros(6);mujoco.mj_contactForce(subject.model,subject.data,i,vector);total+=vector[0]
        return total
    assert force(pair)>0
    changed=incoming_legal_pairs(pair,0)
    altered=PairedFoundation(foundation,diagnostic_excluded_pairs=changed)
    altered.data.qpos[:]=pair.data.qpos
    assert len(changed)==4 and altered.model.npair==pair.model.npair-4
    assert force(altered)==0
    np.testing.assert_array_equal(pair.model.body_mass,altered.model.body_mass)
    np.testing.assert_array_equal(pair.model.body_inertia,altered.model.body_inertia)
    assert altered.model.neq==0 and not np.any(altered.model.body_gravcomp)
