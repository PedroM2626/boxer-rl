"""A failed attempted GPU substep must already have a durable ledger callback."""
from types import SimpleNamespace
import numpy as np
import pytest
from boxer.paired_warp_reference import WarpPairedReference


def test_reference_debits_before_the_kernel_can_fail():
    pair = object.__new__(WarpPairedReference)
    pair.gpu_substeps = pair.solver_flag_observations = pair.solver_flag_bits_seen = 0
    pair.gpu_forward_calls = 3; pair.native_reset_forward_calls = 4
    records = []; pair.gpu_account = lambda value: records.append(value)
    def fail(ctrl):
        assert records[-1]['gpu_world_substeps'] == 1
        raise FloatingPointError('simulated failed kernel')
    pair.physics = SimpleNamespace(step=fail)
    pair.data = SimpleNamespace(ctrl=np.zeros(58))
    with pytest.raises(FloatingPointError): pair._physics_step()
    assert records[-1] == dict(gpu_world_substeps=1, gpu_forward_calls=3,
        native_reset_forward_calls=4, solver_limit_world_checks=0, overflow_bits_seen=0)
