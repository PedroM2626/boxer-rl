"""GPU backend migration must retain every learned state and reject false origins."""
from pathlib import Path
import copy
import jax
import numpy as np
import pytest
from boxer.striking_gpu_profile import SOURCE, SOURCE_SHA, prepare_gpu_state
from boxer.striking_gpu_profile import BATCH_SOURCE, BATCH_SOURCE_SHA, prepare_scaled_state
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked


def test_gpu_transition_preserves_all_source_leaves_and_rejects_relabeling():
    original, _ = load_checked(Path(SOURCE), SOURCE_SHA)
    result = prepare_gpu_state(original, SOURCE_SHA)
    for key in original:
        assert jax.tree_util.tree_structure(original[key]) == jax.tree_util.tree_structure(result[key])
        for before, after in zip(jax.tree_util.tree_leaves(original[key]), jax.tree_util.tree_leaves(result[key])):
            np.testing.assert_array_equal(before, after)
    require_ppo_checkpoint(result)
    changed = copy.deepcopy(result); changed['gpu_collection_profile']['solver_iterations'] = 30
    with pytest.raises(ValueError): require_ppo_checkpoint(changed)
    changed = copy.deepcopy(result); changed['gpu_batched_ppo'] = False
    with pytest.raises(ValueError): require_ppo_checkpoint(changed)
    with pytest.raises(ValueError): prepare_gpu_state(original, '0'*64)


def test_scaled_transition_preserves_prefix_and_enforces_original_updates_per_credit():
    original, _ = load_checked(Path(BATCH_SOURCE), BATCH_SOURCE_SHA)
    initial = prepare_scaled_state(original, BATCH_SOURCE_SHA)
    for key in original.keys()-{'gpu_collection_profile'}:
        for before, after in zip(jax.tree_util.tree_leaves(original[key]), jax.tree_util.tree_leaves(initial[key])):
            np.testing.assert_array_equal(before, after)
    changed = copy.deepcopy(initial)
    changed['gpu_collection_profile']['updates_per_rollout'] = 16
    with pytest.raises(ValueError): require_ppo_checkpoint(changed)
    changed = copy.deepcopy(initial)
    for key in ('consumed_steps', 'optimized_steps', 'world_control_steps'): changed[key] += 16384
    changed['mixed_credits_by_mode_and_corner'] = (np.asarray(changed['mixed_credits_by_mode_and_corner'])+4096).tolist()
    for role in ('actor', 'critic'):
        state = changed['optimizer'][role+'_adam']
        changed['optimizer'][role+'_adam'] = (state[0]._replace(count=state[0].count+16), state[1])
    with pytest.raises(ValueError): require_ppo_checkpoint(changed)
    for role in ('actor', 'critic'):
        state = changed['optimizer'][role+'_adam']
        changed['optimizer'][role+'_adam'] = (state[0]._replace(count=state[0].count+240), state[1])
    require_ppo_checkpoint(changed)
