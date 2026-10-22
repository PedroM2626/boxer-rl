"""Batched first-fall diagnosis; never a promotion certificate.

Frozen evaluation physics and commands match the original standing gate.
Each reset is shared between disturbed/undisturbed cases for paired comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1, stage_config
from boxer.curriculum_train import inference, load_policy_params, contract


def diagnose(checkpoints, seeds=(7, 23, 31, 47), worlds=32):
    cfg = stage_config("balance")
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.njmax = 90
    env = UnassistedG1(config=cfg)
    cases = [(seed, disturbed) for seed in seeds for disturbed in (False, True)]
    count = len(cases) * worlds
    keys = jp.concatenate([jax.random.split(jax.random.PRNGKey(seed), worlds)
                           for seed, _ in cases])
    impulse_mask = jp.repeat(jp.array([d for _, d in cases]), worlds)
    contact_ids = [env.mj_model.sensor(name).adr[0] for name in (
        "right_foot_left_foot_found", "left_foot_right_shin_found", "right_foot_left_shin_found")]
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)

    def command_zero(state):
        state.info["command"] = jp.zeros(3)
        contact = jp.array([state.data.sensordata[env.mj_model.sensor_adr[s]] > 0
                            for s in env._feet_floor_found_sensor])
        return state.replace(obs=env._get_obs(state.data, state.info, contact))

    commands = jax.vmap(command_zero)
    results = []
    for checkpoint in checkpoints:
        params, metadata = load_policy_params(checkpoint)
        if contract(env) != metadata["contract"]:
            raise ValueError("diagnostic model contract mismatch")
        policy = inference(env, params)

        @jax.jit
        def rollout(keys):
            state = reset(keys)
            alive = jp.ones(count, dtype=bool)
            def body(carry, tick):
                state, alive = carry
                state = commands(state)
                state = state.replace(data=state.data.replace(qvel=state.data.qvel.at[:, :2].add(
                    (tick == 200) * impulse_mask[:, None] * jp.array([.2, -.2]))))
                action, _ = policy(state.obs, jax.random.PRNGKey(0))
                previous = state
                state = step(state, action)
                def select(before, after):
                    if not after.ndim or after.shape[0] != count:
                        return after
                    return jp.where(alive.reshape((count,) + (1,) * (after.ndim - 1)), after, before)
                state = jax.tree.map(select, previous, state)
                up = state.data.sensordata[:, env.mj_model.sensor("upvector_torso").adr[0] + 2]
                height = state.data.qpos[:, 2]
                collision = jp.any(state.data.sensordata[:, jp.array(contact_ids)] > 0, axis=1)
                finite = jp.all(jp.isfinite(state.data.qpos), axis=1) & jp.all(jp.isfinite(state.data.qvel), axis=1)
                died = alive & (state.done > 0)
                obs = jp.stack([died, up < .6, height < .55, collision, ~finite], axis=-1)
                return (state, alive & ~died), obs
            return jax.lax.scan(body, (state, alive), jp.arange(500))[1]

        trace = np.asarray(rollout(keys))
        died = trace[:, :, 0].astype(bool)
        first = np.where(died.any(axis=0), died.argmax(axis=0), 500)
        failures = trace[np.minimum(first, 499), np.arange(count), 1:]
        rows = []
        for index, (seed, disturbed) in enumerate(cases):
            sl = slice(index * worlds, (index + 1) * worlds)
            when, reasons = first[sl], failures[sl]
            failed = when < 500
            rows.append(dict(seed=seed, disturbed=disturbed,
                survivors=int((~failed).sum()), worlds=worlds,
                before_impulse=int((when < 200).sum()),
                after_impulse=int(((when >= 200) & failed).sum()),
                within_one_second_of_impulse=int(((when >= 200) & (when < 250)).sum()),
                first_fall_seconds=np.where(failed, (when + 1) * .02, 10.).tolist(),
                reasons={name: int((reasons[:, i] & failed).sum()) for i, name in enumerate(
                    ("tilt", "low_pelvis", "leg_collision", "nonfinite"))}))
        results.append(dict(checkpoint=str(checkpoint), stage_steps=metadata["stage_steps"],
                            sha256=metadata["sha256"], trials=rows))
        print(dict(step=metadata["stage_steps"], survivors=[r["survivors"] for r in rows]), flush=True)
    return dict(kind="first_fall_diagnostic", promotion_eligible=False, physics_batch_worlds=count,
                note="Failure reasons may overlap; seeds 31/47 are diagnostic holdouts.", results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, diagnose(args.checkpoints))


if __name__ == "__main__":
    main()
