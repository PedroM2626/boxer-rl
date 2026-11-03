"""GPU PPO for an unassisted full-body curriculum stage.

Checkpoints contain normalizer, actor and critic, with a verified model contract.
Brax does not expose Adam state here; restarting is explicitly a warm start.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks, train as ppo
from mujoco_playground import wrapper

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1, TRAINABLE_STAGES, PLAYGROUND_REVISION, MENAGERIE_REVISION, stage_config
from boxer.curriculum_control import consecutive_passes, promotion_ready


class PromotionComplete(Exception):
    """End PPO at a verified checkpoint once the physical gates pass twice."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def jax_compat():
    if not hasattr(jax, "device_put_replicated"):
        def replicate(tree, devices):
            stacked = jax.tree.map(lambda x: jp.stack([x] * len(devices)), tree)
            mesh = jax.sharding.Mesh(tuple(devices), axis_names=("replica",))
            return jax.device_put(stacked, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("replica")))
        jax.device_put_replicated = replicate


def network_factory():
    return functools.partial(networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 128, 128),
        policy_obs_key="state", value_obs_key="privileged_state")


def contract(env):
    m = env.mj_model
    return dict(playground_revision=PLAYGROUND_REVISION,
                menagerie_revision=MENAGERIE_REVISION,
                actuators=[m.actuator(i).name for i in range(m.nu)],
                position_gains=m.actuator_gainprm[:, 0].tolist(),
                default_pose=np.asarray(env._default_pose).tolist(),
                action_scale=float(env._config.action_scale), dt=env.dt,
                observation_size={k: list(v) if isinstance(v, tuple) else int(v)
                                  for k, v in env.observation_size.items()},
                policy_layers=[256, 128, 128], value_layers=[256, 128, 128],
                normalization_variance_epsilon=0.01, assistance=False)


def save_policy(root, params, metadata):
    payload = pickle.dumps(jax.device_get(params), protocol=pickle.HIGHEST_PROTOCOL)
    filename = f"step-{metadata['stage_steps']}.pkl"
    metadata = dict(metadata, sha256=hashlib.sha256(payload).hexdigest(), file=filename,
                    optimizer_saved=False)
    atomic_bytes(root / filename, payload)
    atomic_json(root / (filename + ".json"), metadata)
    atomic_json(root / "latest.json", metadata)


def load_policy_params(root):
    root = Path(root)
    manifest = json.loads((root / "latest.json").read_text())
    payload = (root / manifest["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError(f"checkpoint checksum mismatch: {root}")
    return pickle.loads(payload), manifest


def inference(env, params):
    net = network_factory()(env.observation_size, env.action_size,
                           preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(params, deterministic=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=TRAINABLE_STAGES, default="balance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200_000_000)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--checkpoints", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--restart-from", type=Path,
                        help="one-time policy rollback; retain own consumed-step counter")
    parser.add_argument("--force-restart", action="store_true",
                        help="explicitly discard updates after the same recovery seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tracking-sigma", type=float,
                        help="training-only velocity reward width for isolated comparisons; "
                             "omission preserves the task default, evaluation is unchanged")
    parser.add_argument("--lateral-practice-probability", type=float, default=0.0,
                        help="fraction of moving easy-walking commands replaced by pure lateral requests")
    parser.add_argument("--phantom-shins", action="store_true",
                        help="spar: opponent shins sweep into the robot's legs (curriculum_env.PHANTOM_*)")
    parser.add_argument("--rapid-punches", action="store_true",
                        help="spar: near back-to-back punches with arm-skill-sized corrections "
                             "(curriculum_env.RAPID_PUNCH_*)")
    parser.add_argument("--striker-arms", type=Path,
                        help="spar: npz of striker arm actions (striker_train arms) replayed in a share "
                             "of episodes (curriculum_env.STRIKER_ARM_FRACTION)")
    parser.add_argument("--publish-run", type=Path)
    parser.add_argument("--preview-snapshots", action="store_true")
    parser.add_argument("--adaptive-gates", action="store_true")
    parser.add_argument("--gate-every", type=int, default=5_000_000)
    parser.add_argument("--total-offset", type=int, default=0,
                        help="transitions from earlier stages, excluding this stage")
    parser.add_argument("--git-executable", default="git")
    parser.add_argument("--git-worktree")
    args = parser.parse_args(argv)
    if args.gate_every <= 0 or args.total_offset < 0 or args.learning_rate <= 0:
        raise ValueError("gate interval must be positive and total offset non-negative")
    if args.force_restart and not (args.restart_from and args.warm_start):
        raise ValueError("--force-restart requires --warm-start and --restart-from")
    if args.tracking_sigma is not None and (
            not np.isfinite(args.tracking_sigma) or args.tracking_sigma <= 0):
        raise ValueError("tracking sigma must be finite and positive")
    if (not np.isfinite(args.lateral_practice_probability)
            or not 0 <= args.lateral_practice_probability <= 1
            or (args.lateral_practice_probability and args.stage != "walk_easy")):
        raise ValueError("lateral practice requires easy walking and a finite probability in [0, 1]")
    jax_compat()
    if args.worlds < 1 or args.worlds & (args.worlds - 1):
        raise ValueError("worlds must be a positive power of two")
    config = stage_config(args.stage)
    if args.tracking_sigma is not None:
        config.reward_config.tracking_sigma = args.tracking_sigma
    if args.lateral_practice_probability:
        config.lateral_practice_probability = args.lateral_practice_probability
    if args.phantom_shins:
        config.phantom_shins = True
    if args.rapid_punches:
        config.rapid_punches = True
    if args.striker_arms:
        config.striker_arms = str(args.striker_arms)
    config.naconmax = max(8192, args.worlds * 16)
    env = UnassistedG1(args.stage, config=config)
    env_contract = contract(env)
    checkpoint = args.output / "checkpoint"
    if (checkpoint / "latest.json").exists() and not args.warm_start:
        raise ValueError("output already has a policy; use a new directory or --warm-start")
    params, initial, offset = None, None, 0
    recovery_seed = None
    if args.warm_start or args.init_from:
        source = checkpoint if args.warm_start else args.init_from
        params, initial = load_policy_params(source)
        if initial["contract"] != env_contract:
            raise ValueError("policy/model contract differs; refusing an unsafe transplant")
        if args.warm_start:
            if initial["stage"] != args.stage:
                raise ValueError("warm start stage differs")
            offset = initial["stage_steps"]
        print("Warm start: actor, critic and normalizer restored; Adam and RNG are fresh.", flush=True)
        recovery_seed = initial.get("recovery_seed_sha256")
    source_steps = initial["stage_steps"] if initial else 0
    if args.restart_from:
        if not args.warm_start:
            raise ValueError("--restart-from requires --warm-start to preserve consumed steps")
        selected_params, selected = load_policy_params(args.restart_from)
        if selected["contract"] != env_contract or selected["stage"] != args.stage:
            raise ValueError("recovery seed stage/model contract differs")
        if args.force_restart or recovery_seed != selected["sha256"]:
            params, source_steps = selected_params, selected["stage_steps"]
            recovery_seed = selected["sha256"]
            print(f"Recovery seed {source_steps}; consumed counter retained at {offset}.", flush=True)
    if offset >= args.steps:
        raise ValueError("stage budget already reached")
    batch_size = max(128, args.worlds // 4)
    block = batch_size * 20 * 4
    blocks = (args.steps - offset) // block
    if blocks < 1 or args.checkpoints < 1:
        raise ValueError("remaining budget must fit a PPO block and checkpoints must be positive")
    epochs = min(args.checkpoints, blocks)
    # Round DOWN, so a ceiling is never silently exceeded by PPO's ceil schedule.
    actual = epochs * (blocks // epochs) * block
    config = dict(stage=args.stage, requested_steps=args.steps, offset_steps=offset,
                  invocation_steps=actual, worlds=args.worlds, seed=args.seed,
                  batch_size=batch_size, num_minibatches=4, unroll_length=20,
                  contract=env_contract, environment=env._config.to_dict(),
                  optimizer_resume=False,
                  protocol="recovery-v3", adaptive_gates=args.adaptive_gates,
                  learning_rate=args.learning_rate, full_reset=True,
                  policy_source_steps=source_steps, recovery_seed_sha256=recovery_seed,
                  forced_restart=args.force_restart,
                  gate_every=args.gate_every, total_offset=args.total_offset,
                  initialized_from=str(args.init_from) if args.init_from else None)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "config.json", config)
    started = time.monotonic()
    last_publication = -5_000_000
    history_path = args.output / "gate_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_gate = history[-1]["stage_steps"] - offset if history else -args.gate_every

    def job_phase(phase, step, **extra):
        if args.publish_run and (args.publish_run / "state.json").exists():
            path = args.publish_run / "state.json"
            state = json.loads(path.read_text())
            state.update(phase=phase, stage=args.stage,
                         total_steps=args.total_offset + offset + int(step),
                         updated_utc=utc(), **extra)
            atomic_json(path, state)

    def assess(step):
        nonlocal last_gate
        manifest = json.loads((checkpoint / "latest.json").read_text())
        current = offset + int(step)
        if (history and history[-1]["stage_steps"] == current
                and history[-1]["checkpoint_sha256"] == manifest["sha256"]):
            last_gate = int(step)
            return
        gate_path = args.output / "gates" / f"step-{current}.json"
        job_phase("evaluating", step)
        gate_log = args.output / "eval.log"
        log_offset = gate_log.stat().st_size if gate_log.exists() else 0
        with gate_log.open("a") as stream:
            subprocess.run([sys.executable, "-u", "-m", "boxer.curriculum_eval",
                str(checkpoint), "--output", str(gate_path),
                "--video", str(args.output / "evaluation.mp4")],
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        with gate_log.open("rb") as stream:
            stream.seek(log_offset)
            if b"overflow -" in stream.read().lower():
                raise RuntimeError("Evaluation physics allocation overflow; gate is invalid")
        result = json.loads(gate_path.read_text())
        if (result["stage_steps"] != current
                or result["checkpoint_sha256"] != manifest["sha256"]):
            raise ValueError("gate result does not identify the current checkpoint")
        history.append(result)
        atomic_json(history_path, history)
        atomic_json(args.output / "evaluation.json", result)
        ready = promotion_ready(history, args.stage)
        atomic_json(args.output / "training_result.json", dict(
            status="promoted" if ready else "gate_pending", stage=args.stage,
            stage_steps=current, total_steps=args.total_offset + current,
            consecutive_passes=consecutive_passes(history, args.stage),
            required_passes=2, updated_utc=utc()))
        print(json.dumps(dict(event="physical_gate", stage_steps=current,
            passed=result["passed"], consecutive_passes=consecutive_passes(history, args.stage))), flush=True)
        last_gate = int(step)
        job_phase("gate_passed" if ready else "training", step,
                  consecutive_passes=consecutive_passes(history, args.stage))
        if args.publish_run:
            from boxer.curriculum_job import publish
            publish(args.publish_run, args.stage, args.git_executable, args.git_worktree)
        if ready:
            raise PromotionComplete

    def checkpoint_fn(step, make_policy, policy_params):
        del make_policy
        leaves = jax.tree.leaves(jax.device_get(policy_params))
        if not all(np.isfinite(x).all() for x in leaves):
            raise FloatingPointError("non-finite policy; preserving the previous checkpoint")
        save_policy(checkpoint, policy_params, dict(stage=args.stage,
            stage_steps=offset + int(step), invocation_steps=int(step),
            total_steps=args.total_offset + offset + int(step),
            protocol="recovery-v3", recovery_seed_sha256=recovery_seed,
            policy_source_steps=source_steps, updated_utc=utc(), contract=env_contract,
            training_tracking_sigma=float(env._config.reward_config.tracking_sigma),
            lateral_practice_probability=args.lateral_practice_probability))
        if args.adaptive_gates and int(step) == 0:
            assess(step)

    def progress(step, metrics):
        nonlocal last_publication
        record = dict(stage=args.stage, stage_steps=offset + int(step),
                      total_steps=args.total_offset + offset + int(step),
                      updated_utc=utc(), elapsed_seconds=time.monotonic() - started,
                      metrics={k: float(np.asarray(v)) for k, v in metrics.items()})
        atomic_json(args.output / "progress.json", record)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if args.adaptive_gates:
            if int(step) - last_gate >= args.gate_every or int(step) >= actual:
                assess(step)
            return
        if ((args.publish_run or args.preview_snapshots) and int(step) > 0
                and (int(step) - last_publication >= 5_000_000
                     or int(step) >= actual)):
            if args.preview_snapshots:
                # The parent waits with Adam resident; evaluation does not
                # interrupt/restart optimization and has a small separate batch.
                with (args.output / "preview.log").open("a") as stream:
                    subprocess.run([sys.executable, "-u", "-m", "boxer.curriculum_eval",
                        str(checkpoint), "--worlds", "8", "--preview",
                        "--output", str(args.output / "preview.json"),
                        "--video", str(args.output / "preview.mp4")],
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
            if args.publish_run:
                from boxer.curriculum_job import publish
                publish(args.publish_run, args.stage, args.git_executable, args.git_worktree)
            last_publication = int(step)

    print(json.dumps(config), flush=True)
    try:
        ppo.train(environment=env, num_timesteps=actual,
            num_envs=args.worlds, episode_length=500,
            wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
            learning_rate=args.learning_rate, entropy_cost=0.005, discounting=0.97,
            unroll_length=20, batch_size=batch_size, num_minibatches=4,
            num_updates_per_batch=4, normalize_observations=True,
            normalize_observations_std_eps=0.01,
            max_grad_norm=1.0, network_factory=network_factory(), seed=args.seed,
            num_evals=epochs + 1, num_eval_envs=8, run_evals=False,
            restore_params=params, policy_params_fn=checkpoint_fn, progress_fn=progress)
    except PromotionComplete:
        print("Physical gate passed twice; advancing without waiting for the budget ceiling.", flush=True)
        return 0
    if args.adaptive_gates:
        atomic_json(args.output / "training_result.json", dict(
            status="budget_reached", stage=args.stage, stage_steps=offset + actual,
            total_steps=args.total_offset + offset + actual,
            consecutive_passes=consecutive_passes(history, args.stage),
            required_passes=2, updated_utc=utc()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
