"""A striker's starting point: imitate the champion stack, natively.

``collect`` plays native ring bouts in worker processes like
``native_finetune``: the learner corner boxes with the champion tactics, the
scripted moves and the arm skill, against the champion pool and scripted
opponents. Every control step it records the striker observation
(``boxer.striker``) and what the stack did: the tactics' advance and circle
outputs and the arm targets it actually sent, in striker units. It also
records the native reward, for a first value function.

``clone`` fits a striker to those records (the policy mean by least
squares, the value by discounted returns) and saves a checkpoint that
``spar_match`` and ``native_finetune --striker`` load.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from boxer.native_finetune import (FoundationStub, NativeBout, _configure_worker_env, _receive,
                                   export_native_model)


def collect_worker(conn, seed, steps, motor_path, tactics_path, arm_path, pool, scripted_fraction,
                   high_guard_fraction, fall_penalty, model_path, constants, output):
    try:
        _collect(seed, steps, motor_path, tactics_path, arm_path, pool, scripted_fraction,
                 high_guard_fraction, fall_penalty, model_path, constants, output, conn)
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise


def _collect(seed, steps, motor_path, tactics_path, arm_path, pool, scripted_fraction,
             high_guard_fraction, fall_penalty, model_path, constants, output, conn):
    _configure_worker_env()
    import jax
    import mujoco
    from boxer.arm_skill import ArmSkill
    from boxer.curriculum_train import inference, load_policy_params
    from boxer.paired_foundation import PairedFoundation
    from boxer.spar_match import PolicyBrain
    from boxer.striker import StrikerArm, striker_observe, to_unit

    foundation = FoundationStub(constants)
    motor = jax.jit(inference(foundation, load_policy_params(motor_path)[0]))
    pair = PairedFoundation(foundation, model=mujoco.MjModel.from_binary_path(str(model_path)))
    rng = np.random.default_rng(seed)
    opponents = []
    for k, path in enumerate(pool):
        brain = PolicyBrain(load_policy_params(path)[0], deterministic=bool(k % 2 == 0))
        brain.key = jax.random.PRNGKey(seed * 101 + k)
        opponents.append(brain)
    learner = PolicyBrain(load_policy_params(tactics_path)[0])
    learner.key = jax.random.PRNGKey(seed)
    bout = NativeBout(pair, motor, rng, opponents, scripted_fraction, 1000, fall_penalty,
                      high_guard_fraction, ArmSkill(load_policy_params(arm_path)[0]))

    def shadow():
        return StrikerArm(pair.pose, pair.action_scale, pair.dt)
    own = shadow()
    conn.send("ready")
    record = {k: [] for k in ("obs", "label", "reward", "terminal", "truncated")}
    for _ in range(steps):
        i = bout.learner
        tactics, learner.key = learner.act(bout.observe(), learner.key)
        tactics = np.asarray(tactics, np.float64)
        view = [own, bout.arms[1]] if i == 0 else [bout.arms[0], own]
        obs = striker_observe(pair, i, view, bout.t / bout.episode_steps)
        reward, terminal, truncated, _ = bout.step(tactics)
        own.follow(bout.executed_arms[i])
        label = np.concatenate([tactics[3:5], to_unit(own.arm_pose).ravel()])
        for name, value in (("obs", obs), ("label", label), ("reward", reward),
                            ("terminal", terminal), ("truncated", truncated)):
            record[name].append(value)
        if terminal or truncated:
            bout.reset()
            own = shadow()
    np.savez_compressed(output, **{k: np.asarray(v) for k, v in record.items()})
    conn.send("done")


def collect(args):
    _configure_worker_env()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "pair.mjb"
    context = mp.get_context("spawn")
    exporter = context.Process(target=export_native_model, args=(model_path, True))
    exporter.start()
    exporter.join()
    if exporter.exitcode:
        raise RuntimeError("could not export the native pair model")
    constants = json.loads(model_path.with_suffix(".json").read_text())
    pipes, processes = [], []
    for w in range(args.workers):
        parent, child = context.Pipe()
        process = context.Process(target=collect_worker, args=(
            child, args.seed * 1000 + w, args.steps // args.workers, str(args.motor), str(args.tactics),
            str(args.arm), [str(p) for p in args.pool], args.scripted_fraction, args.high_guard_fraction,
            args.fall_penalty, model_path, constants, root / f"worker-{w}.npz"), daemon=True)
        process.start()
        pipes.append(parent)
        processes.append(process)
        assert _receive(parent) == "ready"
    print(f"{args.workers} collectors running", flush=True)
    for pipe in pipes:
        assert _receive(pipe) == "done"
    for process in processes:
        process.join()
    print(json.dumps(dict(event="collected", steps=args.steps // args.workers * args.workers)), flush=True)


def discounted_returns(reward, terminal, truncated, gamma):
    returns = np.zeros_like(reward)
    running = 0.
    for t in reversed(range(len(reward))):
        if terminal[t] or truncated[t]:
            running = 0.
        running = reward[t] + gamma * running
        returns[t] = running
    return returns


def clone(args):
    import jax
    import jax.numpy as jp
    import optax
    from brax.training.acme import running_statistics
    from boxer.curriculum_train import save_policy, utc
    from boxer.striker import STRIKER_ACTION_SIZE, STRIKER_OBS_SIZE, striker_net

    parts = [np.load(path) for path in sorted(args.data.glob("worker-*.npz"))]
    obs = np.concatenate([p["obs"] for p in parts]).astype(np.float32)
    label = np.clip(np.concatenate([p["label"] for p in parts]), -.97, .97).astype(np.float32)
    returns = np.concatenate([discounted_returns(p["reward"], p["terminal"], p["truncated"], args.gamma)
                              for p in parts]).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(obs))
    held = order[:len(obs) // 20]
    train = order[len(obs) // 20:]
    net, _ = striker_net()
    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key)
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((STRIKER_OBS_SIZE,), jp.float32))
    norm = running_statistics.update(norm, jp.asarray(obs[train]))
    params = dict(policy=net.policy_network.init(k1), value=net.value_network.init(k2))
    # NormalTanh scale = softplus(x) + 0.001; aim at the exploration PPO starts with.
    target_scale = args.scale
    optimizer = optax.adam(args.learning_rate)
    state = optimizer.init(params)

    def losses(params, o, y, r):
        logits = net.policy_network.apply(norm, params["policy"], o)
        loc, scale = jp.split(logits, 2, axis=-1)
        imitation = ((jp.tanh(loc) - y) ** 2).mean(axis=0)
        spread = ((jax.nn.softplus(scale) + .001 - target_scale) ** 2).mean()
        value = ((net.value_network.apply(norm, params["value"], o) - r) ** 2).mean()
        return imitation, spread, value

    def total(params, o, y, r):
        imitation, spread, value = losses(params, o, y, r)
        return imitation.sum() + spread + value

    @jax.jit
    def update(params, state, o, y, r):
        grads = jax.grad(total)(params, o, y, r)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    evaluate = jax.jit(losses)
    log = []
    for epoch in range(args.epochs):
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled) - args.batch + 1, args.batch):
            index = shuffled[start:start + args.batch]
            params, state = update(params, state, obs[index], label[index], returns[index])
        imitation, spread, value = evaluate(params, obs[held], label[held], returns[held])
        row = dict(epoch=epoch + 1, held_out_mse=np.asarray(imitation).round(5).tolist(),
                   scale_error=float(spread), value_mse=float(value))
        log.append(row)
        print(json.dumps(row), flush=True)
    root = args.output
    save_policy(root / "checkpoint", (norm, params["policy"], params["value"]),
                dict(stage="striker-clone", stage_steps=0, samples=int(len(obs)), data=str(args.data),
                     obs_size=STRIKER_OBS_SIZE, action_size=STRIKER_ACTION_SIZE, updated_utc=utc()))
    (root / "clone.json").write_text(json.dumps(log, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--output", type=Path, default=Path("runs/striker-data"))
    c.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    c.add_argument("--tactics", type=Path, default=Path("checkpoints/tactics"))
    c.add_argument("--arm", type=Path, default=Path("checkpoints/arm-skill"))
    c.add_argument("--pool", type=Path, nargs="*",
                   default=[Path("checkpoints/tactics"), Path("checkpoints/tactics-highguard-v1")])
    c.add_argument("--scripted-fraction", type=float, default=.5)
    c.add_argument("--high-guard-fraction", type=float, default=.6)
    c.add_argument("--fall-penalty", type=float, default=30.)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--steps", type=int, default=640_000)
    c.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("clone")
    k.add_argument("--data", type=Path, default=Path("runs/striker-data"))
    k.add_argument("--output", type=Path, default=Path("runs/striker-clone"))
    k.add_argument("--epochs", type=int, default=30)
    k.add_argument("--batch", type=int, default=4096)
    k.add_argument("--learning-rate", type=float, default=1e-3)
    k.add_argument("--scale", type=float, default=.15, help="policy scale the clone starts PPO with")
    k.add_argument("--gamma", type=float, default=.97)
    k.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    (collect if args.command == "collect" else clone)(args)


if __name__ == "__main__":
    main()
