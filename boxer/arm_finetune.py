"""PPO for the arm skill (``boxer.arm_skill``) on native MuJoCo bouts.

Both corners box with a frozen tactics policy (the opponent is sometimes the
scripted brain) and the frozen leg motor. One corner, chosen per bout, has
its punches steered by the arm skill being trained. Every punch it throws
is one episode: from launch to the end of the retraction, a transition per
control step. Between its punches the bout runs on without transitions.

Reward per step of a punch: +1 for a clean punch (the native impact
scorecard), minus ``residual_cost`` x the squared residual (in units of
``RESIDUAL_SCALE``), and minus ``fall_penalty`` if the boxer falls, which
also ends the episode. A punch that runs to its end is a true terminal.

Workers, the frozen identity normalizer and the update follow
``boxer.native_finetune``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from boxer.native_finetune import (FoundationStub, _configure_worker_env, _receive, export_native_model,
                                   floor_std, gae)


class ArmBout:
    """A native bout, advanced one learner punch step at a time."""

    def __init__(self, pair, motor, rng, tactics, scripted_fraction, fall_penalty, residual_cost,
                 bout_steps=1000):
        self.pair, self.motor, self.rng, self.tactics = pair, motor, rng, tactics
        self.scripted_fraction, self.fall_penalty = scripted_fraction, fall_penalty
        self.residual_cost, self.bout_steps = residual_cost, bout_steps
        self.reset()

    def reset(self):
        from boxer.spar_match import PunchArm, RingBrain
        seed = int(self.rng.integers(1 << 30))
        self.pair.reset(seed=seed)
        self.arms = [PunchArm(self.pair.pose, self.pair.action_scale, self.pair.dt) for _ in range(2)]
        self.learner = int(self.rng.integers(2))
        opponent = (RingBrain(np.random.default_rng(seed)) if self.rng.random() < self.scripted_fraction
                    else self.tactics[int(self.rng.integers(len(self.tactics)))])
        own = self.tactics[int(self.rng.integers(len(self.tactics)))]
        self.brains = {self.learner: own, 1 - self.learner: opponent}
        self.t = 0

    def _control_step(self):
        """One 20 ms step; returns (scored, fallen, done)."""
        from boxer.curriculum_env import LEFT_ARM
        from boxer.spar_match import PolicyBrain, _body, motor_actions
        pair = self.pair
        commands, high = [None, None], [False, False]
        for k in range(2):
            brain = self.brains[k]
            if isinstance(brain, PolicyBrain):
                command, punch, high[k] = brain.decide_pair(pair, k, self.arms, self.t / self.bout_steps)
            else:
                bodies = [_body(pair, j) for j in range(2)]
                delta = bodies[1 - k]["pos"] - bodies[k]["pos"]
                command, punch = brain.decide(float(np.linalg.norm(delta)), bodies[k]["rot"].T @ delta,
                                              self.arms[k])
                high[k] = brain.high
            if punch is not None:
                self.arms[k].launch(*punch[:2], self.rng, *punch[2:])
            commands[k] = command
        commands = np.clip(commands, [-.35, -.25, -.5], [.55, .25, .5])
        action = motor_actions(self.motor, pair.observation(commands))
        for k in range(2):
            action[k, LEFT_ARM:] = self.arms[k].actions(bool(high[k]))
        outcome = pair.step(action)
        self.t += 1
        fallen = np.asarray(outcome["fallen"], bool)
        return outcome["scored"], fallen, bool(outcome["done"]) or self.t >= self.bout_steps

    def until_punch(self):
        """Runs the bout until the learner is mid-punch; returns its observation."""
        from boxer.arm_skill import arm_observe
        while self.arms[self.learner].ready:
            _, _, done = self._control_step()
            if done:
                self.reset()
        return arm_observe(self.pair, self.learner, self.arms[self.learner])

    def punch_step(self, residual):
        """Applies ``residual`` (in [-1, 1]) for one step of the learner's punch."""
        from boxer.arm_skill import RESIDUAL_SCALE, arm_observe
        i, arm = self.learner, self.arms[self.learner]
        arm.residual = RESIDUAL_SCALE * np.asarray(residual, np.float64)
        scored, fallen, done = self._control_step()
        reward = float(scored[i]) - self.residual_cost * float(np.sum(np.square(residual)))
        terminal = arm.ready or bool(fallen[i])
        reward -= self.fall_penalty * float(fallen[i])
        info = dict(hits=float(scored[i]), fell=float(fallen[i]), punches=float(arm.ready))
        if done:
            self.reset()
            terminal = True
        return reward, terminal, info


def worker(conn, *args):
    try:
        _worker(conn, *args)
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise


def _worker(conn, seed, motor_path, tactics_paths, scripted_fraction, fall_penalty, residual_cost,
            min_std, model_path, constants):
    _configure_worker_env()
    import jax
    import mujoco
    from boxer.curriculum_train import inference, load_policy_params
    from boxer.arm_skill import arm_observe
    from boxer.paired_foundation import PairedFoundation
    from boxer.spar_match import PolicyBrain

    foundation = FoundationStub(constants)
    motor = jax.jit(inference(foundation, load_policy_params(motor_path)[0]))
    pair = PairedFoundation(foundation, model=mujoco.MjModel.from_binary_path(str(model_path)))
    rng = np.random.default_rng(seed)
    tactics = []
    for k, path in enumerate(tactics_paths):
        brain = PolicyBrain(load_policy_params(path)[0], deterministic=False)
        brain.key = jax.random.PRNGKey(seed * 101 + k)
        tactics.append(brain)
    bout = ArmBout(pair, motor, rng, tactics, scripted_fraction, fall_penalty, residual_cost)
    sample_with = make_sampler(min_std)

    def sample_next(params, obs, key):
        key, sub = jax.random.split(key)
        return (*sample_with(params, obs, sub), key)
    sample = jax.jit(sample_next)
    key = jax.random.PRNGKey(seed)
    obs = bout.until_punch()
    conn.send("ready")
    while True:
        message = conn.recv()
        if message is None:
            break
        params, steps = message
        record = {k: [] for k in ("obs", "raw", "logp", "reward", "terminal", "next_obs")}
        stats = dict(hits=0., fell=0., punches=0.)
        for _ in range(steps):
            action, raw, logp, key = sample(params, obs, key)
            reward, terminal, info = bout.punch_step(np.asarray(action))
            # After a terminal step next_obs only feeds the (masked) bootstrap;
            # it is the first observation of the learner's next punch.
            next_obs = (bout.until_punch() if terminal
                        else arm_observe(bout.pair, bout.learner, bout.arms[bout.learner]))
            for name, value in (("obs", obs), ("raw", np.asarray(raw)), ("logp", float(logp)),
                                ("reward", reward), ("terminal", terminal), ("next_obs", next_obs)):
                record[name].append(value)
            for name in stats:
                stats[name] += info[name]
            obs = next_obs
        conn.send(({k: np.asarray(v) for k, v in record.items()}, stats))


def make_sampler(min_std):
    from boxer.arm_skill import arm_net
    net, dist = arm_net()

    def sample(params, obs, key):
        logits = floor_std(net.policy_network.apply(params[0], params[1], obs[None]), min_std)
        raw = dist.sample_no_postprocessing(logits, key)
        return dist.postprocess(raw)[0], raw[0], dist.log_prob(logits, raw)[0]
    return sample


def arm_match(arm, tactics, seeds, output, ring=False):
    """Champion tactics with ``arm`` (red first) vs the same tactics with scripted punches."""
    import os
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(Path.cwd()))
    command = [sys.executable, "-u", "-m", "boxer.spar_match", "--red-tactics", str(tactics),
               "--blue-tactics", str(tactics), "--red-arm", str(arm), "--swap", "--output", str(output),
               "--seeds", *map(str, seeds)] + (["--ring"] if ring else [])
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = json.loads(Path(output).read_text())
    results = summary.pop("results")
    first = summary["tactics"]["first"]
    corner = [0 if b["red"] == first else 1 for b in results]
    thrown = sum(b["punches_thrown"][c] for b, c in zip(results, corner))
    landed = sum(b["clean_punches"][c] for b, c in zip(results, corner))
    thrown_other = sum(b["punches_thrown"][1 - c] for b, c in zip(results, corner))
    landed_other = sum(b["clean_punches"][1 - c] for b, c in zip(results, corner))
    summary["landing_rate"] = dict(arm=landed / max(thrown, 1), scripted=landed_other / max(thrown_other, 1))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", type=Path, help="arm-skill checkpoint (default: zero residual)")
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--tactics", type=Path, nargs="+", default=[Path("checkpoints/tactics")])
    parser.add_argument("--output", type=Path, default=Path("runs/arm-skill"))
    parser.add_argument("--ring", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=750_000)
    parser.add_argument("--scripted-fraction", type=float, default=.3)
    parser.add_argument("--fall-penalty", type=float, default=5.)
    parser.add_argument("--residual-cost", type=float, default=.02)
    parser.add_argument("--min-std", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=.97)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--gate-seeds", type=int, nargs="*", default=list(range(8)),
                        help="native bouts (corners swapped) per checkpoint against scripted punches")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _configure_worker_env()

    import jax
    import jax.numpy as jp
    import optax
    from boxer.arm_skill import arm_net, initial_arm_params
    from boxer.checkpoint_io import atomic_json
    from boxer.curriculum_train import load_policy_params, save_policy, utc

    if args.init_from:
        norm, policy, value = load_policy_params(args.init_from)[0]
    else:
        norm, policy, value = initial_arm_params(jax.random.PRNGKey(args.seed))
    net, dist = arm_net()
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    trainable = dict(policy=policy, value=value)
    opt_state = optimizer.init(trainable)

    @jax.jit
    def values(trainable, obs):
        return net.value_network.apply(norm, trainable["value"], obs)

    def loss(trainable, batch, key):
        logits = floor_std(net.policy_network.apply(norm, trainable["policy"], batch["obs"]), args.min_std)
        logp = dist.log_prob(logits, batch["raw"])
        ratio = jp.exp(logp - batch["logp"])
        advantage = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-6)
        surrogate = jp.minimum(ratio * advantage, jp.clip(ratio, .8, 1.2) * advantage)
        value_error = net.value_network.apply(norm, trainable["value"], batch["obs"]) - batch["returns"]
        entropy = dist.entropy(logits, key)
        total = -surrogate.mean() + .25 * (value_error ** 2).mean() - args.entropy_cost * entropy.mean()
        return total, dict(kl=(batch["logp"] - logp).mean(), entropy=entropy.mean(),
                           clipped=(jp.abs(ratio - 1) > .2).mean())

    @jax.jit
    def update(trainable, opt_state, batch, key):
        (_, info), grads = jax.value_and_grad(loss, has_aux=True)(trainable, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state, info

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "pair.mjb"
    context = mp.get_context("spawn")
    exporter = context.Process(target=export_native_model, args=(model_path, args.ring))
    exporter.start()
    exporter.join()
    if exporter.exitcode:
        raise RuntimeError("could not export the native pair model")
    constants = json.loads(model_path.with_suffix(".json").read_text())
    pipes, processes = [], []
    for w in range(args.workers):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(
            child, args.seed * 1000 + w, args.motor, [str(p) for p in args.tactics], args.scripted_fraction,
            args.fall_penalty, args.residual_cost, args.min_std, model_path, constants), daemon=True)
        process.start()
        pipes.append(parent)
        processes.append(process)
        assert _receive(parent) == "ready"
    print(f"{args.workers} native workers ready", flush=True)
    atomic_json(root / "config.json", json.loads(json.dumps(vars(args), default=str)) | dict(started_utc=utc()))
    log = (root / "progress.jsonl").open("a")
    key = jax.random.PRNGKey(args.seed)
    done_steps, next_checkpoint, started = 0, 0, time.monotonic()

    def checkpoint():
        save_policy(root / "checkpoint", (norm, trainable["policy"], trainable["value"]),
                    dict(stage="arm-skill", stage_steps=done_steps,
                         init_from=str(args.init_from) if args.init_from else None, updated_utc=utc()))
    try:
        while done_steps < args.steps:
            if done_steps >= next_checkpoint:
                checkpoint()
                next_checkpoint += args.checkpoint_every
            params = jax.device_get((norm, trainable["policy"]))
            for pipe in pipes:
                pipe.send((params, args.unroll))
            results = [_receive(pipe) for pipe in pipes]
            batch, stats = {k: [] for k in ("obs", "raw", "logp", "advantage", "returns")}, []
            for record, info in results:
                v = np.asarray(values(trainable, record["obs"]))
                nv = np.asarray(values(trainable, record["next_obs"]))
                terminal = record["terminal"].astype(float)
                advantage, returns = gae(record["reward"], v, nv, terminal, np.zeros_like(terminal),
                                         args.gamma, args.lam)
                for name, value_ in (("obs", record["obs"]), ("raw", record["raw"]), ("logp", record["logp"]),
                                     ("advantage", advantage), ("returns", returns)):
                    batch[name].append(value_)
                stats.append(info)
            batch = {k: jp.asarray(np.concatenate(v), jp.float32) for k, v in batch.items()}
            size = batch["obs"].shape[0]
            for _ in range(args.epochs):
                key, perm_key = jax.random.split(key)
                order = np.asarray(jax.random.permutation(perm_key, size))
                for chunk in np.array_split(order, args.minibatches):
                    key, sub = jax.random.split(key)
                    trainable, opt_state, info = update(trainable, opt_state,
                                                        {k: v[chunk] for k, v in batch.items()}, sub)
            done_steps += size
            punches = sum(s["punches"] for s in stats)
            record = dict(step=done_steps, elapsed=time.monotonic() - started,
                          steps_per_second=done_steps / (time.monotonic() - started),
                          reward_per_step=float(np.mean([r["reward"].mean() for r, _ in results])),
                          hits_per_punch=sum(s["hits"] for s in stats) / max(punches, 1.),
                          punches=punches, falls=sum(s["fell"] for s in stats),
                          **{k: float(v) for k, v in info.items()})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        checkpoint()
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in processes:
            process.join(timeout=10)
    if args.gate_seeds:
        from boxer.tactics_gate import candidates, stage
        for manifest in candidates(root / "checkpoint", 4):
            step = json.loads(manifest.read_text())["stage_steps"]
            directory = stage(manifest, root / "gate" / f"step-{step}")
            summary = arm_match(directory, args.tactics[0], args.gate_seeds,
                                root / "gate" / f"step-{step}.json", args.ring)
            print(json.dumps(dict(event="arm_gate", step=step, **summary["tactics"],
                                  landing_rate=summary["landing_rate"], knockdowns=summary["knockdowns"])),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
