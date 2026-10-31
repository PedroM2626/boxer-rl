"""PPO fine-tuning of boxing tactics on native MuJoCo rollouts.

Batched MJX/Warp self-play beyond the champion learns contact details that
native MuJoCo does not share (docs/tactics.md). Here every training sample
comes from the physics used for evaluation. Worker processes each run a
``PairedFoundation`` bout with the frozen sparring motor, the native impact
scorecard and the shared ``observe``/``decode``/``PunchArm`` code. Each bout
picks the learner's corner at random, and its opponent from the scripted
brain and a pool of frozen tactics policies. A share of the scripted
opponents can hold the high guard throughout, which no straight gets through
(docs/moves.md, scoring audit): only hooks score on them. A central learner
runs PPO on the gathered transitions.

The observation normalizer stays frozen at the starting checkpoint's
statistics, so behaviour log-probabilities stay exact across the update.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from boxer import moves

REWARD = dict(hit=2.0, taken=1.5, opponent_fell=5.0, in_range=.3, punch=.02)


def _configure_worker_env():
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Native rollouts need no GPU; a CUDA context per worker exhausts it.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=1")


class NativeBout:
    """One native two-robot bout, stepped one learner decision at a time."""

    def __init__(self, pair, motor, rng, opponents, scripted_fraction, episode_steps, fall_penalty,
                 high_guard_fraction=0.):
        from boxer.spar_match import PunchArm
        self.pair, self.motor, self.rng = pair, motor, rng
        self.opponents, self.scripted_fraction = opponents, scripted_fraction
        self.high_guard_fraction = high_guard_fraction
        self.episode_steps, self.fall_penalty = episode_steps, fall_penalty
        self.PunchArm = PunchArm
        self.reset()

    def reset(self):
        from boxer.spar_match import RingBrain
        seed = int(self.rng.integers(1 << 30))
        self.pair.reset(seed=seed)
        self.arms = [self.PunchArm(self.pair.pose, self.pair.action_scale, self.pair.dt) for _ in range(2)]
        self.learner = int(self.rng.integers(2))
        if not self.opponents or self.rng.random() < self.scripted_fraction:
            self.opponent = RingBrain(np.random.default_rng(seed),
                                      high=bool(self.rng.random() < self.high_guard_fraction))
        else:
            self.opponent = self.opponents[int(self.rng.integers(len(self.opponents)))]
        self.t = 0

    def observe(self):
        from boxer.spar_match import _body
        from boxer.tactics_env import observe
        bodies = [_body(self.pair, k) for k in range(2)]
        i = self.learner
        return observe(np, bodies[i], bodies[1 - i], self.arms[i].features(),
                       self.arms[1 - i].features(), self.t / self.episode_steps).astype(np.float32)

    def step(self, tactics):
        from boxer.curriculum_env import LEFT_ARM
        from boxer.spar_match import PolicyBrain, _body, motor_actions
        from boxer.tactics_env import REACH, decode
        pair, i = self.pair, self.learner
        bodies = [_body(pair, k) for k in range(2)]
        commands, launched, hooked, high = [None, None], False, False, [False, False]
        for k in range(2):
            delta = bodies[1 - k]["pos"] - bodies[k]["pos"]
            distance = float(np.linalg.norm(delta))
            local = bodies[k]["rot"].T @ delta
            if k == i:
                command, want, hand, aim, move, high[k] = decode(
                    np.asarray(tactics, np.float64), distance, local)
                punch = (int(hand), float(aim), int(move)) if want and self.arms[k].ready else None
                launched = punch is not None
                hooked = launched and punch[2] == moves.HOOK
            elif isinstance(self.opponent, PolicyBrain):
                command, punch, high[k] = self.opponent.decide_pair(
                    pair, k, self.arms, self.t / self.episode_steps)
            else:
                command, punch = self.opponent.decide(distance, local, self.arms[k])
                high[k] = self.opponent.high
            if punch is not None:
                self.arms[k].launch(*punch[:2], self.rng, *punch[2:])
            commands[k] = command
        commands = np.clip(commands, [-.35, -.25, -.5], [.55, .25, .5])
        action = motor_actions(self.motor, pair.observation(commands))
        for k in range(2):
            action[k, LEFT_ARM:] = self.arms[k].actions(bool(high[k]))
        outcome = pair.step(action)
        fallen = np.asarray(outcome["fallen"], bool)
        scored = outcome["scored"]
        distance = float(np.linalg.norm(_body(pair, 1)["pos"] - _body(pair, 0)["pos"]))
        reward = (REWARD["hit"] * scored[i] - REWARD["taken"] * scored[1 - i]
                  - self.fall_penalty * fallen[i] + REWARD["opponent_fell"] * (fallen[1 - i] and not fallen[i])
                  + pair.dt * REWARD["in_range"] * np.exp(-((distance - REACH) / .15) ** 2)
                  - REWARD["punch"] * launched)
        self.t += 1
        terminal = bool(outcome["done"])
        truncated = not terminal and self.t >= self.episode_steps
        return float(reward), terminal, truncated, dict(hits=float(scored[i]), taken=float(scored[1 - i]),
                                                        fell=float(fallen[i]), punches=float(launched),
                                                        hooks=float(hooked))


def worker(conn, *args):
    """Process entry point; any failure is sent back to the learner as text."""
    try:
        _worker(conn, *args)
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise


def _receive(pipe):
    message = pipe.recv()
    if isinstance(message, tuple) and len(message) == 2 and message[0] == "error":
        raise RuntimeError("native worker failed:\n" + message[1])
    return message


class FoundationStub:
    """The few ``UnassistedG1`` facts a native worker needs, without building MJX.

    Building the Playground environment costs each worker over 1 GB; the
    learner builds it once and passes these constants plus a saved model.
    """

    def __init__(self, constants):
        from types import SimpleNamespace
        self._default_pose = np.asarray(constants["pose"])
        self._init_q = np.asarray(constants["init_q"])
        self._config = SimpleNamespace(action_scale=constants["action_scale"])
        names = constants["actuators"]
        self.mj_model = SimpleNamespace(actuator=lambda i: SimpleNamespace(name=names[i]))
        self.observation_size = {k: tuple(v) for k, v in constants["observation_size"].items()}
        self.action_size = constants["action_size"]

    @staticmethod
    def constants(foundation):
        return dict(pose=np.asarray(foundation._default_pose).tolist(),
                    init_q=np.asarray(foundation._init_q).tolist(),
                    action_scale=float(foundation._config.action_scale),
                    actuators=[foundation.mj_model.actuator(i).name for i in range(foundation.mj_model.nu)],
                    observation_size={k: list(v) for k, v in foundation.observation_size.items()},
                    action_size=int(foundation.action_size))


def export_native_model(path):
    """Writes the native pair model and ``FoundationStub`` constants next to it."""
    _configure_worker_env()
    import mujoco
    from boxer.curriculum_env import UnassistedG1
    from boxer.paired_foundation import build_pair
    foundation = UnassistedG1("spar")
    mujoco.mj_saveModel(build_pair(foundation), str(path), None)
    Path(path).with_suffix(".json").write_text(json.dumps(FoundationStub.constants(foundation)))


def _worker(conn, seed, motor_path, opponent_paths, scripted_fraction, episode_steps, fall_penalty,
            min_std, model_path, constants, high_guard_fraction=0.):
    _configure_worker_env()
    import jax
    import mujoco
    from boxer.curriculum_train import inference, load_policy_params
    from boxer.paired_foundation import PairedFoundation
    from boxer.spar_match import PolicyBrain

    foundation = FoundationStub(constants)
    motor = jax.jit(inference(foundation, load_policy_params(motor_path)[0]))
    pair = PairedFoundation(foundation, model=mujoco.MjModel.from_binary_path(str(model_path)))
    rng = np.random.default_rng(seed)
    opponents = []
    for k, path in enumerate(opponent_paths):
        brain = PolicyBrain(load_policy_params(path)[0], deterministic=bool(k % 2 == 0))
        brain.key = jax.random.PRNGKey(seed * 101 + k)
        opponents.append(brain)
    bout = NativeBout(pair, motor, rng, opponents, scripted_fraction, episode_steps, fall_penalty,
                      high_guard_fraction)
    sample_with = make_sampler(min_std)

    def sample_next(params, obs, key):
        key, sub = jax.random.split(key)
        return (*sample_with(params, obs, sub), key)
    sample = jax.jit(sample_next)
    key = jax.random.PRNGKey(seed)
    obs = bout.observe()
    conn.send("ready")
    while True:
        message = conn.recv()
        if message is None:
            break
        params, steps = message
        record = {k: [] for k in ("obs", "raw", "logp", "reward", "terminal", "truncated", "next_obs")}
        stats = dict(hits=0., taken=0., fell=0., punches=0., hooks=0., bouts=0)
        for _ in range(steps):
            action, raw, logp, key = sample(params, obs, key)
            reward, terminal, truncated, info = bout.step(np.asarray(action))
            next_obs = bout.observe()
            for name, value in (("obs", obs), ("raw", np.asarray(raw)), ("logp", float(logp)),
                                ("reward", reward), ("terminal", terminal), ("truncated", truncated),
                                ("next_obs", next_obs)):
                record[name].append(value)
            for name in ("hits", "taken", "fell", "punches", "hooks"):
                stats[name] += info[name]
            if terminal or truncated:
                stats["bouts"] += 1
                bout.reset()
                next_obs = bout.observe()
            obs = next_obs
        conn.send(({k: np.asarray(v) for k, v in record.items()}, stats))


def make_sampler(min_std):
    """Samples like brax's NormalTanh policy, with the scale floored at ``min_std``."""
    import jax
    import jax.numpy as jp
    net, dist = tactics_net()

    def sample(params, obs, key):
        logits = net.policy_network.apply(params[0], params[1], obs[None])
        logits = floor_std(logits, min_std)
        raw = dist.sample_no_postprocessing(logits, key)
        return dist.postprocess(raw)[0], raw[0], dist.log_prob(logits, raw)[0]
    return sample


def floor_std(logits, min_std):
    """Raises the NormalTanh scale (softplus(x) + 0.001) to at least ``min_std``."""
    import jax.numpy as jp
    if not min_std:
        return logits
    loc, scale = jp.split(logits, 2, axis=-1)
    target = jp.log(jp.expm1(jp.maximum(min_std - .001, 1e-6)))     # softplus^-1
    return jp.concatenate([loc, jp.maximum(scale, target)], axis=-1)


def tactics_net():
    from brax.training.acme import running_statistics
    from boxer.tactics_env import ACTION_SIZE, OBS_SIZE, tactics_network_factory
    net = tactics_network_factory()(OBS_SIZE, ACTION_SIZE,
                                    preprocess_observations_fn=running_statistics.normalize)
    return net, net.parametric_action_distribution


def gae(rewards, values, next_values, terminal, truncated, gamma, lam):
    """Generalized advantage estimates for one worker's time-ordered steps."""
    advantages = np.zeros_like(rewards)
    running = 0.
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminal[t]) - values[t]
        cut = terminal[t] or truncated[t]
        running = delta + gamma * lam * (0. if cut else running)
        advantages[t] = running
    return advantages, advantages + values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/tactics"))
    parser.add_argument("--motor", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--pool", type=Path, nargs="*", default=[Path("checkpoints/tactics")])
    parser.add_argument("--output", type=Path, default=Path("runs/tactics-native"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=256, help="steps per worker per iteration")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--scripted-fraction", type=float, default=.3)
    parser.add_argument("--high-guard-fraction", type=float, default=0.,
                        help="share of scripted opponents that hold the high guard")
    parser.add_argument("--fall-penalty", type=float, default=30.)
    parser.add_argument("--min-std", type=float, default=.15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=.97)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--native-gate", type=int, default=4,
                        help="gate this many final checkpoints against --init-from (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _configure_worker_env()

    import jax
    import jax.numpy as jp
    import optax
    from boxer.checkpoint_io import atomic_json
    from boxer.curriculum_train import load_policy_params, save_policy, utc
    from boxer.tactics_gate import gate_generation

    from boxer.tactics_env import ACTION_SIZE, OBS_SIZE, policy_sizes, widen_legacy
    initial = load_policy_params(args.init_from)[0]
    if policy_sizes(initial) != (OBS_SIZE, ACTION_SIZE):
        # A pre-hook policy: same behaviour, new inputs/outputs start silent.
        initial = widen_legacy(initial)
    norm, policy, value = initial
    net, dist = tactics_net()
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
    # Build the Playground environment in a throwaway process: it keeps over
    # 2 GB resident, which would otherwise crowd out rollout workers.
    exporter = context.Process(target=export_native_model, args=(model_path,))
    exporter.start()
    exporter.join()
    if exporter.exitcode:
        raise RuntimeError("could not export the native pair model")
    constants = json.loads(model_path.with_suffix(".json").read_text())

    pipes, processes = [], []
    for w in range(args.workers):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(
            child, args.seed * 1000 + w, args.motor, [str(p) for p in args.pool],
            args.scripted_fraction, 1000, args.fall_penalty, args.min_std, model_path, constants,
            args.high_guard_fraction),
            daemon=True)
        process.start()
        pipes.append(parent)
        processes.append(process)
        # Start one at a time: initialization and XLA compilation peak at
        # 1-1.7 GB per worker, while steady rollouts need far less.
        assert _receive(parent) == "ready"
    print(f"{args.workers} native workers ready", flush=True)

    atomic_json(root / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                | dict(pool=[str(p) for p in args.pool], reward=REWARD, started_utc=utc()))
    log = (root / "progress.jsonl").open("a")
    key = jax.random.PRNGKey(args.seed)
    done_steps, next_checkpoint, started = 0, 0, time.monotonic()
    try:
        while done_steps < args.steps:
            if done_steps >= next_checkpoint:
                save_policy(root / "checkpoint", (norm, trainable["policy"], trainable["value"]),
                            dict(stage="tactics-native", stage_steps=done_steps, init_from=str(args.init_from),
                                 updated_utc=utc()))
                next_checkpoint += args.checkpoint_every
            params = jax.device_get((norm, trainable["policy"]))
            for pipe in pipes:
                pipe.send((params, args.unroll))
            results = [_receive(pipe) for pipe in pipes]
            batch, stats = {k: [] for k in ("obs", "raw", "logp", "advantage", "returns")}, []
            for record, info in results:
                v = np.asarray(values(trainable, record["obs"]))
                nv = np.asarray(values(trainable, record["next_obs"]))
                advantage, returns = gae(record["reward"], v, nv, record["terminal"].astype(float),
                                         record["truncated"], args.gamma, args.lam)
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
            bouts = sum(s["bouts"] for s in stats)
            record = dict(step=done_steps, elapsed=time.monotonic() - started,
                          steps_per_second=done_steps / (time.monotonic() - started),
                          reward_per_step=float(np.mean([r["reward"].mean() for r, _ in results])),
                          hits_per_1000=1000 * sum(s["hits"] for s in stats) / size,
                          taken_per_1000=1000 * sum(s["taken"] for s in stats) / size,
                          falls=sum(s["fell"] for s in stats), bouts_finished=bouts,
                          hook_share=sum(s["hooks"] for s in stats) / max(sum(s["punches"] for s in stats), 1.),
                          **{k: float(v) for k, v in info.items()})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        save_policy(root / "checkpoint", (norm, trainable["policy"], trainable["value"]),
                    dict(stage="tactics-native", stage_steps=done_steps, init_from=str(args.init_from),
                         updated_utc=utc()))
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in processes:
            process.join(timeout=10)
    if args.native_gate:
        gate_generation(root, args.init_from, args.native_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
