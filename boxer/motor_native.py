"""PPO fine-tuning of the leg motor in native two-robot bouts.

The motor (``checkpoints/spar``) learned on the GPU, one robot at a time, with
generated or replayed arm motions and random shoves standing in for an
opponent. Its remaining falls come from what that cannot reproduce: two
bodies leaning, pushing and tangling at close range. Here both robots of a
native ``PairedFoundation`` ring bout run the learner motor, and both
robots' transitions train it. Each corner's footwork commands and arms come
from a brain drawn per bout from a pool (strikers, the champion tactics with
the arm skill, the scripted ring brain), exactly as in ``spar_match``.

Only the 15 leg and waist outputs are learned: the arms are always replaced,
so their outputs carry no credit and stay out of the log-probabilities. The
reward keeps the solo task's main terms (velocity tracking, upright torso,
smooth actions; ``REWARD``) and charges a fall. A fall ends the bout, as in
the foundation: terminal for the robot that fell, a cut for the other.

The motor's own critic sees privileged simulator state that native bouts do
not provide, so a new critic is fitted on the policy's observation plus
where the opponent is and which brains fight (``value_features``); the
policy waits until the critic has warmed up. The policy's observation
normalizer stays frozen, so behaviour log-probabilities stay exact. The
checkpoint keeps the starting critic, so ``curriculum_eval`` and every bout
load it like any motor.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from boxer.native_finetune import (FoundationStub, _configure_worker_env, _receive, export_native_model,
                                   gae)

LEGS = 15                       # leg and waist actuators; the arms follow from LEFT_ARM = 15
REWARD = dict(linear=1.0, yaw=.75, orientation=-2., tilt_rate=-.15, action_rate=-.01, fall=10.)
SIGMA = .25                     # velocity tracking width, as in the solo task
BRAINS = ("striker", "low-motion striker", "champion", "scripted")
VALUE_SIZE = 103 + 3 + 2 * len(BRAINS)


def motor_reward(obs, previous, action, command, fell, dt):
    """One robot's step reward from its next observation (``PairedFoundation.observation``)."""
    velocity, gyro, gravity = obs[0:3], obs[3:6], obs[6:9]
    linear = np.exp(-np.sum((velocity[:2] - command[:2]) ** 2) / SIGMA)
    yaw = np.exp(-(gyro[2] - command[2]) ** 2 / SIGMA)
    shaped = (REWARD["linear"] * linear + REWARD["yaw"] * yaw
              + REWARD["orientation"] * np.sum(gravity[:2] ** 2)
              + REWARD["tilt_rate"] * np.sum(gyro[:2] ** 2)
              + REWARD["action_rate"] * np.sum((action[:LEGS] - previous[:LEGS]) ** 2))
    return float(dt * shaped - REWARD["fall"] * fell)


def value_features(pair, obs, kinds):
    """Critic input per corner: the motor observation, the opponent's place, both brains."""
    rows = []
    for i, m in enumerate(pair.maps):
        roots = [pair.data.qpos[n['root']:n['root'] + 2] for n in pair.maps]
        rotation = pair.data.site(m['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)
        local = rotation[:2, :2].T @ (roots[1 - i] - roots[i])
        brains = np.zeros(2 * len(BRAINS))
        brains[kinds[i]] = brains[len(BRAINS) + kinds[1 - i]] = 1.
        rows.append(np.concatenate([obs[i], local, [np.linalg.norm(local)], brains]))
    return np.asarray(rows, np.float32)


class MotorBout:
    """One native ring bout whose two robots both run the learner motor."""

    def __init__(self, pair, rng, pool, arm_skill, episode_steps, weights=None):
        """``pool``: brain kind (index into ``BRAINS``) -> loaded brain, or None for scripted;
        ``weights``: kind -> relative chance that a corner draws it (default: equal)."""
        self.pair, self.rng, self.pool = pair, rng, pool
        kinds = sorted(pool)
        chances = np.array([(weights or {}).get(k, 1.) for k in kinds], float)
        self.kinds_drawn, self.chances = kinds, chances / chances.sum()
        self.arm_skill, self.episode_steps = arm_skill, episode_steps
        pair.self_trip_falls = False        # knockdown rules: only real falls
        self.reset()

    def reset(self):
        from boxer.spar_match import RingBrain, _new_arm
        seed = int(self.rng.integers(1 << 30))
        self.pair.reset(seed=seed)
        self.kinds = [int(k) for k in self.rng.choice(self.kinds_drawn, 2, p=self.chances)]
        self.brains = [self.pool[k] or RingBrain(np.random.default_rng([seed, i]))
                       for i, k in enumerate(self.kinds)]
        self.arms = [_new_arm(self.pair, brain) for brain in self.brains]
        self.t = 0
        self.decide()

    def decide(self):
        """Commands and arm moves for this step, as ``spar_match`` makes them."""
        pair = self.pair
        roots = [pair.data.qpos[m['root']:m['root'] + 2] for m in pair.maps]
        commands, self.high = [], [False, False]
        for i, m in enumerate(pair.maps):
            delta = roots[1 - i] - roots[i]
            rotation = pair.data.site(m['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)
            local = rotation[:2, :2].T @ delta
            brain = self.brains[i]
            if hasattr(brain, 'decide_pair'):
                command, punch, self.high[i] = brain.decide_pair(pair, i, self.arms, self.t / self.episode_steps)
            else:
                command, punch = brain.decide(float(np.linalg.norm(delta)), local, self.arms[i])
                self.high[i] = getattr(brain, 'high', False)
            if punch is not None:
                self.arms[i].launch(*punch[:2], self.rng, *punch[2:])
            commands.append(command)
        self.commands = np.clip(commands, [-.35, -.25, -.5], [.55, .25, .5])
        self.obs = pair.observation(self.commands)
        self.values = value_features(pair, self.obs, self.kinds)

    def step(self, motor_action):
        """Applies both robots' 29 motor outputs (arms replaced); returns per-robot outcomes."""
        from boxer.curriculum_env import LEFT_ARM
        pair = self.pair
        action = np.array(motor_action, np.float64)
        for i in range(2):
            if self.arm_skill is not None:
                self.arm_skill.steer(pair, i, self.arms[i])
            action[i, LEFT_ARM:] = self.arms[i].actions(self.high[i])
        previous = pair.last_action.copy()
        commands = self.commands
        outcome = pair.step(action)
        fallen = np.asarray(outcome["fallen"], bool) & outcome["done"]
        after = pair.observation(commands)
        rewards = np.array([motor_reward(after[i], previous[i], action[i], commands[i], fallen[i], pair.dt)
                            for i in range(2)])
        self.t += 1
        ended = bool(outcome["done"]) or self.t >= self.episode_steps
        terminal = fallen.copy()
        truncated = ended & ~terminal
        if not ended:
            self.decide()
            next_values = self.values
        else:
            next_values = value_features(pair, after, self.kinds)
        return rewards, terminal, truncated, next_values, dict(fell=fallen.astype(float),
                                                               scored=np.asarray(outcome["scored"]))


def leg_log_prob(loc, scale, raw):
    """Gaussian log-density of the pre-tanh leg and waist actions (the tanh Jacobian
    depends on ``raw`` only and cancels in PPO ratios)."""
    import jax.numpy as jp
    z = (raw[..., :LEGS] - loc[..., :LEGS]) / scale[..., :LEGS]
    return jp.sum(-.5 * z ** 2 - jp.log(scale[..., :LEGS]) - .5 * jp.log(2 * jp.pi), axis=-1)


def split_logits(logits):
    """(loc, scale) of brax's NormalTanh distribution (min_std 0.001)."""
    import jax
    import jax.numpy as jp
    loc, scale = jp.split(logits, 2, axis=-1)
    return loc, jax.nn.softplus(scale) + .001


def motor_policy(observation_size, action_size):
    from brax.training.acme import running_statistics
    from boxer.curriculum_train import network_factory
    return network_factory()(observation_size, action_size,
                             preprocess_observations_fn=running_statistics.normalize).policy_network


def worker(conn, *args):
    try:
        _worker(conn, *args)
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise


def _worker(conn, seed, model_path, constants, pool_paths, arm_path, episode_steps, weights=None):
    _configure_worker_env()
    import jax
    import jax.numpy as jp
    import mujoco
    from boxer.arm_skill import ArmSkill
    from boxer.curriculum_train import load_policy_params
    from boxer.paired_foundation import PairedFoundation
    from boxer.spar_match import load_brain

    foundation = FoundationStub(constants)
    pair = PairedFoundation(foundation, model=mujoco.MjModel.from_binary_path(str(model_path)))
    rng = np.random.default_rng(seed)
    pool = {k: (load_brain(path) if path else None) for k, path in pool_paths.items()}
    for k, brain in pool.items():
        if brain is not None:
            brain.key = jax.random.PRNGKey(seed * 101 + k)
    bout = MotorBout(pair, rng, pool, ArmSkill(load_policy_params(arm_path)[0]), episode_steps, weights)
    policy = motor_policy(foundation.observation_size, foundation.action_size)
    privileged = jp.zeros((2,) + tuple(foundation.observation_size["privileged_state"]))

    @jax.jit
    def sample(params, obs, key):
        key, sub = jax.random.split(key)
        loc, scale = split_logits(policy.apply(params[0], params[1], {"state": obs, "privileged_state": privileged}))
        raw = loc + scale * jax.random.normal(sub, loc.shape)
        return jp.tanh(raw), raw, leg_log_prob(loc, scale, raw), key

    key = jax.random.PRNGKey(seed)
    conn.send("ready")
    while True:
        message = conn.recv()
        if message is None:
            break
        params, steps = message
        streams = [{k: [] for k in ("obs", "value_obs", "raw", "logp", "reward", "terminal", "truncated",
                                    "next_value_obs")} for _ in range(2)]
        stats = dict(falls=0., bouts=0, scored=0., by_brain=np.zeros(len(BRAINS)),
                     falls_by_brain=np.zeros(len(BRAINS)))
        for _ in range(steps):
            obs, value_obs = bout.obs, bout.values
            action, raw, logp, key = sample(params, jp.asarray(obs), key)
            kinds = list(bout.kinds)
            try:
                rewards, terminal, truncated, next_values, info = bout.step(np.asarray(action))
            except FloatingPointError:      # invalid native state: drop the step, start a new bout
                bout.reset()
                continue
            for i in range(2):
                for name, value in (("obs", obs[i]), ("value_obs", value_obs[i]), ("raw", np.asarray(raw[i])),
                                    ("logp", float(logp[i])), ("reward", rewards[i]),
                                    ("terminal", bool(terminal[i])), ("truncated", bool(truncated[i])),
                                    ("next_value_obs", next_values[i])):
                    streams[i][name].append(value)
                stats["by_brain"][kinds[i]] += 1
                stats["falls_by_brain"][kinds[i]] += info["fell"][i]
            stats["falls"] += info["fell"].sum()
            stats["scored"] += info["scored"].sum()
            if terminal.any() or truncated.any():
                stats["bouts"] += 1
                bout.reset()
        conn.send(([{k: np.asarray(v) for k, v in s.items()} for s in streams], stats))


def value_net():
    import flax.linen as nn
    import jax.numpy as jp

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, x):
            for width in (256, 256):
                x = nn.swish(nn.Dense(width)(x))
            return jp.squeeze(nn.Dense(1)(x), -1)
    return Critic()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--striker", type=Path, default=Path("checkpoints/striker-v1"))
    parser.add_argument("--low-motion-striker", type=Path, default=Path("checkpoints/striker"),
                        help="a striker trained with an arm-motion cost")
    parser.add_argument("--tactics", type=Path, default=Path("checkpoints/tactics"))
    parser.add_argument("--arm", type=Path, default=Path("checkpoints/arm-skill"))
    parser.add_argument("--weights", type=float, nargs=4, default=[1., 1., 1., 1.],
                        help="relative chances of each brain per corner, in BRAINS order")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=256, help="control steps per worker per update")
    parser.add_argument("--episode-steps", type=int, default=3000, help="bout length cap (60 s)")
    parser.add_argument("--steps", type=int, default=3_000_000, help="robot transitions")
    parser.add_argument("--warmup", type=int, default=8, help="updates that fit only the critic")
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=.01, help="stop an update's epochs beyond this")
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _configure_worker_env()

    import jax
    import jax.numpy as jp
    import optax
    from brax.training.acme import running_statistics
    from boxer.checkpoint_io import atomic_json
    from boxer.curriculum_train import load_policy_params, save_policy, utc

    (norm, policy_params, old_value), manifest = load_policy_params(args.init_from)
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
    stub = FoundationStub(constants)
    policy = motor_policy(stub.observation_size, stub.action_size)
    privileged = jp.zeros((1,) + tuple(stub.observation_size["privileged_state"]))
    critic = value_net()
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    value_params = critic.init(init_key, jp.zeros((1, VALUE_SIZE)))
    value_norm = running_statistics.init_state(jax.ShapeDtypeStruct((VALUE_SIZE,), jp.float32))

    paths = {0: str(args.striker), 1: str(args.low_motion_striker), 2: str(args.tactics), 3: None}
    pipes, processes = [], []
    for w in range(args.workers):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(
            child, args.seed * 1000 + w, model_path, constants, paths, str(args.arm), args.episode_steps,
            dict(enumerate(args.weights))),
            daemon=True)
        process.start()
        pipes.append(parent)
        processes.append(process)
        assert _receive(parent) == "ready"
    print(f"{args.workers} native motor workers ready", flush=True)
    atomic_json(root / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                | dict(reward=REWARD, brains={BRAINS[k]: p for k, p in paths.items()}, started_utc=utc()))

    policy_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    critic_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.critic_learning_rate))
    policy_state, critic_state = policy_opt.init(policy_params), critic_opt.init(value_params)

    def values(params, vnorm, x):
        return critic.apply(params, running_statistics.normalize(x, vnorm))

    values_jit = jax.jit(values)

    def policy_loss(params, batch):
        obs = {"state": batch["obs"], "privileged_state": jp.broadcast_to(privileged, (batch["obs"].shape[0],
                                                                                        privileged.shape[1]))}
        loc, scale = split_logits(policy.apply(norm, params, obs))
        logp = leg_log_prob(loc, scale, batch["raw"])
        ratio = jp.exp(logp - batch["logp"])
        advantage = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-6)
        surrogate = jp.minimum(ratio * advantage, jp.clip(ratio, .8, 1.2) * advantage)
        return -surrogate.mean(), dict(kl=(batch["logp"] - logp).mean(), clipped=(jp.abs(ratio - 1) > .2).mean(),
                                       leg_std=scale[:, :LEGS].mean())

    @jax.jit
    def policy_update(params, state, batch):
        (_, info), grads = jax.value_and_grad(policy_loss, has_aux=True)(params, batch)
        updates, state = policy_opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, info

    @jax.jit
    def critic_update(params, state, vnorm, batch):
        loss, grads = jax.value_and_grad(lambda p: ((values(p, vnorm, batch["value_obs"]) - batch["returns"]) ** 2
                                                    ).mean())(params)
        updates, state = critic_opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    def save(steps):
        metadata = {k: v for k, v in manifest.items() if k not in ("sha256", "file", "optimizer_saved")}
        save_policy(root / "checkpoint", (norm, policy_params, old_value),
                    metadata | dict(stage="spar", stage_steps=int(manifest["stage_steps"]) + steps,
                                    native_steps=steps, native_from=str(args.init_from), updated_utc=utc()))

    log = (root / "progress.jsonl").open("a")
    done, update_count, next_checkpoint, started = 0, 0, args.checkpoint_every, time.monotonic()
    try:
        while done < args.steps:
            params = jax.device_get((norm, policy_params))
            for pipe in pipes:
                pipe.send((params, args.unroll))
            results = [_receive(pipe) for pipe in pipes]
            streams = [s for r, _ in results for s in r]
            if update_count == 0:
                value_norm = running_statistics.update(
                    value_norm, jp.asarray(np.concatenate([s["value_obs"] for s in streams]), jp.float32))
            batch = {k: [] for k in ("obs", "value_obs", "raw", "logp", "advantage", "returns")}
            for s in streams:
                v = np.asarray(values_jit(value_params, value_norm, s["value_obs"]))
                nv = np.asarray(values_jit(value_params, value_norm, s["next_value_obs"]))
                advantage, returns = gae(s["reward"], v, nv, s["terminal"].astype(float), s["truncated"],
                                         args.gamma, args.lam)
                for name, value in (("obs", s["obs"]), ("value_obs", s["value_obs"]), ("raw", s["raw"]),
                                    ("logp", s["logp"]), ("advantage", advantage), ("returns", returns)):
                    batch[name].append(value)
            batch = {k: jp.asarray(np.concatenate(v), jp.float32) for k, v in batch.items()}
            size = batch["obs"].shape[0]
            learn_policy = update_count >= args.warmup
            info, critic_loss, epochs_run = {}, 0., 0
            for _ in range(args.epochs):
                key, perm_key = jax.random.split(key)
                order = np.asarray(jax.random.permutation(perm_key, size))
                kls = []
                for chunk in np.array_split(order, args.minibatches):
                    mini = {k: v[chunk] for k, v in batch.items()}
                    value_params, critic_state, critic_loss = critic_update(value_params, critic_state,
                                                                            value_norm, mini)
                    if learn_policy:
                        policy_params, policy_state, info = policy_update(policy_params, policy_state, mini)
                        kls.append(float(info["kl"]))
                epochs_run += 1
                if learn_policy and np.mean(kls) > args.target_kl:
                    break
            update_count += 1
            if learn_policy:
                done += size
            stats = [s for _, s in results]
            by_brain = sum(s["by_brain"] for s in stats)
            falls_by_brain = sum(s["falls_by_brain"] for s in stats)
            record = dict(update=update_count, step=done, elapsed=time.monotonic() - started,
                          learning_policy=learn_policy,
                          reward_per_step=float(np.mean([s["reward"].mean() for s in streams])),
                          falls_per_1000=1000 * sum(s["falls"] for s in stats) / size,
                          falls_per_1000_by_brain={BRAINS[k]: round(1000 * float(falls_by_brain[k] / by_brain[k]), 3)
                                                   for k in range(len(BRAINS)) if by_brain[k]},
                          clean_per_1000=1000 * sum(s["scored"] for s in stats) / size,
                          bouts=sum(s["bouts"] for s in stats), critic_loss=float(critic_loss),
                          epochs=epochs_run, **{k: float(v) for k, v in info.items()})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if done >= next_checkpoint:
                save(done)
                next_checkpoint += args.checkpoint_every
        save(done)
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in processes:
            process.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
