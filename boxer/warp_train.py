"""Self-play PPO on the mujoco_warp env: warp steps the physics on the GPU,
jax runs the learner on the CPU inside the same process.

Why this split: the 6 GB card has already committed ~2 GB to warp's batch at
2048 worlds; a second CUDA context (XLA on GPU) on top is the exact OOM that
kills WSL, so the learner stays on CPU where its memory lives in system RAM.
The batch geometry is brax-parity -- every iteration collects
``batch_size * num_minibatches * unroll`` learner transitions and runs
``num_updates_per_batch`` epochs of ``batch_size * unroll``-wide minibatch
steps over them -- because that is the compute mix that trained every MJX arm
without divergence. The network parameterisation follows brax 0.14's default
as settled by tools/brax_ppo_probe.py: one MLP with a 2 x action_size head
whose halves are the Normal's loc and raw scale (scale = softplus(raw) +
min_std 0.001), actions squashed through tanh, the log-prob taken at the raw
action, silu MLPs with lecun_uniform kernels, plain adam, advantages
normalised per minibatch inside the loss, value loss scaled 0.5 x 0.5.

Deliberate deviations from brax, each documented where it lives:
* the log-prob is taken at the raw pre-tanh action (brax stores the same
  ``raw_action``), so the density is the textbook squashed one; the
  parameter-free tanh jacobian cancels in the ratio and is omitted.
* GAE is recomputed inside the loss from the current critic like brax. Two
  differences, both because WarpBoxing respawns inside ``step``: a round that
  runs out its clock keeps its bootstrapped timeout value in the delta, where
  brax zeroes the whole truncated delta; and a round that ends with a boxer on
  the floor has its bootstrap masked to zero, because the observation behind a
  termination is the respawn, and brax gets a zero terminal value for free
  where we would otherwise pay the agent ~26 discounted reward to fall.
* the entropy term keeps brax's analytic scale part but drops its sampled
  tanh-jacobian correction.
* rounds are per world: ``WarpBoxing.step`` respawns the rows that closed one
  and returns their next round's first observation, so ``--round-steps`` is a
  cap, not a clock the batch shares, and needs no alignment with the iteration
  length.

Run in the warp venv with jax pinned to CPU:

    JAX_PLATFORMS=cpu PYTHONPATH=. \
      /home/pedro/venvs/warp_test/bin/python -m boxer.warp_train ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import flax.linen as linen
import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.warp_env import WarpBoxing

if jax.devices()[0].platform != "cpu":
    raise SystemExit(
        "the warp trainer must keep jax on CPU (JAX_PLATFORMS=cpu): "
        "mujoco_warp owns the CUDA device in this process")

LECUN = jax.nn.initializers.lecun_uniform()
TANH_MIN_STD = 0.001


class MLP(linen.Module):
    """brax's MLP: silu after every hidden layer, bare linear output."""
    widths: tuple
    out: int

    @linen.compact
    def __call__(self, x):
        for width in self.widths:
            x = linen.silu(linen.Dense(width, kernel_init=LECUN)(x))
        return linen.Dense(self.out, kernel_init=LECUN)(x)


class RunningNorm:
    """Running observation statistics, brax's normalize_observations=True."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        n = batch.shape[0]
        mean, var = batch.mean(axis=0), batch.var(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.mean += delta * (n / total)
        self.var = (self.var * self.count + var * n
                    + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var,
                "count": np.asarray(self.count)}

    def load(self, state) -> None:
        self.mean, self.var = np.asarray(state["mean"]), np.asarray(state["var"])
        self.count = float(state["count"])


def split_params(parameters, std_bounds=(0.05, 2.0)):
    """(batch, 2A) -> loc, scale: brax's NormalTanhDistribution, bounded.

    The clip is the deliberate deviation brax's own arms did not need: warp1's
    std grew 0.71 -> 27.4 over 356 iterations while entropy climbed 33 -> 110,
    because a tanh-squashed Normal with std ~16 is a saturated coin flip -- the
    "hits" that followed were flailing contacts, not punches. Bounding the scale
    (and dropping the entropy cost to brax's own 1e-4 default order of magnitude)
    keeps exploration from out-running the reward.
    """
    loc, raw_scale = jp.split(parameters, 2, axis=-1)
    return loc, jp.clip(jax.nn.softplus(raw_scale) + TANH_MIN_STD, *std_bounds)


def normal_logp(loc, scale, action):
    # brax evaluates the Normal at the stored post-tanh action; its
    # parameter-free tanh jacobian cancels in the ratio, so it is omitted.
    z = (action - loc) / scale
    return jp.sum(-0.5 * z ** 2 - jp.log(scale) - 0.5 * jp.log(2 * jp.pi),
                  axis=-1)


def make_act(policy, std_bounds):
    @jax.jit
    def act(params_p, obs, key):
        """(rows, D) normalised obs -> (env action, raw pre-tanh action, logp)."""
        loc, scale = split_params(policy.apply(params_p, obs), std_bounds)
        u = loc + scale * jax.random.normal(key, loc.shape)
        return jp.tanh(u), u, normal_logp(loc, scale, u)
    return act


def make_update(policy, value, tx, clip_epsilon, entropy_cost, value_cost,
                gamma, lam, std_bounds):
    """brax-parity update: GAE is recomputed inside the loss from the CURRENT
    value net (baseline = V(obs), bootstrap = V(terminal_obs)), so advantages
    fade as the critic fits -- the mechanism that kept brax's clip fraction
    low across the MJX arms. Advantages and returns are stop-gradients of
    that pass; only the policy ratio and the value prediction carry grads."""

    @jax.jit
    def update(params_p, params_v, opt_state, batch):
        def loss_fn(p_p, p_v, b):
            rows, t_steps = b["reward"].shape
            obs = b["obs"].reshape(rows * t_steps, -1)
            loc, scale = split_params(policy.apply(p_p, obs), std_bounds)
            scale = scale.reshape(rows, t_steps, -1)
            logp = normal_logp(loc.reshape(rows, t_steps, -1), scale,
                               b["raw_action"])
            v_seq = value.apply(p_v, obs).reshape(rows, t_steps)
            boot = value.apply(p_v, b["terminal_obs"])[:, 0]

            v_next = jp.concatenate([v_seq[:, 1:], boot[:, None]], axis=1)
            # A round that ends because the clock ran out is a truncation: the
            # next observation is a legitimate continuation, so brax's bootstrapped
            # timeout stays. A round that ends with a boxer on the floor is a
            # TERMINATION, and the observation behind it is the respawn -- a fresh
            # standing round worth ~26 discounted reward. Bootstrapping across that
            # seam pays the agent ~26 for falling, which is the exact bias this
            # whole change was meant to remove.
            cont = 1.0 - b["done"] * (1.0 - b["timeout"])
            delta = b["reward"] + gamma * v_next * cont - v_seq

            def rev(acc, x):
                cut = jp.where(x[1] > 0.0, 0.0, gamma * lam)
                acc = x[0] + cut * acc
                return acc, acc
            _, adv = jax.lax.scan(rev, jp.zeros(rows),
                                  (delta.T, b["done"].T), reverse=True)
            adv = adv.T
            vs = adv + jax.lax.stop_gradient(v_seq)
            adv, vs = jax.lax.stop_gradient((adv, vs))
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = jax.lax.stop_gradient(vs)

            ratio = jp.exp(logp - b["logp_old"])
            clipped = jp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -jp.mean(jp.minimum(ratio * adv_n, clipped * adv_n))
            v_loss = jp.mean((v_seq - ret) ** 2)
            entropy = jp.mean(jp.sum(
                jp.log(scale) + 0.5 * (1.0 + jp.log(2 * jp.pi)), axis=-1))
            total = (policy_loss + v_loss * 0.5 * value_cost
                     - entropy_cost * entropy)
            metrics = {"policy_loss": policy_loss, "value_loss": v_loss,
                       "entropy": entropy,
                       "clip_frac": jp.mean(jp.abs(ratio - 1.0) > clip_epsilon),
                       "std": jp.mean(scale)}
            return total, metrics
        (grads_p, grads_v), metrics = jax.grad(loss_fn, argnums=(0, 1),
                                               has_aux=True)(
            params_p, params_v, batch)
        updates, opt_state = tx.update((grads_p, grads_v), opt_state,
                                       (params_p, params_v))
        params_p, params_v = optax.apply_updates((params_p, params_v), updates)
        return params_p, params_v, opt_state, metrics
    return update


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run = ap.add_argument_group("run")
    run.add_argument("--total-steps", type=int, default=200_000_000,
                     help="learner steps; self-play counts two per world step")
    run.add_argument("--worlds", type=int, default=2048)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=Path("runs/warp1"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--save-every", type=int, default=15,
                     help="iterations between checkpoints (~40 over a 200M run)")
    env_g = ap.add_argument_group("environment")
    env_g.add_argument("--n-frames", type=int, default=5)
    env_g.add_argument("--spawn-distance", type=float, default=0.75,
                       help="warp1 trained at 1.3 m and never closed inside the 0.45-0.55 m "
                            "band where the arm can score; 0.75 puts a step and a half away")
    env_g.add_argument("--action-scale", type=float, default=0.25)
    env_g.add_argument("--round-steps", type=int, default=480,
                       help="cap on control steps per round; rounds also close "
                            "early on a knockdown, out of ring, or separation")
    env_g.add_argument("--imitation", type=Path, default=None,
                       help="runs/reference/jab_targets.json from "
                            "tools/retarget_dense.py; absent = pure RL, no shaping")
    env_g.add_argument("--w-imit", type=float, default=0.3)
    ppo = ap.add_argument_group("ppo")
    ppo.add_argument("--unroll-length", type=int, default=20)
    ppo.add_argument("--batch-size", type=int, default=2048,
                     help="brax semantics: learner rows per minibatch, each "
                          "row an unroll-length sequence, so the flat "
                          "transition count is batch_size * unroll_length")
    ppo.add_argument("--num-minibatches", type=int, default=8)
    ppo.add_argument("--num-updates-per-batch", type=int, default=4,
                     help="epochs over the collected transitions")
    ppo.add_argument("--learning-rate", type=float, default=3e-4)
    ppo.add_argument("--entropy-cost", type=float, default=5e-3,
                     help="train.py used brax's 1e-2 for a 25-DOF robot; on the "
                          "29-joint warp arm that let entropy run 33 -> 110 and "
                          "the action std 0.7 -> 27 in 356 iterations")
    ppo.add_argument("--std-min", type=float, default=0.3,
                     help="warp1 ran the std to the floor (0.22) and froze; a scale below "
                          "~0.3 cannot move a 29-joint arm off its mean")
    ppo.add_argument("--std-max", type=float, default=2.0,
                     help="cap on the tanh-Normal scale; past ~2 the squashed "
                          "action is a saturated coin flip whatever the mean says")
    ppo.add_argument("--discounting", type=float, default=0.97)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clipping-epsilon", type=float, default=0.2)
    ppo.add_argument("--value-cost", type=float, default=0.5,
                     help="brax vf_loss_coefficient; the MSE carries a further 0.5")
    ppo.add_argument("--policy-layers", type=int, nargs="+",
                     default=[128, 128, 128, 128])
    ppo.add_argument("--value-layers", type=int, nargs="+",
                     default=[256, 256, 256, 256, 256])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rows = args.worlds * 2
    transitions = args.batch_size * args.num_minibatches * args.unroll_length
    if transitions % rows:
        raise SystemExit(f"batch_size * num_minibatches * unroll_length "
                         f"({transitions:,}) must be divisible by the learner "
                         f"count ({rows})")
    iteration_steps = transitions // rows
    minibatch = args.batch_size * args.unroll_length
    if iteration_steps * rows % minibatch:
        raise SystemExit("geometry error: transitions not divisible by the "
                         "minibatch size")
    n_minib = iteration_steps * rows // minibatch
    per_iteration = iteration_steps * rows
    total_iterations = max(-(-args.total_steps // per_iteration), 1)
    print(f"{args.worlds} worlds / {rows} learners; iteration = "
          f"{iteration_steps} control steps x {rows} learners = "
          f"{per_iteration:,} learner transitions; {total_iterations} "
          f"iterations ~= {per_iteration * total_iterations:,} learner steps; "
          f"{args.num_updates_per_batch} epochs x {n_minib} minibatches of "
          f"{minibatch:,}", flush=True)

    env = WarpBoxing(nworld=args.worlds, n_frames=args.n_frames,
                     spawn_distance=args.spawn_distance,
                     action_scale=args.action_scale,
                     round_steps=args.round_steps,
                     imitation_path=str(args.imitation) if args.imitation else None,
                     w_imit=args.w_imit)
    policy = MLP(tuple(args.policy_layers), 2 * env.action_size)
    value = MLP(tuple(args.value_layers), 1)
    std_bounds = (args.std_min, args.std_max)
    act = make_act(policy, std_bounds)
    tx = optax.adam(learning_rate=args.learning_rate)
    update = make_update(policy, value, tx, args.clipping_epsilon,
                         args.entropy_cost, args.value_cost,
                         args.discounting, args.gae_lambda, std_bounds)

    key = jax.random.PRNGKey(args.seed)
    params_p = policy.init({"params": key}, jp.zeros((1, env.obs_size)))
    params_v = value.init({"params": jax.random.split(key)[1]},
                          jp.zeros((1, env.obs_size)))
    norm = RunningNorm(env.obs_size)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "warp_config.json").write_text(json.dumps({
        "worlds": args.worlds, "action_size": env.action_size,
        "obs_size": env.obs_size, "round_steps": args.round_steps,
        "n_frames": args.n_frames, "spawn_distance": args.spawn_distance,
        "action_scale": args.action_scale,
        "imitation": {"path": str(args.imitation) if args.imitation else None,
                      "w_imit": args.w_imit},
        "policy_layers": list(args.policy_layers),
        "value_layers": list(args.value_layers),
        "ppo": {k: getattr(args, k) for k in (
            "unroll_length", "batch_size", "num_minibatches",
            "num_updates_per_batch", "learning_rate", "entropy_cost",
            "discounting", "gae_lambda", "clipping_epsilon", "value_cost",
            "std_min", "std_max")},
    }, indent=2) + "\n")

    ckpt_dir = args.output / "checkpoint"
    latest = ckpt_dir / "latest.txt"
    step = 0
    resumed = False
    if args.resume:
        if not latest.exists():
            raise SystemExit(f"--resume: no {latest} under {ckpt_dir}")
        path = ckpt_dir / f"step-{latest.read_text().strip()}.pkl"
        params_p, params_v, opt_state, norm_state, step = pickle.loads(
            path.read_bytes())
        norm.load(norm_state)
        resumed = True
        print(f"resumed {path} at {step:,} learner steps", flush=True)
    if not resumed:
        opt_state = tx.init((params_p, params_v))

    obs = env.reset(seed=args.seed)
    progress_log = args.output / "progress.jsonl"
    clock = {"start": time.time(), "step": step, "tick": time.time()}

    start_iteration = step // per_iteration
    for i in range(1, total_iterations - start_iteration + 1):
        iteration = start_iteration + i
        t_it = iteration_steps
        obs_buf = np.zeros((t_it, rows, env.obs_size), np.float32)
        raw_buf = np.zeros((t_it, rows, env.action_size), np.float32)
        logp_buf = np.zeros((t_it, rows), np.float32)
        rew_buf = np.zeros((t_it, rows), np.float32)
        done_buf = np.zeros((t_it, rows), np.float32)
        timeout_buf = np.zeros((t_it, rows), np.float32)
        hits_red = hits_blue = 0.0
        dist_mean = imit_mean = carry_mean = 0.0
        floor_mean = np.zeros(2)
        both_up_mean = 0.0
        rounds_mean = fell_mean = ring_mean = far_mean = 0.0
        flat_obs = obs.reshape(rows, env.obs_size)
        for t in range(t_it):
            key, sub = jax.random.split(key)
            actions, raw, logp = act(params_p, jp.asarray(
                norm.normalize(flat_obs)), sub)
            obs, reward, done, parts = env.step(
                np.asarray(actions).reshape(env.nworld, 2, -1))
            obs_buf[t] = flat_obs
            raw_buf[t] = np.asarray(raw)
            logp_buf[t] = np.asarray(logp)
            rew_buf[t] = np.asarray(reward).reshape(rows)
            done_buf[t] = np.repeat(done.astype(np.float32), 2)
            timeout_buf[t] = np.repeat(parts["timed_out"].astype(np.float32), 2)
            hits_red += float(np.sum(parts["scored"]))
            hits_blue += float(np.sum(parts["taken"]))
            dist_mean += float(np.mean(parts["distance"]))
            imit_mean += float(np.mean(parts["imit"]))
            carry_mean += float(np.mean(parts["carry"]))
            floor_mean += np.mean(parts["on_floor"], axis=0)
            both_up_mean += float(np.mean(1.0 - np.max(parts["on_floor"], axis=1)))
            rounds_mean += float(np.sum(done))
            fell_mean += float(np.sum(parts["fell"]))
            ring_mean += float(np.sum(parts["left_ring"]))
            far_mean += float(np.sum(parts["too_far"]))
            flat_obs = obs.reshape(rows, env.obs_size)
            step += rows
        terminal_obs = flat_obs.copy()

        # No round clock here any more: WarpBoxing.step respawns the worlds that
        # closed a round and hands back their first observation, with done=1 so
        # the GAE mask refuses to bootstrap across the seam.

        # stats update before SGD, brax order: the stored raw observations are
        # re-normalised with the fresh statistics for the updates, while the
        # behaviour log-probs stay those computed under the older ones.
        norm.update(obs_buf.reshape(-1, env.obs_size))
        normed = norm.normalize(
            obs_buf.reshape(-1, env.obs_size)).reshape(t_it, rows, -1)
        term_normed = norm.normalize(terminal_obs)
        minibatch_rows = rows // n_minib
        metrics = {}
        for epoch in range(args.num_updates_per_batch):
            perm = np.random.default_rng(
                (args.seed, iteration, epoch)).permutation(rows)
            for m in range(n_minib):
                idx = perm[m * minibatch_rows:(m + 1) * minibatch_rows]
                batch = {
                    "obs": jp.asarray(normed[:, idx].transpose(1, 0, 2)),
                    "raw_action": jp.asarray(raw_buf[:, idx].transpose(1, 0, 2)),
                    "logp_old": jp.asarray(logp_buf[:, idx].T),
                    "reward": jp.asarray(rew_buf[:, idx].T),
                    "done": jp.asarray(done_buf[:, idx].T),
                    "timeout": jp.asarray(timeout_buf[:, idx].T),
                    "terminal_obs": jp.asarray(term_normed[idx]),
                }
                params_p, params_v, opt_state, metrics = update(
                    params_p, params_v, opt_state, batch)

        now = time.time()
        interval, delta = now - clock["tick"], step - clock["step"]
        clock["step"], clock["tick"] = step, now
        sps = delta / interval if interval > 0 else 0.0
        record = {
            "step": int(step), "iteration": iteration,
            "walltime": now - clock["start"],
            "learner_steps_per_second": sps,
            "world_steps_per_second": sps / 2,
            "reward_red": float(rew_buf[:, 0::2].mean()),
            "reward_blue": float(rew_buf[:, 1::2].mean()),
            "hits_red": hits_red, "hits_blue": hits_blue,
            "distance": dist_mean / t_it, "imit": imit_mean / t_it,
            "carry": carry_mean / t_it,
            "down_red": float(floor_mean[0] / t_it),
            "down_blue": float(floor_mean[1] / t_it),
            "both_up": both_up_mean / t_it,
            # how long a fight lasts now, and how much of it ends with someone on
            # the floor rather than the clock running out
            "rounds": rounds_mean,
            "round_length": t_it * env.nworld / max(rounds_mean, 1.0),
            # which of the three closure reasons is actually firing, as a fraction
            # of the rounds closed (they can co-occur, so these need not sum to 1)
            "end_fell": fell_mean / max(rounds_mean, 1.0),
            "end_ring": ring_mean / max(rounds_mean, 1.0),
            "end_far": far_mean / max(rounds_mean, 1.0),
            **{k: float(jp.asarray(v)) for k, v in metrics.items()},
        }
        with progress_log.open("a") as sink:
            sink.write(json.dumps(record) + "\n")
        print(f"[{step:>12,}] it {iteration:>4} {sps:>9,.0f} learner/s  "
              f"hits r/b {hits_red / rows:.2f}/{hits_blue / rows:.2f}  "
              f"dist {dist_mean / t_it:.3f} m  "
              f"floor {record['down_red']:.2f}/{record['down_blue']:.2f} "
              f"up{record['both_up']:.2f} "
              f"round {record['round_length']:.0f}st "
              f"ends f{record['end_fell']:.2f}/r{record['end_ring']:.2f}"
              f"/d{record['end_far']:.2f}  "
              f"r_red {record['reward_red']:+.2f}  "
              f"ploss {record.get('policy_loss', float('nan')):+.3f} "
              f"vloss {record.get('value_loss', float('nan')):.3f} "
              f"ent {record.get('entropy', float('nan')):+.2f} "
              f"std {record.get('std', float('nan')):.2f} "
              f"clip {record.get('clip_frac', float('nan')):.3f}", flush=True)

        if iteration % args.save_every == 0 or iteration == total_iterations:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"step-{step}.pkl"
            blob = (jax.tree.map(np.asarray, params_p),
                    jax.tree.map(np.asarray, params_v),
                    jax.tree.map(np.asarray, opt_state),
                    norm.state(), int(step))
            path.write_bytes(pickle.dumps(blob))
            (ckpt_dir / "latest.txt").write_text(str(step))
            print(f"checkpoint -> {path}", flush=True)

    print(f"done: {step:,} learner steps, checkpoint at {ckpt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
