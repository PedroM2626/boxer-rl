"""Fixed matchups, scored in the physics that produced the arms.

Self-play reward is not a monotonic progress signal: a policy that gets better at
boxing can score less because its opponent is also better, so nothing in a training
log answers "can it win?". ``boxer/arena.py`` was built to answer that for the MJX
arms and cannot read a warp arm: the two envs disagree about the guard-error block
of the observation (SITE frames there, sensors here), and the final warp1 weights
read as an upright boxer punching in warp (+0.89 reward/step) while the same arrays
collapse both bodies inside two seconds on the MJX env. A win rate measured in the
wrong physics is worse than no win rate, so this file does the same accounting
natively on ``WarpBoxing``.

One deliberate difference from ``arena.play``: ``WarpBoxing.step`` respawns the
worlds that closed a round, so a logged window holds as many rounds per world as
that world closed, not one round the batch shares. Each closure is a result --
whoever is out on the step the round ends on -- and every fraction below is over
closed rounds. A round that runs its clock out with both corners up is
``undecided``: the judges' decision, and the most common outcome of a boxer who
cannot hurt anyone.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_arena \
        runs/warp7 runs/warp6 --worlds 256

Run it with ``--self-check`` to test the accounting without a matchup.
"""
from __future__ import annotations

import argparse

import numpy as np

from boxer.warp_env import WarpBoxing
from boxer.warp_train import make_opponent, warp_run

# The scorecard's target weights, banded on a single hit's value: head 1.0, torso
# 0.5, pelvis/hip/thigh 0.35. The same bands the training log uses, so an arena
# number and a log number can be compared without re-deriving either.
HEAD, TORSO = 0.999, 0.499

# Geometry a run trained under: name -> (type, default, env attribute). The
# defaults are WarpBoxing's own; a run's warp_config.json overrides them, and a
# flag overrides that.
GEOMETRY = (
    ("n_frames", int, 5, "n_frames"),
    ("spawn_distance", float, 0.75, "_spawn_distance"),
    ("action_scale", float, 0.25, "_action_scale"),
    ("round_steps", int, 500, "_round_steps"),
    ("damage_cap", float, 3.0, "_damage_cap"),
    ("stun_steps", int, 10, "_stun_steps"),
)


def round_bounds(done: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(world, start, end) for every round that closed in a (steps, worlds) log.

    Rounds are ordered per world by the step they closed on. Round j runs from the
    step after round j-1's closure through its own end step, inclusive: the respawn
    happens on the ``done`` step, so the observation handed back at that index
    already belongs to the next round, while the metrics logged with it belong to
    the one that ended.
    """
    end, world = np.nonzero(done > 0.5)  # rows of the log are steps, columns worlds
    order = np.lexsort((end, world))  # the LAST key is primary: group by world
    world, end = world[order], end[order]
    first = np.concatenate(([True], world[1:] != world[:-1]))
    previous = np.where(first, -1, np.concatenate(([-1], end[:-1])))
    return world, previous + 1, end


def per_round(values: np.ndarray, world, start, end) -> np.ndarray:
    """Sums a (steps, worlds[, corner]) log over each closed round."""
    running = np.cumsum(np.asarray(values, dtype=np.float64), axis=0)
    tail, head = running[end, world], running[np.maximum(start - 1, 0), world]
    # The mask belongs to the round axis. Broadcast it as (rounds, 1) or a 1-D
    # ``start > 0`` lines itself up with the corner axis instead, drops the leading
    # term for every round after the first, and reads as one round's punches
    # counted into the next -- which is what this function did before the
    # self-check caught it.
    mask = (start > 0).reshape((-1,) + (1,) * (head.ndim - 1))
    return tail - np.where(mask, head, 0.0)


def outcomes(done: np.ndarray, out_by: np.ndarray):
    """Per-closed-round ``(a_win, b_win, draw, undecided)``, one entry per closure.

    A is the red corner and B the blue one. A round's result is fixed by the step
    it ends on: ``out_by`` is a per-step indicator, not an accumulator, so reading
    it anywhere else is meaningless -- a boxer that was down mid-round and got back
    up has not lost.
    """
    world, _start, end = round_bounds(done)
    red_out = out_by[end, world, 0] > 0.5
    blue_out = out_by[end, world, 1] > 0.5
    return (blue_out & ~red_out, red_out & ~blue_out, red_out & blue_out,
            ~red_out & ~blue_out)


def play(env: WarpBoxing, policy_a, policy_b, steps: int,
         seed: int = 0) -> dict[str, float]:
    """Rolls ``steps`` control steps of ``env.nworld`` parallel rounds, A as red.

    Returns means over the rounds that closed inside the window, plus ``rounds``,
    how many there were. ``a_win + b_win + draw + undecided`` is 1 as long as one
    round closed.
    """
    # log name -> the name WarpBoxing reports it under
    CHANNELS = (("out_by", "out_by"), ("damage", "damage"), ("scored", "scored_by"),
                ("attempts", "attempts"), ("on_floor", "on_floor"),
                ("distance", "distance"), ("fell", "fell"),
                ("judged_out", "judged_out"), ("left_ring", "left_ring"),
                ("too_far", "too_far"), ("timed_out", "timed_out"))
    obs = env.reset(seed=seed)
    shape = (steps, env.nworld)
    corner_channels = {"out_by", "damage", "scored", "attempts", "on_floor"}
    log = {key: np.zeros(shape + (2,) if key in corner_channels else shape,
                         np.float32) for key, _ in CHANNELS}
    log["done"] = np.zeros(shape, np.float32)

    for t in range(steps):
        actions = np.zeros((env.nworld, 2, env.action_size), np.float32)
        actions[:, 0] = policy_a(obs[:, 0], t)
        actions[:, 1] = policy_b(obs[:, 1], t)
        obs, _reward, done, parts = env.step(actions)
        log["done"][t] = done
        for key, name in CHANNELS:
            log[key][t] = parts[name]

    world, start, end = round_bounds(log["done"])
    a_win, b_win, draw, undecided = outcomes(log["done"], log["out_by"])
    scored = log["scored"].astype(np.float64)
    totals = {
        "punch": per_round(scored, world, start, end),
        "landed": per_round(scored > 0.0, world, start, end),
        "attempts": per_round(log["attempts"], world, start, end),
        "head": per_round(scored >= HEAD, world, start, end),
        "torso": per_round((scored >= TORSO) & (scored < HEAD), world, start, end),
        "body": per_round((scored > 0.0) & (scored < TORSO), world, start, end),
    }
    # A diverged physics world poisons every sum it contributes, and the arena has
    # printed NaN before. Keep the rounds that stayed finite and say how many were
    # thrown away, so a missing number is visible instead of reading as a zero.
    finite = np.isfinite(np.stack(list(totals.values()))).all(axis=(0, 2))
    finite &= np.isfinite(log["damage"][end, world]).all(axis=1)

    # Which way a win arrived: a corner put out on accumulated damage lost to
    # punches, and one out only because its torso rolled past the down threshold
    # lost to a fall. Without the split, winning because you fell on top of the
    # other boxer is indistinguishable from winning on the scorecards.
    knocked_out = log["damage"][end, world] >= env._damage_cap
    survived = (end - start + 1)[finite]
    dt = env.n_frames * float(env._model.opt.timestep)

    def mean(values) -> float:
        kept = np.asarray(values, dtype=np.float64)[finite]
        return float(kept.mean()) if kept.size else float("nan")

    def ratio(numerator, denominator) -> float:
        kept = np.stack([np.asarray(numerator, np.float64),
                         np.asarray(denominator, np.float64)])[:, finite]
        return float(kept[0].sum() / kept[1].sum()) if kept[1].sum() > 0 else 0.0

    return {
        "a_win": mean(a_win), "b_win": mean(b_win),
        "draw": mean(draw), "undecided": mean(undecided),
        "a_win_ko": mean(a_win & knocked_out[:, 1]),
        "a_win_down": mean(a_win & ~knocked_out[:, 1]),
        "b_win_ko": mean(b_win & knocked_out[:, 0]),
        "b_win_down": mean(b_win & ~knocked_out[:, 0]),
        "rounds": int(finite.sum()),
        "dropped_rounds": float(1.0 - finite.mean()) if finite.size else 1.0,
        # how the rounds ended, as fractions of closed rounds; these can co-occur
        # (a knockdown on the last step of the clock is both), so they need not sum
        "end_fell": mean(log["fell"][end, world]),
        "end_ko": mean(log["judged_out"][end, world]),
        "end_ring": mean(log["left_ring"][end, world]),
        "end_far": mean(log["too_far"][end, world]),
        "end_time": mean(log["timed_out"][end, world]),
        "round_seconds": float(survived.mean() * dt) if survived.size else float("nan"),
        # weighted hit value landed on the opponent per round (head 1.0, torso 0.5,
        # body 0.35). It is a value, not a count of punches -- see head/torso/body.
        "a_punch_value": mean(totals["punch"][:, 0]),
        "b_punch_value": mean(totals["punch"][:, 1]),
        "a_head": mean(totals["head"][:, 0]), "a_torso": mean(totals["torso"][:, 0]),
        "a_body": mean(totals["body"][:, 0]),
        "b_head": mean(totals["head"][:, 1]), "b_torso": mean(totals["torso"][:, 1]),
        "b_body": mean(totals["body"][:, 1]),
        "a_attempts": mean(totals["attempts"][:, 0]),
        "b_attempts": mean(totals["attempts"][:, 1]),
        # RoboStriker's counter: contacts above the force threshold over offensive
        # attempts. A ratio, because every sum in this project has been inflated by
        # repetition rather than by boxing.
        "a_accuracy": ratio(totals["landed"][:, 0], totals["attempts"][:, 0]),
        "b_accuracy": ratio(totals["landed"][:, 1], totals["attempts"][:, 1]),
        # posture over the whole window rather than per round: how much of the
        # fight a corner spent past the down threshold
        "floor_a": float(log["on_floor"][:, :, 0].mean()),
        "floor_b": float(log["on_floor"][:, :, 1].mean()),
        "both_up": float(1.0 - np.max(log["on_floor"], axis=2).mean()),
        "distance": float(log["distance"].mean()),
    }


def pool(forward: dict[str, float], reverse: dict[str, float]) -> dict[str, float]:
    """Pools a roll and its corner-swapped twin, attributing counts to the agent.

    In ``forward`` A holds the red corner; in ``reverse`` A holds blue, so every
    ``a_``/``b_`` key flips corners before the halves are averaged. ``rounds`` is a
    count rather than a mean, so it adds.
    """
    pooled = {}
    for key, value in forward.items():
        if key.startswith("a_"):
            other = "b_" + key[2:]
        elif key.startswith("b_"):
            other = "a_" + key[2:]
        else:
            other = key
        pooled[key] = (value + reverse[other] if key == "rounds"
                       else 0.5 * (value + reverse[other]))
    return pooled


def matchup(env: WarpBoxing, spec_a: str, spec_b: str, steps: int,
            seed: int = 0, swap_sides: bool = True,
            stochastic: bool = False) -> dict[str, float]:
    """Plays two opponents, optionally with the corners exchanged and pooled.

    ``swap_sides`` matters more than it looks: the two corners of the scene are not
    interchangeable experience -- red starts at -x facing +x, blue the other way --
    and a systematic corner advantage would otherwise read as skill. The warp env
    has no anchor to re-pin on the swapped side, which is the artifact that made
    the MJX arena's swap dishonest against a statue opponent.

    ``stochastic`` samples each checkpoint's actions the way training did instead
    of taking its mode; the two can rank the same pair of arms differently.
    """
    a = make_opponent(spec_a, env, stochastic=stochastic)
    b = make_opponent(spec_b, env, stochastic=stochastic)
    forward = play(env, a, b, steps, seed)
    if not swap_sides:
        return forward
    return pool(forward, play(env, b, a, steps, seed + 10_000))


def checkpoints(specs) -> list[tuple[str, dict]]:
    """(spec, recorded config) for the specs that are warp runs, in argument order."""
    return [(spec, warp_run(spec)[2]) for spec in specs
            if spec not in ("passive", "jabber")]


def env_kwargs(cfg: dict, worlds: int, overrides: dict) -> dict:
    """Env geometry for a matchup: the run's own recorded numbers win, a flag
    overrides them, and WarpBoxing's default fills what an older run never
    recorded (``damage_cap`` and ``stun_steps`` postdate the first warp arms)."""
    kwargs = {"nworld": worlds}
    for name, _type, default, _attr in GEOMETRY:
        kwargs[name] = next((v for v in (overrides.get(name), cfg.get(name), default)
                             if v is not None))
    return kwargs


def geometry(env: WarpBoxing) -> dict:
    return {name: getattr(env, attr) for name, _t, _d, attr in GEOMETRY}


def _check() -> int:
    """Accounting tests on synthetic logs, then one real roll of two statues.

    The synthetic half is the same four cases tests/test_arena.py pins the MJX
    arena with, restated for logs in which one world closes several rounds. The
    real roll is the null hypothesis: nobody attacks, so nobody should land.

        PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_arena \
            --self-check
    """
    def expect(name, condition):
        print(f"  {'ok  ' if condition else 'FAIL'} {name}", flush=True)
        assert condition, name

    done = np.zeros((10, 2), np.float32)
    out_by = np.zeros((10, 2, 2), np.float32)
    done[4, 0], out_by[4, 0, 1] = 1.0, 1.0
    a_win, b_win, draw, undecided = outcomes(done, out_by)
    expect("a knockdown of blue is a win for a",
           a_win[0] and not b_win[0] and not draw[0] and not undecided[0])

    done[:] = out_by[:] = 0.0
    done[3, 0], out_by[3, 0] = 1.0, 1.0
    a_win, b_win, draw, undecided = outcomes(done, out_by)
    expect("both corners out is a draw", draw[0] and not a_win[0]
           and not undecided[0])

    done[:] = out_by[:] = 0.0
    done[8, 1] = 1.0
    a_win, b_win, draw, undecided = outcomes(done, out_by)
    expect("a round nobody ends is undecided", undecided[0] and not a_win[0])

    done[:] = out_by[:] = 0.0
    done[2, 0], out_by[2, 0, 1] = 1.0, 1.0
    done[8, 0] = 1.0
    scored = np.zeros((10, 2, 2), np.float64)
    scored[:5, 0, 0] = 1.0
    scored[6:, 0, 1] = 0.5
    a_win, b_win, _draw, undecided = outcomes(done, out_by)
    expect("only the terminal step decides, and a world can close twice",
           a_win[0] and undecided[1] and len(a_win) == 2)

    world, start, end = round_bounds(done)
    totals = per_round(scored, world, start, end)
    expect("per-round sums split the window at the closure",
           (start == [0, 3]).all() and (end == [2, 8]).all()
           and np.allclose(totals[0], [3.0, 0.0])
           and np.allclose(totals[1], [2.0, 1.5]))

    env = WarpBoxing(nworld=8, round_steps=40, spawn_jitter=0.0, yaw_jitter=0.0,
                     pose_noise=0.0)
    statue = make_opponent("passive", env)
    result = play(env, statue, statue, steps=60, seed=0)
    total = sum(result[key] for key in ("a_win", "b_win", "draw", "undecided"))
    expect("two statues close rounds and the fractions add up to 1",
           result["rounds"] > 0 and abs(total - 1.0) < 1e-9
           and result["dropped_rounds"] == 0.0)
    expect("nobody attacks, so nobody lands a punch",
           result["a_punch_value"] == 0.0 and result["b_punch_value"] == 0.0
           and result["a_attempts"] == 0.0 and result["b_attempts"] == 0.0)
    print(f"  roll: {result['rounds']} rounds, undecided {result['undecided']:.2f}, "
          f"end_fell {result['end_fell']:.2f}, end_ko {result['end_ko']:.2f}, "
          f"end_time {result['end_time']:.2f}, floor "
          f"{result['floor_a']:.2f}/{result['floor_b']:.2f}", flush=True)
    return 0


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("policy_a", nargs="?", default=None,
                    help="warp run dir or step .pkl, 'passive' or 'jabber'")
    ap.add_argument("policy_b", nargs="?", default="passive")
    ap.add_argument("--worlds", type=int, default=256,
                    help="parallel rounds per roll; 64 was noise in the MJX arena")
    ap.add_argument("--steps", type=int, default=None,
                    help="control steps to roll; the default is the env's round_steps, "
                         "so every world closes at least one round")
    ap.add_argument("--seed", type=int, default=0)
    overrides = ap.add_argument_group(
        "environment (each default comes from the run's own warp_config.json)")
    for name, kind, _default, _attr in GEOMETRY:
        overrides.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind,
                               default=None, help="override the recorded value")
    ap.add_argument("--no-swap", action="store_true", help="do not exchange corners")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from each policy instead of taking its mean action")
    ap.add_argument("--self-check", action="store_true",
                    help="test the round accounting and stop")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.self_check:
        return _check()
    if args.policy_a is None:
        raise SystemExit("give the corner to score: a warp run directory or a "
                         "step .pkl (or --self-check for the accounting tests)")

    specs = [args.policy_a, args.policy_b]
    runs = checkpoints(specs)
    if not runs:
        raise SystemExit("both corners are scripted, so there is nothing to score: "
                         "give at least one checkpoint")
    overrides = {name: getattr(args, name) for name, _t, _d, _a in GEOMETRY}
    env = WarpBoxing(**env_kwargs(runs[0][1], args.worlds, overrides))
    # Printed before any number: an agent trained at one action scale and evaluated
    # at another is a different agent, not a worse one, and the spawn distance and
    # the round clock change what it sees just as much.
    built = geometry(env)
    print("env: " + " ".join(f"{key}={value}" for key, value in built.items())
          + f" obs={env.obs_size} (from {runs[0][0]})", flush=True)
    for spec, cfg in runs:
        clash = {key: (value, built[key]) for key, value in cfg.items()
                 if key in built and float(value) != float(built[key])}
        if clash:
            raise SystemExit(
                f"{spec} trained with {clash} (recorded, running) and this arena is "
                "built from the other corner's run; pass the flag to say which "
                "value you mean")

    steps = args.steps or env._round_steps
    print(f"{args.policy_a}  vs  {args.policy_b}  ({env.nworld} worlds x {steps} "
          f"steps, corners {'swapped and pooled' if not args.no_swap else 'fixed'}, "
          f"{'stochastic' if args.stochastic else 'deterministic'})\n", flush=True)
    result = matchup(env, args.policy_a, args.policy_b, steps, seed=args.seed,
                     swap_sides=not args.no_swap, stochastic=args.stochastic)
    if not result["rounds"]:
        raise SystemExit(f"no round closed in {steps} steps: raise --steps or "
                         "lower --round-steps")
    for key, value in result.items():
        print(f"  {key:>16}: {int(value) if key == 'rounds' else f'{value:.3f}'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
