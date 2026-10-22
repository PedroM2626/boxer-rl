"""Measure easy-walking command coverage without running a physics learner."""
import argparse
from pathlib import Path

import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1, stage_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--output", type=Path, default=Path("runs/lateral_coverage_probe.json"))
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("sample count must be positive")
    rows = []
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.samples)
    for probability in (0.0, 0.5):
        env = object.__new__(UnassistedG1)
        env.stage, env._config = "walk_easy", stage_config("walk_easy")
        env._config.lateral_practice_probability = probability
        command = np.asarray(jax.jit(jax.vmap(env.sample_command))(keys))
        stationary = np.all(command == 0, axis=1)
        lateral = ((np.abs(command[:, 0]) < .01) & (np.abs(command[:, 2]) < .02)
                   & (np.abs(command[:, 1]) > .03))
        pure = (command[:, 0] == 0) & (command[:, 2] == 0) & (command[:, 1] != 0)
        rows.append(dict(probability=probability, samples=args.samples, seed=args.seed,
                         stationary=float(stationary.mean()), lateral_coverage=float(lateral.mean()),
                         pure_lateral=float(pure.mean()),
                         positive_lateral=float((pure & (command[:, 1] > 0)).mean()),
                         negative_lateral=float((pure & (command[:, 1] < 0)).mean())))
    atomic_json(args.output, rows)
    print(rows)


if __name__ == "__main__":
    main()
