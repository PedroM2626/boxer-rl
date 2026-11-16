"""Traces both boxers step by step in one bout, to see how a knockdown comes about.

Every control step it records, per corner: torso height and uprightness, the
torso's horizontal speed, the distance between the boxers, the distance to
the ring centre, contact flags (bodies, ropes) and clean punches. Writes a
compressed ``.npz`` and prints the 3 s before each knockdown.

    python -m tools.mirror_trace --red-tactics checkpoints/whole-body-power \
        --blue-tactics checkpoints/whole-body-power --seed 0 --seconds 120 \
        --punch-impulse 2 --output runs/mirror_trace_seed0.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import numpy as np

from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PUNCH_IMPULSE_ENV, PairedFoundation
from boxer.spar_match import bout, load_brain


def trace(pair, motor, brains, seed, seconds, knockdowns=3):
    rows = []

    def observer(elapsed, pair, outcome):
        roots = [pair.data.qpos[m['root']:m['root'] + 3].copy() for m in pair.maps]
        speeds = [np.linalg.norm(pair.data.qvel[m['rootv']:m['rootv'] + 2]) for m in pair.maps]
        up = [pair.data.sensor(m['side'] + 'upvector_torso').data[2] for m in pair.maps]
        rows.append(dict(
            time=elapsed,
            height=[r[2] for r in roots],
            up=up,
            speed=speeds,
            centre=[np.linalg.norm(r[:2]) for r in roots],
            distance=float(np.linalg.norm(roots[0][:2] - roots[1][:2])),
            body_contact=bool(pair.bodies_touching()),
            legs=bool(pair.legs_touching()),
            ropes=pair.on_ropes().tolist(),
            scored=np.asarray(outcome['scored']).tolist(),
            fallen=np.asarray(outcome['fallen']).tolist()))

    result, _, _ = bout(pair, motor, seed, seconds, False, brains, knockdowns, observer=observer)
    return result, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/spar-native'))
    parser.add_argument('--red-tactics', type=Path, required=True)
    parser.add_argument('--blue-tactics', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--punch-impulse', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.punch_impulse is not None:
        os.environ[PUNCH_IMPULSE_ENV] = str(args.punch_impulse)
    foundation = UnassistedG1('spar')
    motor = jax.jit(inference(foundation, load_policy_params(args.checkpoint)[0]))
    brains = {0: load_brain(args.red_tactics), 1: load_brain(args.blue_tactics)}
    pair = PairedFoundation(foundation, ring=True)
    result, rows = trace(pair, motor, brains, args.seed, args.seconds)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    times = arrays['time']
    print('knockdowns', result['knockdowns_suffered'], 'restarts', [r['time'] for r in result['restarts']],
          'clean', result['clean_punches'])
    for restart in result['restarts']:
        window = (times > restart['time'] - 3.) & (times <= restart['time'] + .02)
        print(f"--- restart at {restart['time']} s, fallen {restart['fallen']}")
        for i in np.flatnonzero(window)[::10]:
            print(f"t {times[i]:6.2f} h {np.round(arrays['height'][i], 2)} up {np.round(arrays['up'][i], 2)} "
                  f"v {np.round(arrays['speed'][i], 2)} d {arrays['distance'][i]:.2f} "
                  f"centre {np.round(arrays['centre'][i], 2)} body {arrays['body_contact'][i]} "
                  f"legs {arrays['legs'][i]} ropes {arrays['ropes'][i]}")


if __name__ == '__main__':
    main()
