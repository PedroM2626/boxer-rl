"""Two unassisted G1 boxers fight in native MuJoCo.

Each corner is a three-layer controller:

* legs and waist: the learned full-body motor (normally the ``spar`` stage,
  trained to keep balance while its arms punch and it gets shoved);
* arms: the same straight-punch primitives the ``spar`` stage trains under
  (``curriculum_env.PUNCH_*``), latched at launch and retracted to the guard;
* tactics: a scripted ring brain choosing footwork commands and when, with
  which hand and at what height to punch. It is deliberately a controller, not
  a learned policy; it is the slot a self-play policy will take.

No pelvis springs, welds or root forces: the first knockdown ends the bout,
otherwise the bout goes the distance and is decided on clean punches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from boxer.curriculum_env import (LEFT_ARM, PUNCH_BODY, PUNCH_EXTEND, PUNCH_HEAD,
                                  PUNCH_RETRACT, RIGHT_ARM, UnassistedG1)
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation


class PunchArm:
    """NumPy twin of ``UnassistedG1.punch_arms`` driven by a launch request."""

    def __init__(self, pose, action_scale, dt):
        self.pose, self.scale = np.asarray(pose), action_scale
        self.extend, self.retract = PUNCH_EXTEND / dt, PUNCH_RETRACT / dt
        self.age, self.hand, self.target, self.speed = -1.0, 0, np.zeros(4), 1.0

    @property
    def ready(self):
        return self.age < 0

    def launch(self, hand, aim, rng):
        self.hand, self.age = int(hand), 0.0
        self.target = (PUNCH_BODY + aim * (PUNCH_HEAD - PUNCH_BODY)
                       + rng.uniform(-.06, .06, 4))
        self.speed = rng.uniform(.9, 1.15)

    def actions(self):
        arms = np.zeros(14)
        if self.age < 0:
            return arms
        age = self.age
        fraction = (age + 1) / self.extend if age < self.extend else \
            1 - (age - self.extend) / self.retract
        fraction = float(np.clip(fraction, 0, 1))
        sign = 1.0 if self.hand == 0 else -1.0
        start = LEFT_ARM if self.hand == 0 else RIGHT_ARM
        guard = self.pose[start:start + 4]
        offset = fraction * (self.target * [1, sign, sign, 1] - guard)
        arms[start - LEFT_ARM:start - LEFT_ARM + 4] = offset / self.scale
        self.age += self.speed
        if self.age >= self.extend + self.retract:
            self.age = -1.0
        return arms


class RingBrain:
    """Scripted tactics: close to range, punch in combos, step back, circle."""

    def __init__(self, rng, aggression=1.0, reach=.62):
        self.rng, self.aggression, self.reach = rng, aggression, reach
        self.wait = int(rng.integers(5, 30))
        self.circle, self.circle_steps = 0.0, 0

    def decide(self, distance, local, arm):
        """``local`` is the opponent's offset in this boxer's pelvis frame."""
        heading = float(np.clip(np.arctan2(local[1], local[0]), -.5, .5))
        direction = local / max(distance, 1e-6)
        if distance > self.reach + .08:
            speed = float(np.clip((distance - self.reach) * .8, .05, .4))
        elif distance < self.reach - .12:
            speed = -.15
        else:
            speed = 0.0
        command = [speed * direction[0], speed * direction[1], heading]
        if self.circle_steps <= 0 and self.rng.random() < .01:
            self.circle, self.circle_steps = self.rng.choice([-.1, .1]), 50
        if self.circle_steps > 0 and distance < self.reach + .2:
            command[1] += self.circle
            self.circle_steps -= 1
        self.wait -= 1
        punch = None
        if arm.ready and self.wait <= 0 and distance < self.reach + .1 and abs(heading) < .35:
            punch = (int(self.rng.integers(0, 2)), float(self.rng.random() > .5))
            # Combos: a short pause after most punches, a long reset now and then.
            self.wait = int(self.rng.integers(2, 12) if self.rng.random() < .6 * self.aggression
                            else self.rng.integers(20, 50))
        return np.asarray(command), punch


def bout(pair, motor, seed, seconds=20.0, record=False):
    rng = np.random.default_rng(seed)
    pair.reset(seed=seed)
    arms = [PunchArm(pair.pose, pair.action_scale, pair.dt) for _ in range(2)]
    brains = [RingBrain(np.random.default_rng([seed, i])) for i in range(2)]
    scored, thrown, frames, times, events = np.zeros(2), np.zeros(2, int), [], [], []
    fallen = np.zeros(2, bool)
    steps = int(round(seconds / pair.dt))
    for t in range(steps):
        roots = [pair.data.qpos[m['root']:m['root'] + 2] for m in pair.maps]
        commands, launches = [], []
        for i, m in enumerate(pair.maps):
            delta = roots[1 - i] - roots[i]
            rotation = pair.data.site(m['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)
            local = rotation[:2, :2].T @ delta
            command, punch = brains[i].decide(float(np.linalg.norm(delta)), local, arms[i])
            commands.append(command)
            if punch is not None:
                arms[i].launch(punch[0], punch[1], rng)
                thrown[i] += 1
        commands = np.clip(commands, [-.35, -.25, -.5], [.55, .25, .5])
        obs = pair.observation(commands)
        action, _ = motor({'state': jp.asarray(obs), 'privileged_state': jp.zeros((2, 216))},
                          jax.random.PRNGKey(0))
        action = np.asarray(action).copy()
        for i in range(2):
            action[i, LEFT_ARM:] = arms[i].actions()
        outcome = pair.step(action)
        if outcome['scored'].any():
            events.append(dict(time=round(pair.elapsed, 3),
                               scored=outcome['scored'].tolist()))
        scored += outcome['scored']
        if record and t % 2 == 0:
            frames.append(pair.data.qpos.copy())
            times.append((pair.elapsed, scored.copy()))
        if outcome['done']:
            fallen = np.asarray(outcome['fallen'], bool)
            break
    if fallen.any() and not fallen.all():
        winner, method = int(np.flatnonzero(~fallen)[0]), 'knockdown'
    elif fallen.all():
        winner, method = None, 'double knockdown'
    elif scored[0] != scored[1]:
        winner, method = int(np.argmax(scored)), 'points'
    else:
        winner, method = None, 'draw'
    result = dict(seed=seed, seconds=round(pair.elapsed, 2), clean_punches=scored.tolist(),
                  punches_thrown=thrown.tolist(), fallen=fallen.tolist(),
                  winner=None if winner is None else ('red', 'blue')[winner],
                  method=method, impact_events=events)
    return result, frames, times


def render(pair, frames, times, path, caption):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from boxer.paired_foundation import build_pair
    model = build_pair(pair.foundation, visual_fists=True)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 100, -15
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=540, width=960) as renderer, \
            imageio.get_writer(path, fps=25) as writer:
        for qpos, (elapsed, score) in zip(frames, times):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            roots = [qpos[m['root']:m['root'] + 3] for m in pair.maps]
            camera.lookat[:] = [*(np.mean(roots, axis=0)[:2]), .8]
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 44), fill='black')
            draw.text((10, 6), caption, fill='white')
            draw.text((10, 25), f'{elapsed:5.2f} s   Red {int(score[0])}  -  Blue {int(score[1])}'
                      '   | free legs, no pelvis support', fill='white')
            writer.append_data(np.asarray(image))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path("checkpoints/spar"))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--output', type=Path, default=Path('runs/spar/match.json'))
    parser.add_argument('--video', type=Path, help='render the first seed')
    args = parser.parse_args(argv)
    params, metadata = load_policy_params(args.checkpoint)
    foundation = UnassistedG1('spar')
    motor = jax.jit(inference(foundation, params))
    pair = PairedFoundation(foundation)
    bouts = []
    for index, seed in enumerate(args.seeds):
        record = args.video is not None and index == 0
        result, frames, times = bout(pair, motor, seed, args.seconds, record)
        bouts.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'impact_events'}), flush=True)
        if record:
            render(pair, frames, times, args.video,
                   f"G1 vs G1 | motor {metadata['stage']} @ {metadata['stage_steps']:,} | seed {seed}")
    survived = sum(not any(b['fallen']) for b in bouts)
    summary = dict(checkpoint=str(args.checkpoint), stage=metadata['stage'],
                   stage_steps=metadata['stage_steps'], checkpoint_sha256=metadata['sha256'],
                   bouts=len(bouts), went_the_distance=survived,
                   knockdowns=sum(b['method'] == 'knockdown' for b in bouts),
                   mean_seconds=float(np.mean([b['seconds'] for b in bouts])),
                   clean_punches_per_bout=float(np.mean([sum(b['clean_punches']) for b in bouts])),
                   punches_thrown_per_bout=float(np.mean([sum(b['punches_thrown']) for b in bouts])),
                   assistance=False, results=bouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
