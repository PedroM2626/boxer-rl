"""Two unassisted G1 boxers fight in native MuJoCo.

Each corner is a three-layer controller:

* legs and waist: the learned full-body motor (normally the ``spar`` stage,
  trained to keep balance while its arms punch and it gets shoved);
* arms: the same arm moves the ``spar`` stage trains under
  (``boxer.moves``: straights and hooks from a normal or high guard), latched at
  launch and retracted to the guard;
* tactics: footwork commands and when, with which hand and at what height to
  punch. Either the scripted ``RingBrain`` or a learned self-play policy from
  ``boxer.tactics_train`` (``PolicyBrain``), chosen per corner.

No pelvis springs, welds or root forces. By default the first knockdown
ends the bout; otherwise it goes the distance and is decided on clean
punches. With ``knockdowns=N`` (``--knockdowns N``, N > 1) a knockdown is
counted and the referee restarts the bout instead. Only real falls count
(pelvis below 0.55 m or torso tilted past the limit), not the foundation's
self-trip rule (a foot touching the robot's own other foot or shin): the G1s cannot get up, so both
are reset standing at fresh starting positions, scores and the clock carry
on, and the N-th knockdown of one boxer stops the bout (TKO). Each knockdown
scored is worth ``KNOCKDOWN_POINTS`` clean punches on the cards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from boxer import moves
from boxer.curriculum_env import LEFT_ARM, UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation

KNOCKDOWN_POINTS = 10
# Constant motor inputs, passed as NumPy so a step costs no extra JAX dispatch.
MOTOR_PRIVILEGED = np.zeros((2, 216), np.float32)
MOTOR_KEY = np.zeros(2, np.uint32)          # == jax.random.PRNGKey(0)


def motor_actions(motor, obs):
    """Both boxers' 29 joint targets from the motor (copy, safe to edit)."""
    action, _ = motor({'state': obs, 'privileged_state': MOTOR_PRIVILEGED}, MOTOR_KEY)
    return np.array(action)


class PunchArm:
    """NumPy twin of the batched arm controller (``boxer.moves``) for one boxer.

    Holds the normal or high guard and runs one straight or hook at a time,
    latched at launch.
    """

    def __init__(self, pose, action_scale, dt):
        self.pose, self.scale, self.dt = np.asarray(pose), action_scale, dt
        self.age, self.hand, self.target, self.speed, self.aim = -1.0, 0, np.zeros(4), 1.0, 0.0
        self.move, self.high = moves.STRAIGHT, False
        # Learned correction to the punching arm's (shoulder, elbow) targets,
        # left-arm convention; set each step by ``arm_skill.ArmSkill``.
        self.residual = np.zeros(4)

    @property
    def ready(self):
        return self.age < 0

    def features(self):
        """(age fraction, active, hand sign, aim, hook, high guard), as ``TacticsEnv._arm_features``."""
        active = self.age >= 0
        extend, retract = moves.durations(np, self.move, self.dt)
        return np.array([max(self.age, 0) / (extend + retract), float(active),
                         (2. * self.hand - 1) if active else 0., self.aim if active else 0.,
                         float(active and self.move == moves.HOOK), float(self.high)])

    def launch(self, hand, aim, rng, move=moves.STRAIGHT):
        self.hand, self.age, self.aim, self.move = int(hand), 0.0, float(aim), int(move)
        self.target = moves.strike_pose(np, self.move, self.aim, rng.uniform(-.06, .06, 4))
        self.speed = rng.uniform(.9, 1.15)
        self.residual = np.zeros(4)

    def actions(self, high=None):
        """14 arm actions for this step; ``high`` sets the guard height."""
        if high is not None:
            self.high = bool(high)
        arms = np.zeros(14)
        for side in range(2):
            start = LEFT_ARM + 7 * side
            guard = self.pose[start:start + 4]
            rest = moves.rest_pose(np, moves.mirror(np, guard, side), self.high)
            pose = rest
            if self.age >= 0 and self.hand == side:
                pose = moves.punch_pose(np, self.move, self.aim, self.target, self.age, rest, self.dt)
                pose = pose + self.residual
            arms[7 * side:7 * side + 4] = (moves.mirror(np, pose, side) - guard) / self.scale
        if self.age >= 0:
            self.age += self.speed
            if moves.finished(np, self.move, self.age, self.dt):
                self.age, self.residual = -1.0, np.zeros(4)
        return arms


class RingBrain:
    """Scripted tactics: close to range, punch in combos, step back, circle."""

    def __init__(self, rng, aggression=1.0, reach=.62, move=None, high=False):
        """``move``: always throw this ``boxer.moves`` punch (default straights);
        ``high``: hold the high guard. Both exist for move diagnostics."""
        self.rng, self.aggression, self.reach = rng, aggression, reach
        self.move, self.high = move, high
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
            if self.move is not None:
                punch += (int(self.move),)
            # Combos: a short pause after most punches, a long reset now and then.
            self.wait = int(self.rng.integers(2, 12) if self.rng.random() < .6 * self.aggression
                            else self.rng.integers(20, 50))
        return np.asarray(command), punch


class PolicyBrain:
    """A learned tactics policy (``boxer.tactics_train``) playing one corner."""

    def __init__(self, params, deterministic=True):
        from boxer.tactics_env import tactics_policy
        policy = tactics_policy(params, deterministic=deterministic)

        def act(obs, key):
            key, sub = jax.random.split(key)
            return policy(obs, sub)[0], key
        # Splitting the key inside the jitted call saves per-step dispatch.
        self.act = jax.jit(act)
        self.key = jax.random.PRNGKey(0)

    def decide_pair(self, pair, i, arms, clock):
        from boxer.tactics_env import decode, observe
        bodies = [_body(pair, k) for k in range(2)]
        obs = observe(np, bodies[i], bodies[1 - i], arms[i].features(), arms[1 - i].features(), clock)
        tactics, self.key = self.act(np.asarray(obs, np.float32), self.key)
        tactics = np.asarray(tactics, np.float64)
        delta = bodies[1 - i]['pos'] - bodies[i]['pos']
        command, want, hand, aim, move, high = decode(tactics, np.linalg.norm(delta),
                                                      bodies[i]['rot'].T @ delta)
        punch = (int(hand), float(aim), int(move)) if want and arms[i].ready else None
        return command, punch, bool(high)


def _body(pair, i):
    m = pair.maps[i]
    rot = pair.data.site(m['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)[:2, :2].copy()
    rot /= max(np.linalg.norm(rot[:, 0]), 1e-6)
    return dict(pos=pair.data.qpos[m['root']:m['root'] + 2].copy(), rot=rot,
                vel=pair.data.qvel[m['rootv']:m['rootv'] + 2].copy(),
                up=float(pair.data.sensor(m['side'] + 'upvector_torso').data[2]),
                # float32, as when this subtracted the JAX array directly.
                height=float(np.float32(pair.data.qpos[m['root'] + 2]) - pair.init_height32))


def bout(pair, motor, seed, seconds=20.0, record=False, brains=None, knockdowns=1, arm_skills=None):
    """One bout; ``brains`` maps corner -> PolicyBrain, others are scripted.

    ``knockdowns``: knockdowns of one boxer that stop the bout (1: the first
    fall ends it, the original rule). ``arm_skills`` maps corner ->
    ``arm_skill.ArmSkill`` that steers that corner's punches.

    Networks run on the CPU even when a GPU is visible. GPU float32 matmuls
    use TF32 by default (motor actions differ by up to 5e-4), and a bout is
    chaotic enough that the same seed then plays out differently. On the CPU
    a seed always replays the same bout.
    """
    # Under knockdown rules only real falls count: a robot whose feet touch
    # (the foundation's self-trip rule) is still standing and boxes on.
    trips, pair.self_trip_falls = pair.self_trip_falls, knockdowns == 1
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            return _bout(pair, motor, seed, seconds, record, brains, knockdowns, arm_skills or {})
    finally:
        pair.self_trip_falls = trips


def _bout(pair, motor, seed, seconds, record, brains, knockdowns, arm_skills):
    rng = np.random.default_rng(seed)
    pair.reset(seed=seed)
    arms = [PunchArm(pair.pose, pair.action_scale, pair.dt) for _ in range(2)]
    brains = [(brains or {}).get(i) or RingBrain(np.random.default_rng([seed, i]))
              for i in range(2)]
    scored, thrown, frames, times, events = np.zeros(2), np.zeros(2, int), [], [], []
    hooks, guarded, high = np.zeros(2, int), np.zeros(2, int), [False, False]
    fallen = np.zeros(2, bool)
    downs, restarts = np.zeros(2, int), []
    turned_away = np.zeros(2, int)      # steps facing more than 45 deg off the opponent
    roped = np.zeros(2, int)            # steps touching the ropes
    steps = int(round(seconds / pair.dt))
    for t in range(steps):
        roots = [pair.data.qpos[m['root']:m['root'] + 2] for m in pair.maps]
        commands, launches = [], []
        for i, m in enumerate(pair.maps):
            delta = roots[1 - i] - roots[i]
            rotation = pair.data.site(m['side'] + 'imu_in_pelvis').xmat.reshape(3, 3)
            local = rotation[:2, :2].T @ delta
            turned_away[i] += abs(np.arctan2(local[1], local[0])) > np.pi / 4
            if isinstance(brains[i], PolicyBrain):
                command, punch, high[i] = brains[i].decide_pair(pair, i, arms, t / steps)
            else:
                command, punch = brains[i].decide(float(np.linalg.norm(delta)), local, arms[i])
                high[i] = getattr(brains[i], 'high', False)
            commands.append(command)
            if punch is not None:
                arms[i].launch(*punch[:2], rng, *punch[2:])
                thrown[i] += 1
                hooks[i] += len(punch) > 2 and punch[2] == moves.HOOK
        commands = np.clip(commands, [-.35, -.25, -.5], [.55, .25, .5])
        obs = pair.observation(commands)
        action = motor_actions(motor, obs)
        for i in range(2):
            if i in arm_skills:
                arm_skills[i].steer(pair, i, arms[i])
            action[i, LEFT_ARM:] = arms[i].actions(high[i])
            guarded[i] += high[i]
        outcome = pair.step(action)
        roped += pair.on_ropes()
        elapsed = (t + 1) * pair.dt
        if outcome['scored'].any():
            events.append(dict(time=round(elapsed, 3),
                               scored=outcome['scored'].tolist()))
        scored += outcome['scored']
        if record and t % 2 == 0:
            frames.append(pair.data.qpos.copy())
            times.append((elapsed, scored.copy(), downs.copy()))
        if outcome['done']:
            fallen = np.asarray(outcome['fallen'], bool)
            downs += fallen
            if downs.max() >= knockdowns or t == steps - 1:
                break
            # The referee's restart: both boxers stand again at fresh spots.
            restarts.append(dict(time=round(elapsed, 3), fallen=fallen.tolist()))
            pair.reset(seed=int(rng.integers(1 << 30)))
            arms = [PunchArm(pair.pose, pair.action_scale, pair.dt) for _ in range(2)]
            fallen = np.zeros(2, bool)
    stopped = downs >= knockdowns
    cards = scored + KNOCKDOWN_POINTS * downs[::-1]
    if knockdowns > 1:
        fallen = stopped
    if fallen.any() and not fallen.all():
        winner = int(np.flatnonzero(~fallen)[0])
        method = 'knockdown' if knockdowns == 1 else 'tko'
    elif fallen.all():
        winner, method = None, 'double knockdown'
    elif cards[0] != cards[1]:
        winner, method = int(np.argmax(cards)), 'points'
    else:
        winner, method = None, 'draw'
    result = dict(seed=seed, seconds=round(min(steps, t + 1) * pair.dt, 2), clean_punches=scored.tolist(),
                  knockdowns_suffered=downs.tolist(), restarts=restarts, cards=cards.tolist(),
                  punches_thrown=thrown.tolist(), hooks_thrown=hooks.tolist(),
                  high_guard_seconds=(guarded * pair.dt).round(2).tolist(),
                  turned_away_seconds=(turned_away * pair.dt).round(2).tolist(),
                  rope_seconds=(roped * pair.dt).round(2).tolist(), fallen=fallen.tolist(),
                  winner=None if winner is None else ('red', 'blue')[winner],
                  method=method, impact_events=events)
    return result, frames, times


def render(pair, frames, times, path, caption):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from boxer.paired_foundation import build_pair
    model = build_pair(pair.foundation, visual_fists=True, ring=pair.ring)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 100, -15
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=540, width=960) as renderer, \
            imageio.get_writer(path, fps=25) as writer:
        last_down, downs_before = -np.inf, np.zeros(2)
        for qpos, (elapsed, score, downs) in zip(frames, times):
            if (downs != downs_before).any():
                last_down, downs_before = elapsed, downs
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            roots = [qpos[m['root']:m['root'] + 3] for m in pair.maps]
            camera.lookat[:] = [*(np.mean(roots, axis=0)[:2]), .8]
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 44), fill='black')
            draw.text((10, 6), caption, fill='white')
            knockdowns = f'   knockdowns {int(downs[0])}-{int(downs[1])}' if downs.any() else ''
            draw.text((10, 25), f'{elapsed:6.2f} s   Red {int(score[0])}  -  Blue {int(score[1])}{knockdowns}'
                      '   | free legs, no pelvis support', fill='white')
            if elapsed - last_down < 2.:
                draw.rectangle((330, 470, 630, 510), fill='darkred')
                draw.text((345, 482), 'KNOCKDOWN - referee restart', fill='white')
            writer.append_data(np.asarray(image))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path("checkpoints/spar"))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--output', type=Path, default=Path('runs/spar/match.json'))
    parser.add_argument('--video', type=Path, help='render the first seed')
    parser.add_argument('--red-tactics', type=Path, help='learned tactics checkpoint for red')
    parser.add_argument('--blue-tactics', type=Path, help='learned tactics checkpoint for blue')
    parser.add_argument('--knockdowns', type=int, default=1,
                        help='knockdowns of one boxer that stop the bout; earlier ones restart it')
    parser.add_argument('--ring', action='store_true', help='fight inside the roped ring')
    parser.add_argument('--red-arm', type=Path, help='arm-skill checkpoint steering red punches')
    parser.add_argument('--blue-arm', type=Path, help='arm-skill checkpoint steering blue punches')
    parser.add_argument('--swap', action='store_true',
                        help='also play every seed with the tactics exchanged between corners')
    args = parser.parse_args(argv)
    params, metadata = load_policy_params(args.checkpoint)
    tactics = {i: PolicyBrain(load_policy_params(path)[0])
               for i, path in enumerate((args.red_tactics, args.blue_tactics)) if path}
    names = [str(path) if path else 'scripted' for path in (args.red_tactics, args.blue_tactics)]
    from boxer.arm_skill import ArmSkill
    skills = {i: ArmSkill(load_policy_params(path)[0])
              for i, path in enumerate((args.red_arm, args.blue_arm)) if path}
    # A corner's arm skill belongs to the policy named for it, like its tactics.
    names = [n + (f' + arm {p}' if p else '') for n, p in zip(names, (args.red_arm, args.blue_arm))]
    foundation = UnassistedG1('spar')
    motor = jax.jit(inference(foundation, params))
    pair = PairedFoundation(foundation, ring=args.ring)
    bouts = []
    schedule = [(seed, False) for seed in args.seeds]
    if args.swap:
        schedule += [(seed, True) for seed in args.seeds]
    for index, (seed, swapped) in enumerate(schedule):
        record = args.video is not None and index == 0
        brains = {1 - i if swapped else i: brain for i, brain in tactics.items()}
        arm_skills = {1 - i if swapped else i: skill for i, skill in skills.items()}
        result, frames, times = bout(pair, motor, seed, args.seconds, record, brains, args.knockdowns,
                                     arm_skills)
        result.update(red=names[1] if swapped else names[0], blue=names[0] if swapped else names[1])
        bouts.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'impact_events'}), flush=True)
        if record:
            render(pair, frames, times, args.video,
                   f"Red {result['red']} vs Blue {result['blue']} | seed {seed}")
    survived = sum(not any(b['fallen']) for b in bouts)
    # Win rate of the tactics named for red at launch, across both corner assignments.
    first = names[0]
    wins = sum(b['winner'] == ('red' if b['red'] == first else 'blue') for b in bouts if b['winner'])
    losses = sum(1 for b in bouts if b['winner']) - wins
    points = [b['clean_punches'][0 if b['red'] == first else 1] for b in bouts]
    against = [b['clean_punches'][1 if b['red'] == first else 0] for b in bouts]
    summary = dict(checkpoint=str(args.checkpoint), stage=metadata['stage'],
                   stage_steps=metadata['stage_steps'], checkpoint_sha256=metadata['sha256'],
                   bouts=len(bouts), went_the_distance=survived,
                   knockdowns=sum(b['method'] == 'knockdown' for b in bouts),
                   stoppages=sum(b['method'] in ('knockdown', 'tko', 'double knockdown') for b in bouts),
                   total_knockdowns=int(sum(sum(b['knockdowns_suffered']) for b in bouts)),
                   mean_seconds=float(np.mean([b['seconds'] for b in bouts])),
                   clean_punches_per_bout=float(np.mean([sum(b['clean_punches']) for b in bouts])),
                   punches_thrown_per_bout=float(np.mean([sum(b['punches_thrown']) for b in bouts])),
                   turned_away_seconds_per_boxer=float(np.mean([b['turned_away_seconds'] for b in bouts])),
                   ring=args.ring, rope_seconds_per_boxer=float(np.mean([b['rope_seconds'] for b in bouts])),
                   tactics=dict(first=names[0], second=names[1], first_wins=wins, first_losses=losses,
                                draws=len(bouts) - wins - losses,
                                first_clean_per_bout=float(np.mean(points)),
                                second_clean_per_bout=float(np.mean(against))),
                   assistance=False, results=bouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
