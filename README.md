# boxer — two humanoids learning to fight each other in JAX

Two [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
humanoids spar in a single 3D physics world simulated entirely in JAX. Three ways of
training that duel are implemented — symmetric self-play, a frozen opponent, and a
best-response league with one network per corner — and they are measured against each
other rather than assumed, which is what the Results section below is.

The environment is a Brax `PipelineEnv` on the **MJX** backend (MuJoCo's JAX
port), trained with Brax's PPO. Collisions are real — a landed punch is a
contact between the fist geom and the opponent's head or torso geom, not a
proximity heuristic, and it is gated on the fist closing on a target while the
striker is still upright.

**Where this is at:** the physics, the throughput and the measurement machinery work;
the boxers do not box yet. The best policy holds a guard, closes distance and lands
~0.3 gated punches a round, and wins 64% of rounds against a passive opponent by being
the one still standing. Symmetric self-play at the same budget loses 88% of its rounds
to that passive opponent. Three reward interventions have been tried and reported as
null results rather than quietly dropped; the diagnosis and the experiments still open
are in [Results](#results).

## Results

Measured with `python -m boxer.bench` on a laptop RTX 3060 (6 GB) and a
16-logical-core CPU. A *world* is one physics simulation containing both
boxers; PPO sees twice as many *learners*, and a *control step* is 20 ms of
simulated time (5 substeps of 4 ms).

| Device | Worlds | Learners | Control steps/s | Learner steps/s | Physics substeps/s |
| --- | --- | --- | --- | --- | --- |
| CPU (jaxlib; no CUDA build on native Windows) | 32 | 64 | 65 | 130 | 325 |
| CPU | 128 | 256 | 65 | 130 | 326 |
| GPU via WSL2 | 64 | 128 | 1,728 | 3,457 | 8,642 |
| GPU via WSL2 | 256 | 512 | 2,662 | 5,325 | 13,312 |
| GPU via WSL2 | 1,024 | 2,048 | 3,069 | 6,139 | 15,347 |

Two things worth reading off this table:

- The GPU is ~40-47x the CPU, and CPU throughput is **flat in the world count**
  (65/s at both 32 and 128 worlds) because JAX on CPU leaves ~5 of the 16
  logical cores idle and `jax --xla_force_host_platform_device_count` segfaults
  on Windows. CPU is a development path, not a training path.
- VRAM is the binding constraint on this card, not FLOPs. 1,024 worlds only adds
  15% over 256 and spends the whole 6 GB (XLA logs `RESOURCE_EXHAUSTED` and
  recovers). 512 worlds *for training* -- which adds the unroll buffer, logits
  and Adam moments on top of the physics state -- needs a ~6 GB peak and thrashes
  on rematerialization, so 256 worlds is the setting that actually runs here.
  Raise it on a bigger card.

### The modes, at equal physics

Both arms of the original comparison saw **12.45M world steps** with 256 worlds
(`frozen` counts one learner per world and stopped at 12,451,840 learner steps;
`selfplay` counts two and stopped at 24,903,680 -- the same simulated sparring).
Every number below is **256 arena rounds**, corners exchanged and the halves
pooled, at `action_scale` 0.25 -- the scale the policies were trained at.

| A vs B | A win | B win | round | A hits | B hits | A heads | B heads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200M x passive | **0.938** | 0.049 | 1.24 s | 184.8 | -- | 40.7 | -- |
| 200M x jabber | **0.947** | 0.035 | 1.22 s | 161.7 | 0.28 | 34.2 | 0.00 |
| 200M x 12.45M | **0.754** | 0.203 | 1.27 s | 49.2 | 3.6 | 17.1 | 0.07 |
| 12.45M frozen x passive | 0.824 | 0.141 | 1.26 s | 3.14 | -- | 0.37 | -- |
| selfplay x passive | 0.869 | 0.104 | 1.26 s | 3.56 | 0.05 | 0.47 | 0.00 |
| league (alt/red) x passive | 0.867 | 0.123 | 1.25 s | 10.2 | 0.01 | 0.44 | 0.00 |

**Two earlier conclusions in this section were wrong and are withdrawn.** Both
were measured before the arena was fixed to evaluate checkpoints at the action
scale they trained against (see "What is honest here"): the arena built
`BoxingEnv()` and got the class default of 1.0, which multiplies both the
commanded joint excursion and the sampled noise by four. Measured that way, the
symmetric self-play arm lost 88% of its rounds to a passive opponent and the
best-response league arm lost 87%, which produced the claim that self-play is
"worse than not training". Re-measured at 0.25, **both beat the statue in 87% of
rounds**. The training-mode ranking that survived is much flatter than the one I
published: what mattered was the step count, not the mode.

What the corrected table does say:

- **Scale helped, monotonically.** Against the same passive opponent: 0.824 at
  12.45M, 0.938 at 204M. Head to head, the 204M arm takes 75% of rounds against
  20% for the 12.45M arm.
- **The three modes are within ~5 points of each other at equal budget** (0.82
  frozen, 0.87 self-play, 0.87 league). The mode is not where the project is
  stuck.
- **The `hits` column is not counting punches.** The 204M arm registers 185
  "hits" and 41 "head shots" per round, which at ~62 steps per round is 0.7
  contacts *per step*. Rendered, the behaviour is: guard up for ~0.3 s, close to
  chest contact, wrap both arms around the opponent's head and neck, and topple
  together into a ground clinch. The metric rewards that because hits accumulate
  per physics substep and nothing in the environment stops the action when the two
  are holding each other. See "What counts as a punch".

### Reweighting the reward changed nothing

Because the diagnosis so far was "the knockout term outshouts balance", the reward was
rescaled and retrained: `ko` 10 → 3, `upright` 0.3 → 0.8, and the range term multiplied
by `upright` instead of `alive` so that collapsing into the opponent stops collecting the
reward for being at fighting distance. A second arm trained under it, same opponent, same
worlds, same batch geometry:

| learner steps | arm | round | `down_red` | `down_blue` | `upright` |
| --- | --- | --- | --- | --- | --- |
| 0.66M | old | 50.1 | 0.80 | 0.24 | 44.4 |
| 0.66M | rescaled | 50.7 | 0.79 | 0.23 | 44.9 |
| 5.24M | old | 52.8 | 0.67 | 0.38 | 47.7 |
| 5.24M | rescaled | 52.5 | 0.72 | 0.33 | 47.6 |
| 9.83M | old | 54.9 | 0.46 | 0.59 | 50.1 |
| 9.83M | rescaled | 55.0 | 0.47 | 0.60 | 50.3 |

Null result at every matched step count. Two things follow. The first is that the shape
of the terminal and range terms is not what keeps the boxers on the floor. The second is
visible in the `old` column itself: `down_red` is still falling at 9.83M (0.80 → 0.46)
while `down_blue` -- a boxer with *no policy at all* -- rises from 0.24 to 0.59, so the
learner is getting harder to knock over and better at knocking over. The curve has not
plateaued, which means "more steps cannot help" was wrong as stated; what is still true is
that no amount of steps fixes a task whose first sub-skill is standing up, and that
sub-skill is measured separately rather than assumed.

Then the geometry was checked directly, because "nobody punches" has an obvious
candidate explanation: the fist has to reach. `tools/gate_calibration.py` drives the
scripted jab at a still opponent and counts contacts that happened while the punch gate
was open: **0 in 200 steps at 1.15 m, 9 at 0.9 m, 66 at 0.7 m.** So punching is
geometrically rare at exactly the distance the range term rewards, and the range target
was moved to 0.7 m and retrained. That is also a null result: at 2.62M learner steps the
0.7 m arm lands 0.20 gated punches per episode against the 0.9 m arm's 0.42, with
contacts falling from 50 to 41. Moving the reward closer to arm's reach did not make the
policy punch, which means the limit is not where the boxers stand but what they can do
with their legs.

The experiment designed to answer that — an opponent placed out of reach so that only
balance, facing and guard can be earned — was first run at 2.2 m and produced episodes of
9 steps, because the round ends when the boxers separate by more than `MAX_SEPARATION`
(2.4 m) and spawn jitter is +/-0.12 m per boxer. It measured the separation timeout, not
standing. Re-run at 2.0 m, where contact is still impossible and there is 0.4 m of
headroom.

### The falls are not about the opponent

The isolation run finally answered the question the three nulls were circling. With the
passive opponent placed 2.0 m away -- out of arm's reach, so almost nothing can happen
between them -- and 2.62M learner steps:

| arm, all at 2.62M learner steps | `down_red` | `hits_red` | contacts |
| --- | --- | --- | --- |
| standing only, opponent at 2.0 m | **0.82** | 0.00 | 3.0 |
| frozen, opponent in reach (range 0.9 m) | 0.72 | 0.42 | 50.2 |
| frozen, opponent in reach (range 0.7 m) | 0.76 | 0.20 | 40.7 |

A boxer with no opponent to fight, nothing to defend and no contact to make still falls
over in 82% of its rounds. Falling is therefore not caused by the opponent, not by the
distance the range reward prefers, and not by the knockout term's magnitude -- all three
of those were tested and came back null. It is the base skill, and this budget has not
enough of it.

That reverses the conclusion this README held an hour earlier. "More steps cannot help,
fix the reward first" was wrong: the 12.45M arm is still descending at its last
checkpoint (`down_red` 0.80 → 0.46), and the no-contact arm shows the same curve is being
spent on standing rather than on boxing. So the long run is the 200M-step frozen arm.

**Why 200M and not less.** [MuJoCo Playground](https://arxiv.org/abs/2502.08844) -- the
reference MJX stack for this exact robot -- trains the Unitree G1 to walk on flat ground
for **200M timesteps** and then another **100M** on rough terrain, with 8,192 parallel
environments, finishing in under 30 minutes on two RTX 4090s. Plain locomotion is a
strictly easier objective than a two-boxer duel with declared contact pairs, so 200M is
read as the floor here rather than a stretch goal. An intermediate checkpoint of this
run was going to be used as a stop signal at 50M steps; that was withdrawn before it
could fire, because 50M is a quarter of Playground's *walking* budget and the test would
have killed a run that had not yet had a chance.

The comparison also puts the real constraint in perspective: this laptop reaches ~2,200
learner steps/s at 256 worlds, so 200M costs ~25 hours -- roughly **50x** the reference
throughput gap. The binding limit on this project is steps per second, not willingness to
wait, and the bench above shows that 1,024 worlds buys only ~15% more on a 6 GB card.
What would still falsify the scale hypothesis: `down_red` above ~0.3 after the full 200M,
which would leave the control interface -- 29 joints under 50 Hz position actuators at
`action_scale` 0.25 -- as the remaining suspect. `python -m tools.check_falsifier` prints
the curve and the verdict.

For related work that is *not* an analogue, recorded so nobody re-reads them as one:
[FightLadder](https://arxiv.org/abs/2406.02081) is a video-game MARL benchmark built on
gym-retro's Street Fighter II, and the 2.5D fighting work around
[Little Fighter 2](https://arxiv.org/abs/1805.02070) uses an A3C agent over a discrete
state abstraction -- neither simulates a balanced humanoid, so neither says anything about
how many steps standing takes. The closest analogue found is
[ALMA](https://arxiv.org/abs/2504.14305), which trains a humanoid against an adversarial
upper body that pushes it, reports that "the robot inevitably falls initially", runs 4,096
parallel environments, and needs a curriculum that sorts adversarial motions by how long
the robot survives on them. That is the same failure this project is hitting, addressed
with curriculum rather than with reward weights.

The closest thing to this project's exact premise that is open source is
[mbaske/ml-selfplay-fighter](https://github.com/mbaske/ml-selfplay-fighter), "Self-Play
Boxing Match made with Unity Machine Learning Agents" -- Unity, two fighters, self-play,
PPO. Its `trainer_config.yaml` asks for **`max_steps: 200000000`** with `batch_size: 2048`,
`buffer_size: 51200`, `num_epoch: 3`, `learning_rate: 3e-4`, `gamma: 0.99`, `beta: 0.01`.
That is an independent implementation of this same idea choosing 200M as its budget, which
is the same floor MuJoCo Playground spends on the G1 just to walk -- and the outcome is
still a demo rather than a boxer. Two of the three hyperparameters worth copying are
already matched here (learning rate, entropy coefficient); the third, `gamma`, is not:
this environment discounts at 0.97 over a 20 ms control step, which throws away roughly
twice as much future as their setup, on a task where "keep balancing for the next second"
is the whole problem.

Also falsified, by measurement rather than argument, while looking for the cause of the
falls:


- **Disabling Euler damping does not make the robot fall.** 54 steps until down with
  `mjDSBL_EULERDAMP` set and with it cleared, identically. The G1's joints carry no
  `damping` at all -- only `frictionloss` and `armature` -- so the flag has nothing to act
  on; the actuator's own `kv` is the only velocity feedback in the model.
- **The centre of mass is inside the feet in every stance** (`tools/com_check.py`): 7.0 cm
  of margin at `qpos0`, 8.4 cm at the upstream home keyframe, 6.8 cm with the fitted guard
  arms. The pose is not statically impossible.
- **The control targets are consistent with the pose**: all 29 actuators have
  `ctrl == qpos` in the keyframe, so nothing is being driven somewhere else at t=0.

Which leaves the boring explanation: `ctrl == qpos` means zero error means **zero torque**,
and a position-controlled humanoid with no policy is not holding anything up. Plain MuJoCo
with the same model topples at the same moment -- 270 substeps, torso from 1.04 m to the
floor. A zero-action boxer falling in ~1 s is the expected behaviour of this class of model,
not a defect in the scene.

### Why the arms looked broken

Every render so far shows the boxers with their fists at chin height, elbows straight and
both wrists cranked about 90 degrees. That pose was not the robot's guard, it was mine. The
`GUARD_ARMS` numbers came from a random search whose only objective was the fist's
*position* in the torso frame, and it shipped an elbow of `-0.007` rad (straight) with wrist
pitch/yaw at `+1.491 / -1.610` on the left hand and `+1.614 / +1.419` on the right -- the
G1's wrist limits are `±1.614`, so two of those four sat flush against their stops.

The reason it could do that is that the arm is a 7-DoF redundant chain and the `left_palm`
*site* sits 8 cm down the forearm from the wrist yaw joint (`pos="0.08 0 0"` in
`left_wrist_yaw_link`). Rotating the wrist swings that 8 cm lever, so the site moves without
the arm moving -- "break the wrist" was the cheapest way to satisfy a target the arm cannot
reach naturally. The target itself was the error: handed `(0.28, ±0.20, 0.10)` m and told to
keep the wrists inside 0.35 rad, the same search cannot get within 8 cm of the requested
height and lands at `z = +0.02` m. Chin height was only ever reachable by breaking the
wrist, so the height was the fit's mistake, not the robot's.

`tools/fit_guard.py` refits it with the two things the first search lacked -- a hard 0.35 rad
cap on the wrist joints and a prior pulling the arm toward the upstream `home` keyframe. The
pose it ships flexes both elbows to `+0.695` rad (~40 degrees), keeps every wrist inside
0.34 rad, and puts the fists at `(0.33, ±0.22, 0.02)` m in the torso frame: forward and in,
at chest rather than chin height. `tests/test_scene.py::test_stance_holds_the_guard` now
asserts that *shape* (elbow above 0.4 rad, wrists below 0.45) and not just the position, and
its old height assertion (`palm[2] > 0.10`) is gone -- the only pose that ever passed it was
the broken one, so the test was pinning the fit's fantasy instead of a guard.

Two things this does not fix, both worth knowing before reading a video:

- The arms will still trail their commanded angles under load, and that is the model's own
  spec, not a bug in the scene. Shoulder, elbow and waist position actuators inherit `kp=75`
  with `actuatorfrcrange=±25 N·m`, so they saturate at 0.33 rad of position error; the wrists
  get `kp=20` with `±5 N·m` and saturate at 0.25 rad. A saturated position controller cannot
  hold a limb during a dive or on contact, so the visual is a slack arm following a target it
  is not strong enough to reach.
- The stance is baked into the observation (the 6 guard-error dims and the 29
  `qpos - stance` dims) and into the action, so playing an old checkpoint in the
  corrected pose does not show the same agent -- it shows a shifted input
  distribution. The cost was measured rather than assumed, 256 arena rounds per cell
  (`tools/stance_ab.sh`):

  | the 204M arm, 256 rounds | trained pose | refitted pose |
  | --- | --- | --- |
  | vs passive | 0.938 | **0.139** |
  | vs jabber | 0.947 | 0.174 |
  | vs the 12.45M arm | 0.754 | 0.051 |

  A skill that does not survive an offset in thirteen observation dims is not
  robustness, it is fitting to the pose. That is why `guard_pose` is now one of the
  things a run records in `env_config.json`, why the arena and the rollout print it
  on the `env:` line, and why an unrecorded run is assumed to have trained in
  `broken-wrist` rather than in whatever the module happens to default to now.
  Everything in this README before this change was measured in the broken-wrist
  stance, and remains reproducible with `--guard-pose broken-wrist`.


## Why this robot model

The brief said to reuse a consolidated model rather than author one. Two
candidate humanoid descriptions were measured before choosing:

- **Brax's own `humanoid` env** (the classic MuJoCo humanoid, ~13 bodies,
  primitive geoms). Fast and battle-tested, but its arms have two shoulder DOF
  and one elbow DOF and no wrist, so a jab and a hook are hard to distinguish.
- **`unitree_g1` from MuJoCo Menagerie**, the BSD-licensed robot library that
  ships with MuJoCo itself. 29 actuated DOF, three wrist DOF per arm, and —
  decisively — upstream already publishes a `g1_mjx` variant tuned for
  JAX-based simulation plus a `scene_mjx` flat-terrain scene with a curated
  explicit contact-pair list and a `home` keyframe.

The G1 was ~5x slower per substep than the classic humanoid but is a modern
research-standard platform with enough arm DOF to box, and it ships with the
MJX tuning this project would otherwise have had to rediscover. It is pulled
from PyPI (`mujoco-menagerie`), never copied into the repo.

## Architecture

```
boxer/scene.py     composes the two-boxer MuJoCo scene with mujoco.MjSpec
boxer/env.py       BoxingEnv: Brax PipelineEnv on the mjx backend
boxer/selfplay.py  multi-agent bookkeeping + the shared-policy and frozen-opponent
                   wrappers
boxer/policy.py    checkpoint loading and the scripted opponents
boxer/arena.py     fixed-matchup evaluation (win rate, punches landed)
boxer/train.py     PPO training entry point
boxer/bench.py     throughput measurement
boxer/rollout.py   renders a trained policy to video / HTML / stills
tools/             WSL2 GPU bootstrap and health check
```

**Scene.** `mujoco.MjSpec.attach` puts two copies of `g1_mjx.xml` into one root
spec under the `red/` and `blue/` name prefixes, on a shared floor, facing each
other. Nothing about the robot is re-authored: joints, actuators, inertias and
meshes all come from the upstream file. Contacts are declared as explicit
`<pair>` elements — the upstream MJX scene's approach — because that both keeps
the candidate-pair count (and so the MJX contact buffer) bounded and makes every
striking contact addressable by geom id. The upstream pair list is harvested
from `scene_mjx.xml` and re-emitted per boxer, then extended with the
cross-robot pairs a fight needs (fist and forearm against head, torso, arms,
body; legs and feet against each other). Pairs are deduplicated by *unordered*
geom pair, since MuJoCo pairs are unordered and declaring both directions
allocates two slots for one collision.

**Environment.** Observations are egocentric: everything is rotated into the
viewer's torso frame, so the two corners genuinely see the same situation and
one network can serve both. `BoxingEnv.step` takes actions of shape `(2, 29)`
and emits `obs (2, D)`, `reward (2,)` and a *scalar* `done`, because the round
ends for both boxers at once — a knockout, a ring-out or drifting out of range
ends the round for both. Reward is a weighted sum of staying upright, holding
the guard, keeping boxing range, facing the opponent, a dense
fist-speed-times-proximity shaping term, and the contact-derived landed-punch
term, which is signed so damage taken cancels damage dealt. A knockout pays the
winner and costs the loser the same amount, so the terminal step is zero-sum.

The default pose is a fitted boxing guard, not the upstream standing keyframe:
arm joint targets were searched against MuJoCo forward kinematics so both fists
sit in front of the chest, elbows flexed and wrists straight (see
`scene.GUARD_ARMS`, and `tools/fit_guard.py` for how they were fitted). The
upstream `home` pose leaves the fists down by the hips.

**Self-play.** Brax's `EpisodeWrapper` requires `reward` and `done` to share a
leading shape, which a duel does not satisfy. `boxer.selfplay` does the episode
bookkeeping at the world level and then folds the agent axis into the batch, so
Brax sees an ordinary single-agent env with twice as many learners as physics
worlds. Its PPO is then used as-is, apart from the one removed-JAX-API shim
described below.

## Training modes

`--mode` selects how the second corner is controlled:

| mode | second corner | what it is for |
| --- | --- | --- |
| `selfplay` (default) | the same network, opposite corner | symmetric MARL. The real objective: anything the policy learns it must also defend against. |
| `frozen` | a fixed policy: `passive`, `jabber`, or a checkpoint | a stationary target. Removes the moving-opponent non-stationarity, so it answers "can it stand and punch at all" separately from "can it fight someone who adapts". |

`boxer.alternating` layers `frozen` into a third mode where **each corner keeps
its own network** — see below.

```bash
# learn against an opponent that only holds its guard
python -m boxer.train --mode frozen --opponent passive --worlds 256

# learn against a metronome jab, which also gives it something to defend
python -m boxer.train --mode frozen --opponent jabber --learner-side blue

# two independent networks: red boxes at blue's checkpoint, then blue at red's
python -m boxer.alternating --iterations 3 --steps-per-iter 2000000 --worlds 256

# measure one against the other
python -m boxer.arena runs/boxer/checkpoint jabber --rounds 64
python -m boxer.arena runs/frozen/checkpoint runs/alt/red/checkpoint
```

`boxer.arena` exists because **self-play reward is not a progress signal**: a
policy improving at boxing can score *less* while its opponent improves too.
Win rate against a frozen opponent does move monotonically, so that is the
measurement to watch. Corners are exchanged and the halves pooled by default,
since a systematic corner advantage would otherwise read as skill.

The modes are not equally hard to add. `frozen` is one wrapper
(`selfplay.FrozenOpponent`) that hands one corner's actions to a fixed policy and
exposes the other as the learner, so Brax sees an ordinary single-agent env with
one learner per physics world. Two networks learning **simultaneously** would
mean forking Brax's PPO update loop, which owns exactly one parameter set.
`boxer.alternating` is the cheaper formulation of the same idea: it trains red's
net against a frozen copy of blue's checkpoint, then blue's against red's
updated one, resuming each corner's own run directory. Each side gets its own
weights, its own observation normaliser and its own value function — what it
does *not* get is a gradient step that sees both learners at once, so a best
response is always computed against a slightly stale opponent. Symmetric
self-play has no such lag; that asymmetry is the thing to watch if the league
ever beats the shared policy.

### What counts as a punch

A landed punch is gated on three conditions, and each one was added because
measurement or a render killed the pair before it. Measured: two boxers holding a
perfectly passive guard drift together from 1.3 m to 0.42 m apart, topple, and
register ~12 "punches" a round from nothing but the collision.

* **fist speed relative to its own torso**, not in world frame -- a collapsing
  body carries both fists along at several m/s, so world speed cannot tell a jab
  from a fall;
* **the striker still being upright** -- torso-relative speed alone still passed
  the collapse case, because flailing arms move fast relative to the torso;
* **the fast fist closing on the head or torso** -- rendered, a trained boxer
  topples forward with its arm over the opponent's chest. That is upright and
  fast in torso-relative terms, so it passed both tests above and scored. Speed
  is measured against the torso; direction is measured against the target, which
  is the only thing that separates throwing from falling onto someone.

Raw contacts remain available as the separate `contacts` metric, so "touched" and
"hit" stay distinguishable in the logs.

**What the gate still does not do is count a punch as one thing.** Hits accumulate per
physics substep, so a boxer holding both hands on the opponent's head -- which is what
the 204M arm actually does, rendered -- scores ~0.7 "hits" per control step and 40+
"head shots" a round. The environment also has no clinch: nothing stops the action when
the two are wrapped around each other, so closing and toppling is worth more under this
reward than punching from outside. Fixing that means counting a punch as an *event*
(contact preceded by non-contact) and ending or penalising sustained contact, neither of
which is a weight adjustment -- which is the useful thing about the three null results
above.

## Quickstart

Native Windows has no CUDA build of jaxlib, so the CPU path is for development
and the GPU path (WSL2) is for training. The code is identical in both.

```bash
# 1. development / tests (CPU, any platform)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python -m boxer.bench

# 2. training on the GPU (WSL2 Ubuntu; no sudo needed)
wsl bash tools/wsl_gpu_setup.sh
# then, from inside WSL, using the venv path it prints:
PYTHONPATH=/mnt/e/boxer-agent JAX_PLATFORMS=cuda,cpu \
  /home/pedro/venvs/boxer313/bin/python -m boxer.train --worlds 256 --total-steps 20000000
```

`--worlds` is the number of parallel physics worlds; PPO sees `2 * --worlds`
learners. Use `JAX_PLATFORMS=cuda,cpu`, not `cuda` alone: Brax's PPO metrics
logger uses `jax.debug.callback`, which needs a CPU device to place its inputs
on, and with CUDA as the only platform training dies at the first epoch. On the
6 GB card stay at 256 worlds; see the VRAM note above.

### Watch a fight

```bash
PYTHONPATH=. .venv/bin/python -m boxer.rollout runs/boxer/checkpoint --format mp4
PYTHONPATH=. .venv/bin/python -m boxer.rollout runs/boxer/checkpoint --format html
```

`--format sheet` writes a single contact-sheet PNG of evenly spaced frames,
which is the quickest way to sanity-check behaviour without a player.

## What a run actually produces

Verified end to end at 5.9M learner steps (~25 min on the 3060): the pipeline
trains, checkpoints, reloads and renders. `runs/rollout/rollout.png` is a
contact sheet of one deterministic round, and it is honest about where the
policy stands — the first frame has both G1s upright, facing each other, fists
at chin height in the fitted guard; a few frames later both are on the floor and
stay there. Training logs at that point show rounds averaging ~50 steps (1 s)
with roughly one boxer down per round, and `hits_landed` and `head_shots`
non-zero, so punches do connect during the stochastic training rollouts even
though the deterministic evaluation round ends in a collapse.

That is the expected shape of this task, not a surprise: 5.9M steps is roughly
one to two orders of magnitude below what humanoid-plus-task RL normally needs,
and Brax's default policy network is 32 units wide -- sized for an ant, not for
58 position-controlled joints. `--policy-layers` raises that, and the long run
under `runs/boxer/` is the attempt to reach standing-and-punching behaviour.

## What is honest here

- **Every arena number published before 2026-10-08 was measured at the wrong action
  scale.** The arena and the rollout constructed `BoxingEnv()` and so used the class
  default `action_scale=1.0`, while every training run used 0.25. Actions are relative
  joint targets, so that multiplied both the commanded excursion and the sampled noise
  by four, and it was enough to invert conclusions: symmetric self-play and the
  best-response league both appeared to lose ~87% of their rounds to a passive opponent
  and actually win ~87% of them. Training logs were unaffected -- they used the env they
  trained in -- so the standing and `down_red` curves in this README were never in
  doubt. Checkpoints now carry `env_config.json` and evaluation rebuilds the env from
  it, printing the scale it used.
- **`--total-steps` does not mean what it looks like.** Brax splits it across the
  eval points with `ceil()`, so a request smaller than
  `num_evals * batch_size * unroll_length * num_minibatches` is ignored outright:
  the 4.19M run asked for one night is a 9.83M run, and two arms of the mode
  comparison only came out equalised because they shared an eval count.
  `boxer.train` now prints the horizon it will actually honour and warns when the
  number typed is not it (`eval_schedule`, pinned against three real runs).
- **Arena numbers have a real noise floor, and 64 rounds per cell sits below it.**
  See the measurement notes under "The two modes, at equal physics": identical
  seeds and checkpoints disagreed by up to 0.11 in win rate at 64 rounds, which
  is how a wrong conclusion got published and withdrawn in the same table. The
  recorded numbers are 256-round runs.
- **Brax 0.14 warns that its own System, pipelines and environments are no
  longer actively maintained** and points at MJX. This project takes that
  advice: physics is MJX, and Brax contributes the env API and the PPO trainer.
  The warning is not theoretical — Brax's PPO calls `jax.device_put_replicated`,
  which the JAX that Brax now resolves against has removed, so `boxer.train`
  installs a five-line replicated-sharding replacement before calling it
  (`_jax_compat_for_brax`). A static scan of `brax/training` against JAX 0.11.2
  finds that as the only missing attribute.
- **Getting the GPU running took three environment fixes, none of them about
  this project**, and all are encoded in `tools/wsl_gpu_setup.sh`: Python 3.10
  can only resolve brax 0.14.1 (which pins jax to 0.6.2); the `cuda12` wheel
  set's `libcusolver.so.11` raises SIGBUS against a CUDA 13.3 WSL driver, so the
  extra has to be `cuda13`; and scipy 1.18's bundled OpenBLAS SIGBUSes inside
  WSL2 from `scipy.linalg.blas`, which takes down `mujoco.mjx` wholesale because
  mjx imports `scipy.linalg` -- including on the CPU path, which makes it easy
  to misdiagnose as a GPU fault. `tools/wsl_gpu_check.sh` reproduces the
  investigation.
- A short run does not produce a boxer. The gauntlet is posture first (staying
  upright on 29 position-controlled joints is the hard part), then footwork,
  then punching. Expect the first tens of millions of steps to be mostly about
  not falling over.
- The `strike` term is dense shaping and *can* be farmed by waving a fast hand
  near the opponent without contacting. The signed contact term is weighted 10x
  higher so that landing dominates once the policy finds it, but this is the
  term to watch if training produces flailing.
- Ring boundary is a termination condition, not a physical rope.
- Renders are only as good as the policy. See the run artefacts under `runs/`.

