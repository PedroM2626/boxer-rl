# boxer — two humanoids learning to fight each other in JAX

Two [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
humanoids spar in a single 3D physics world simulated entirely in JAX. One
shared policy controls both corners, so every skill it discovers it must also
learn to defend against: symmetric self-play.

The environment is a Brax `PipelineEnv` on the **MJX** backend (MuJoCo's JAX
port), trained with Brax's PPO. Collisions are real — a landed punch is a
contact between the fist geom and the opponent's head or torso geom, not a
proximity heuristic.

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

### The two modes, at equal physics

Both arms saw **12.45M world steps** with 256 worlds (`frozen` counts one learner
per world and stopped at 12,451,840 learner steps; `selfplay` counts two and
stopped at 24,903,680 -- the same simulated sparring). 64 arena rounds per cell,
corners exchanged and the halves pooled. `hits`/`heads` are gated punches per
round averaged over **both** boxers, so they say how much boxing happened, not
who did it.

| A vs B | A win | B win | round | hits/round | heads/round |
| --- | --- | --- | --- | --- | --- |
| frozen × passive | **0.641** | 0.328 | 1.10 s | 0.685 | 0.245 |
| frozen × jabber | **0.602** | 0.359 | 1.07 s | 0.971 | 0.238 |
| selfplay × passive | **0.141** | 0.844 | 0.80 s | 0.222 | 0.054 |
| selfplay × jabber | **0.062** | 0.922 | 0.79 s | 0.424 | 0.022 |
| frozen × selfplay | **0.898** | 0.086 | 0.79 s | 0.521 | 0.023 |

So, in order of how much it cost to find out:

- **Symmetric self-play at this budget is worse than not training.** A policy
  that never learns anything (`passive`) beats the 12.45M-world-step self-play
  policy in 84% of rounds. The frozen arm beats that same policy in 90%.
- **The collapse is not caused by the opponent moving.** In the frozen arm the
  opponent is a statue by construction and the learner still fell in 80% of
  rounds at 0.65M steps. It recovered to 45% by 12.45M, which is the one piece of
  good news here.
- **Neither arm boxes well yet.** Rounds last ~1 s and fewer than one gated punch
  lands per round per boxer, counting both corners together. `frozen` wins mostly
  by *not being knocked down*, and no measurement yet proves it punches more than
  its opponent does -- that attribution is the `a_hits`/`b_hits` split, added to
  the arena after these numbers were taken.

Falsified on the way: that the early `kl_mean` of 5.46 signalled an exploding
value function. It is a first-eval transient -- the same run reports 0.005 by
12.45M steps, with no change to `reward_scaling` or the advantage normaliser.


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
sit in front of the chest at chin height (see `scene.GUARD_ARMS`). The upstream
`home` pose leaves the fists down by the hips.

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

A landed punch is gated on two conditions, and both are needed. Measured: two
boxers holding a perfectly passive guard drift together from 1.3 m to 0.42 m
apart, topple, and register ~12 "punches" a round from nothing but the collision.

* **fist speed relative to its own torso**, not in world frame -- a collapsing
  body carries both fists along at several m/s, so world speed cannot tell a jab
  from a fall;
* **the striker still being upright** -- torso-relative speed alone still passed
  the collapse case, because flailing arms move fast relative to the torso.

Raw contacts remain available as the separate `contacts` metric, so "touched" and
"hit" stay distinguishable in the logs.

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

