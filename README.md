# boxer — two humanoids learning to fight each other in JAX

## Current experiment: unassisted paired boxing diagnostics

All four [foundation stages](docs/curriculum.md) passed their unchanged complete
physical gates twice on distinct checkpoints. Foundation learning finished at
**474,992,640 consumed transitions**; keep it stopped. Both paired robots retain
free roots and legs, without pelvis springs, welds or gravity compensation.

The [striking policy](docs/striking-pilot.md) adds opponent-aware arm residuals and
movement-command offsets over that frozen learned motor actor. It starts two
robots nominally 2 m apart and credits independent physical fist impacts. It
has not established stable boxing or superiority to the archived unassisted
policies. At **1,048,576 boxing learner transitions**, final evaluation found
6/8 surviving pairs and three impacts concentrated in an episode ending in a
fall. Main PPO stopped at its reviewed ceiling; no further main steps are granted.

The [completed 256-episode holdout and contact audit](docs/striking-body-holdout-review.md)
found one reproducible ten-second episode with a legal impact from each corner
and both robots still standing. It is rare (one of 64 adaptive sampled trials),
and adaptive body authority regressed survival against control. Neither actor is
adopted. The [actual replay](fights/striking-pilot/stable-two-way-exchange-1179648.mp4)
is evidence of that episode, not reliable boxing.

The [neutral-partner curriculum pilot](docs/striking-partner-curriculum.md) completed
131,072 diagnostic transitions without passing: survival stayed 6/8 and 8/8 by
learner role, with no final learner impacts. The duration continuation stopped
after 262,144 additional steps on two stagnant complete checks. No phase passed.
The [completed exploration audit](docs/striking-exploration-review.md) found no
stable own strikes under any of four noise profiles. The next
[physical feasibility diagnostic](docs/striking-directed-feasibility.md) found
four reproducible ten-second directed-strike trajectories, but poor overall
controller survival. It is not a trained boxer or a phase pass. A sixteen-seed
holdout produced fifteen further trajectories, all exactly replayed. The
[completed supervised initialization](docs/striking-demonstration-initialization.md)
learned isolated stable strikes against a neutral partner, but neither checkpoint
passed and active self-play regressed. The [contact-history comparison](docs/striking-contact-history-comparison.md)
completed 2,000 further supervised updates per arm and all 48 physical episodes.
Both variants failed; contact feedback reached 5/8 self-play survival versus 3/8
for control, without sufficient own scoring in both roles.
No additional PPO was allocated to that imitation comparison. Both bodies/legs remain free and all contacts
active. No actor is adopted and no physical gate is weakened. Inspect
`/home/pedro/boxing-runs/striking-contact-compare-bc2000/comparison.json`
and processes before another launch; that job is completed. The main run remains
stopped. The [completed coverage/channel audits](docs/striking-policy-coverage-audit.md)
reproduced every physical result and localized selected early falls to the arm
residual channel. Their short interventions are diagnostic, not adopted policies.
The [fresh sample review and motor-anchoring comparison](docs/striking-motor-anchoring.md)
validated 32 unique approach prefixes and completed 1,000 supervised updates
per matched arm plus all 120 full physical episodes. The corrected actor has
7/8 active survival on fresh seeds but only one surviving impact; neither arm
passes the complete criteria. No actor or stage is adopted. Further identical
imitation is stopped. The [completed matched PPO comparison](docs/striking-ppo-anchor-review.md)
preserved the latest BC5000 actor and independent actor/critic Adam histories,
but both 131,072-credit arms failed: stable neutral strikes disappeared and
auxiliary anchoring regressed survival. Neither actor is adopted; no additional
PPO is allocated to that failed objective. The completed saved-Gaussian audit
also found no stable-strike recovery. A short gradient assay showed that the small
approach-only loss does not preserve demonstrated strikes. The
[completed demonstration-retention comparison](docs/striking-retention-review.md)
restores learned mean-policy impacts: one fresh ten-second episode has three red
and two blue valid strikes with both robots standing. It is one selected episode,
and neutral-role-1 survival regresses to 4/8; neither actor passes full criteria.
No phase advances or actor is adopted. The selected exchange and matched failure
traces reproduce exactly; the video uses actual recorded coordinates and closed
fists. The [completed follow-up active continuation](docs/striking-neutral-retention-curriculum.md)
also fails both complete screens: stable active contacts persist, while neutral
practice remains unsuccessful. The [completed neutral-partner task](docs/striking-reward-credit-review.md)
also fails both complete screens: final survival is 7/8, 8/8 and 7/8, with no
stable legal events. No actor is adopted or phase promoted. Training receives no
unchanged extension. The [completed reward-credit audit and matched horizon comparison](docs/striking-reward-credit-review.md)
verified all raw physical gates and checkpoints. Gamma 0.999 produces six
surviving active impacts across four fresh seeds, including one deterministic
ten-second two-way episode, versus no fresh impacts at gamma 0.99. Neither
passes: neutral strikes remain absent and some pairs fall. The further
32,768-credit ordinary continuation has now completed: both complete generations
fail. Final neutral-role-1 striking improves, but active survival regresses from
7/8 to 6/8 and surviving active events from five to one on the same fresh seeds.
No more optimizer steps are allocated. All 32 subsequent passive source/final replays
reproduce the original physical results exactly, with finite states and no
numerical/capacity errors. Latest-final active failures are red low-pelvis
terminations after interbody contact; the neutral failure is blue self-leg
contact. A fixed diagnostic now removes command offsets or arm residuals
separately on those selected failed-seed sets (12 episodes / at most 6,000
world controls, zero learning). Inspect
`/home/pedro/boxing-runs/striking-horizon-control-ablation-1736704/ablation.json`
and its process before any launch. Diagnostic PPO is **1,679,616** and formal
diagnostic simulation **828,100 world controls** before this assay; main counters
remain unchanged. Full
[results and the investigation contract](docs/striking-reward-credit-review.md#completed-transfer-traces-and-controller-ablation-2026-10-23)
retain failures and provenance. No phase is promoted or actor adopted.

The [closed-fist replay repair](docs/closed-fist-rendering.md) restores the
original visual feature without changing learning or physics. Historical videos
and all earlier policies remain archived. Project code, documentation and labels
are in English. The original training stacks below are historical references,
not evidence that the current experiment has already learned their skills.

## Archived assisted boxing: runnable sparring reference

The stopped experiment continued the pinned 2.11M seed toward **200M total
learner transitions**. Completed milestones were 10M, 25M, 50M and 100M.
[Protocol and operations](docs/long-training-protocol.md) explain background
execution, recovery and automatic commits/pushes; the
[published experiment report](docs/skills-200m.md) records the interruption and completed milestones.

The `skills` environment adds a hierarchical controller: a shared PPO policy
chooses punches, head/body aim, advance, circle and guard for both G1 boxers.
The motor controller executes complete extension/retraction cycles. Hits require
an actual **fist-to-head or fist-to-torso contact above 50 N**, with approach
speed above 1 m/s. Forearms block but do not score; punches below the torso do
not score. Impacts are checked at every 4 ms physics substep, using the fist's
pre-impact velocity as well as its post-impact velocity.

**This mode has explicit physical balance assistance.** Bounded pelvis springs
keep the robots upright while PPO learns boxing decisions. The roots remain
free and compliant under impacts, but walking and balance are not learned in
this mode. The video labels the assistance. This is a practical sparring
curriculum, not evidence of an unassisted humanoid boxing policy.

The verified local run, `runs/skills2`, completed **2,109,440 learner
transitions**. In 128 deterministic evaluation rounds, the two sides landed
**3.36 / 3.59 clean punches per round**, including **0.53 / 0.48 head shots**,
with no falls. The delivered [12-second video](fights/assisted_ppo_selfplay.mp4)
shows five red and three blue clean hits. The evaluation contains one damage
knockout; the other rounds run out their clock. See the
[saved evaluation](docs/results/assisted_selfplay_eval.json).
The reproducible seed checkpoint is included in
[`checkpoints/skills_seed_2m`](checkpoints/skills_seed_2m/README.md), including
the policy, critic, optimizer and observation statistics.

The compact guard uses the G1's legal negative elbow range and keeps the
wrists within 0.15 rad. Old checkpoints keep their original guards and direct
joint controls; their results below describe a different experiment.

```bash
# Linux / WSL2, Python 3.12. Use a separate environment for Warp.
python3.12 -m venv "$HOME/venvs/warp_test"
"$HOME/venvs/warp_test/bin/pip" install -r requirements-warp.txt

# Reproduce the two-million-transition self-play run.
bash tools/train_skills.sh --output runs/my_sparring

# Render the learned policy in its own environment, without exploration noise.
MUJOCO_GL=egl PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" tools/warp_video.py runs/my_sparring \
  --length 600 --output fights/my_sparring.mp4

# Compare both corners, exchange corners, and save the measured results.
PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" -m boxer.warp_arena \
  runs/my_sparring runs/my_sparring --worlds 64 --steps 600 \
  --output runs/my_sparring/selfplay.json

# A controller-only baseline, clearly labelled as scripted.
MUJOCO_GL=egl PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" tools/warp_video.py --scripted \
  --output fights/assisted_baseline.mp4

PYTHONPATH=. JAX_PLATFORMS=cpu "$HOME/venvs/warp_test/bin/python" \
  -m pytest tests/test_skills.py tests/test_warp_training.py -q
```

`BOXER_PYTHON` selects another interpreter for `train_skills.sh`. Repeat the
same training command with `--resume --total-steps <larger-total>` to continue.
The trainer rejects incompatible environment/network settings before replacing
the recorded configuration, and rejects an existing run unless `--resume` is
given. See [the assisted boxing notes](docs/assisted-boxing.md) for the control
contract and measured results. All code, logs, documentation and video labels
are in English.

## Earlier direct-control experiments

Two [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
humanoids spar in a single 3D physics world. Three ways of
training that duel are implemented — symmetric self-play, a frozen opponent, and a
best-response league with one network per corner — and they are measured against each
other rather than assumed, which is what the Results section below is.

There are two stacks here, and the split is a measurement rather than a preference.
The original is a Brax `PipelineEnv` on the **MJX** backend trained with Brax's PPO
(`boxer/env.py`, `boxer/train.py`). The second steps the same composed scene on
**mujoco_warp** with a numpy environment and a brax-parity PPO learner
(`boxer/warp_env.py`, `boxer/warp_train.py`). It exists because a punch has to be an
impact: MJX exposes no per-contact force anywhere — `mjx.Data` has no force or `efc`
field — so a force-threshold scorecard cannot be written on it, and the proxy that
looked equivalent (how much approach speed the fist loses) pays a vibrating fist for
its own reversal. On the same 6 GB card the warp path runs 2,048 worlds at 4,700-6,500
learner steps/s against MJX's 2,270 at 1,024.

Collisions are real in both: a landed punch is a contact between the fist geom and
the opponent's head or torso geom, not a proximity heuristic, credited by role and
gated on the striker's fist closing on the opponent.

**Where this is at:** the physics, the throughput and the measurement machinery work, and
as of warp7 the two boxers stand, trade punches that land, and sometimes end a fight on
damage -- though "as of" no longer means "best", because the fights below rank warp6 above
it. Measured on its final checkpoint in its own engine: rounds of **364 of 480 control
steps**, **43% of swings converting** (against 6% at the start of the run and 37% for warp6),
2.17 punch-value per world-round, 0.1% of the rollout on the floor, corners 0.515 m apart,
and **5-9% of rounds closing on a knockout** rather than a torso tilting past 45 degrees.

Win rates arrived on 2026-10-21 with `boxer/warp_arena.py`, which scores a matchup in the
physics that produced the arms -- the MJX arena could only describe a warp checkpoint
approximately, and the last warp1 weights showed why: upright boxer here, two collapsed
bodies there. The first numbers are not flattering. **warp6 outsboxes warp7 head to head:
0.538 of rounds won against 0.395**, with 56.9% of their rounds ending on the damage cap,
and the margin is mostly retirements -- 0.402 of warp6's wins put its opponent out on
damage, against 0.166 of warp7's. A second roll, double the sample and a different seed,
repeated it: **0.520 to 0.421** over 1,347 rounds, retirement gap intact (0.396 against
0.190). Against an opponent that does nothing at all, both arms
win only 0.50-0.56 of their rounds and lose 0.39-0.46 by being the one on the floor after
**1.3 s**, a quarter of the 5.1 s they last against a real boxer. So "the newest arm is the
best arm" is not a claim this repository can make, and which arm is best now has to be
settled by fights rather than by training reward.

What is still *not* demonstrated, in order of how much it matters: **no head shots** --
`head_r` peaked at 0.0035 a round and ended at 0.0000 even though a head hit now does twice
the damage of a body hit, so the weight was not the blocker, and in the head-to-head warp7
landed 0.000 head value a round against warp6's 0.003, and no arm pose in a 1,260-point
scripted sweep scores one at all at 0.45, 0.52 or 0.60 m; and the render is close-range
work -- extended body punches mixed with arms bound at chest height, not a jab in and out.
The MJX-era record, measured with a different instrument: the best policy there holds a
guard, closes distance and lands ~0.3 gated punches a round, and wins 64% of rounds against
a passive opponent by being the one still standing, while symmetric self-play at the same
budget loses 88% of its rounds to that passive opponent. Every reward intervention has been
reported as a null result rather than quietly dropped; the diagnosis and the experiments
still open are in [Results](#results).

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
  are holding each other. See "What counts as a punch". Two instruments have since
  replaced this column: the event counter reads the same arm at 19.4, and the
  scorecard reads it at **0.000** -- the numbers in this table are history, not
  measurement.

### The statue was a ragdoll

The table above is measured against `passive`, which means "zero action: hold the
guard and do nothing". Two of those, 256 rounds, corners swapped:

| `passive` × `passive` | broken-wrist guard | refitted guard |
| --- | --- | --- |
| someone goes down | 0.90 | **0.87** |
| round length | 1.14 s | **1.11 s** |
| punches landed | 0.000 | 0.000 |

Nobody throws anything and 87% of rounds still end with a boxer on the floor, in
about a second. That is the expected physics -- a position controller with no
balancing policy is not holding a humanoid up -- but the consequence for every
number in this README is not: **a win against `passive` has been partly a race to
stay standing**, and the 0.938 of the 204M arm is not 94% proof of boxing. It is
also why the frozen arm could not have answered the question the event reward was
built for: the target leaves the floor's reach in 1.1 s, before a punch has to be
anything.

The fix is `--anchor-opponent`, which pins the frozen corner's root to where it
spawned -- it can still be hit, its limbs still move, but it cannot fall or be
pushed over. With the opponent anchored, landing a punch is the only way to score,
which is the condition the second long run trains under. `tests/test_env.py::
test_an_anchored_corner_cannot_be_knocked_over` holds the two halves of that: the
pinned corner is upright for a full 6 s round, and the unpinned one still goes
down, so it is an anchor and not a global buff to balance.

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


### The warp arms (2026-10-17 onward)

The migration paid for itself before any behaviour appeared: 2,048 worlds at
**4,700-6,500 learner steps/s** on the same card, against MJX's 2,270 at 1,024, so
200M learner steps costs ~10 h instead of ~25 h, and per-contact force exists. What
the four arms produced is below, each with the hypothesis it was testing and the
measurement that killed or kept it.

| arm | what it changed | measured outcome |
| --- | --- | --- |
| warp1 | first arm on the warp env: force scorecard, imitation channel from the retargeted CMU jab, per-agent down gate | 200.2M, and it **froze**: std 0.70→0.22, entropy +31→−5, hits only in the iteration that opens a round, imitation term exactly 0.00000 for all 611 iterations. The warp-native renderer explained what the log could not: both boxers on the floor within ~1 s, one sitting propped on an arm — and the down test (`upright_z < 0.15`, head `< 0.55`) called that alive, so falling collected the dense terms all round. |
| warp2 | down test moved to 45° / head 0.85 m; **every** dense term gated on both corners being up; spawn 0.75 m; range narrowed (σ 0.15) and cheapened (w 0.15); std floor 0.3 | Killed at 137.6M by a bug of my own. Reward flatlined at −0.005 (the energy residue), `value_loss` fell to 0.0003, entropy 47→8, std pinned at its floor — a dead gradient, not a converged policy. `tools/warp_probe_policy.py` measured it in the engine: **87% of the round on the floor, both corners up for 11.8% of it.** Gating a boxer's own posture term on the opponent being upright deletes the incentive to stand back up. |
| warp3 | `upright` pays on the boxer's own posture again (as `env.py:673` does); interaction terms keep the both-up gate; the log gained a floor fraction | 200.2M complete, and it **falsified "the gate was the problem"**: standing early in the round improved (both up over the first 80 steps 0.59→0.73) but the round as a whole still ended at down 0.87 / both-up 0.12, hits never left the ~180-per-round-opening spawn transient, and the late-round state was a torso at `up_z` 0.69 collecting **+0.55/step** for doing nothing. Three of every four learner transitions came from a fight that was already over. |
| warp4 | ported the rest of `env.py`'s round semantics: a round closes on a knockdown, out of the ring, or beyond `MAX_SEPARATION` (2.4 m), with the respawn happening **per world inside `step()`**, plus `w_alive` 0.2 and the zero-sum `w_ko` 3.0 | **200.2M complete (10.94 h), the first arm that boxes.** Round length 50→78 control steps; **punch value per world-round 0.09→0.70** (means 0.157 over the first 100 iterations, 0.677 over the last 80 -- monotone over the whole run, not a round-opening spike); reward +1.06→+1.26; floor 0.007. Probed deterministically on the final checkpoint: a round every 90 steps, 0.72 punch-value thrown and 0.73 taken -- so the mean action punches, not the noise -- and every closure by falling (ring 0.00, separation 0.00). Ran with the termination-bootstrap bias below, so its curve is a floor. |
| warp5 | same arm, with the termination bootstrap masked (`cont = 1 - done * (1 - timeout)`) -- falling stopped paying the value of the next round | **200.2M complete (10.67 h), and the answer to the question warp4 left open.** Round length 51 → 94 → 201 → 382 control steps against a 480 cap; **punch value per world-round 0.08 → 3.17** (last 60 iterations average 3.09); floor fraction 0.010 → 0.001; the corners closed the gap from 0.68 m to **0.52 m**, inside the band the arm can score in; and `end_fell` fell from 1.00 to 0.58 -- **42% of rounds now end because the clock ran out, which has never happened in any arm before**. Probed deterministically on the final checkpoint: a round every 476 steps, 1.56 punch-value thrown and 1.36 taken per round, nobody ever on the floor. Two honest caveats: punch *rate* per step is lower than warp4's (0.0033 vs 0.0082 -- warp4's rate was inflated by the collapse scrum of its 90-step rounds), and the rendered behaviour is a sustained close-range arm-to-arm engagement rather than a jab in and out. |
| warp6 | a punch becomes a discrete, costed action. An **attempt** -- the rising edge of the fist closing on the opponent at `punch_speed_min`, computed from palm velocity along the line between torsos so it exists without contact, with hysteresis at half the threshold so one swing is one attempt -- costs `w_attempt` 0.5 against the +10 hit; and re-arming now requires the fist to have **returned to its guard**, within 0.25 m of its own torso-frame target, not merely 20 steps without contact. Plus the metrics that make the claim checkable: hits split by surface, attempts per round, accuracy, and the fraction of steps the fist is home. | Running. Chosen by measurement rather than taste: warp5 sits at 0.83 torso-equivalent contacts/s against the 2.5/s ceiling its own rearm window allows, with an imitation term of 0.00155, so the mill is the cheapest legal policy under the old objective -- and RoboStriker's metric is a *ratio* whose denominator this env never had. From a random policy at launch: `head 0.00/0.00`, `acc 0.07`, `home 0.07`, reward +0.98. **200.2M complete (9.94 h). **The gates bought accuracy, not restraint.** Accuracy -- the fraction of swings that land -- rose **0.062 → 0.370**, and punch value per world-round **0.09 → 3.65** (warp5: 3.17), with the policy's scale ending at 1.39 where warp5 reached 1.58. But swing *volume* went up, not down: 3.6 attempts per round at iteration 1 is 2.36/s, it dipped to 1.13/s mid-run, and it finished at **20.0 per round = 2.79/s** -- more flailing per second than the random policy started with. The arithmetic says why: a swing that lands 37% of the time at a mean value of 0.49 returns 10 × 0.49 × 0.37 = +1.81 against a charge of 0.5, so swinging is individually profitable and the price is too small to matter. Probed deterministically: 4.22 value thrown and 4.14 taken per round of 427 steps, floor 0.001, 0.54 m apart. **And the head number, stated precisely: `head_r` is 0.0025 per round averaged over the last 60 iterations, 0.0067 at its best, against 6.4 torso contacts -- roughly one head shot per 400 rounds.** The gates did their job (discrete, converting punches) and what they exposed is that nothing in this reward makes a punch *count for anything beyond its own moment*: the hit is paid and un-paid symmetrically, so nothing compounds. |
| warp7 | the fight gets a state: clean hits accumulate as **damage** at their surface weight (head 1.0, torso 0.5, body 0.35), a corner at `damage_cap` 3.0 is **out** -- which closes the round and pays the survivor the zero-sum knockout -- and every clean hit **stuns** its victim out of scoring for 10 control steps. Own damage, opponent damage and stunned join the observation (128 → 131) | **200.2M complete (10.04 h). It changed the shape of the fight, not the style.** The knockout fires: `end_ko` sits at 0.05-0.09, so 5-9% of rounds now end on damage instead of a tilted torso (with the first cap of 5.0 it would have slept through the whole run). The mill halved: 2.37 swings/s at the start, **1.39 at the end**, while accuracy rose 0.057 → **0.433**. Rounds 76 → 364 of 480 steps, floor 0.001, distance 0.61 → 0.515 m. Two things it did not do: punch value per round came out BELOW warp6 (2.17 against 3.65 -- fewer swings, and the stun removes scoring windows, so this metric no longer compares across regimes), and **the head still never lands**: `head_r` peaked at 0.0035 a round at iteration 340 and was 0.0000 at the end, so doubling the damage weight did not buy the +0.32 m of elevation. The policy's scale also ended at 1.75, the highest of any completed arm against a 2.0 ceiling. **Scored in fights since (2026-10-21, `boxer/warp_arena.py`): it loses the head-to-head to warp6, 0.395 of rounds won against 0.538**, and against an opponent that does nothing it is the one on the floor in 46% of its rounds, at 1.3 s a round -- a quarter of the 5.1 s it lasts against a boxer. |

Two things the truncation forced into the open:

- **The knockout needs the truncation to mean anything.** `w_ko` is zero-sum, so
  paid per step across 320 floor steps both corners down cancel to zero and the
  channel rewards a mutual collapse over a knockout — the same leak `env.py` records
  for a boxer diving at the opponent to collect it. The self-check now rolls each
  corner's torso 90° in the sensor buffer and pins the sign, and forces a
  termination to prove the respawn happens inside `step()`.
- **A bootstrap is not free at a round boundary.** `warp4` shipped with the loss
  adding `gamma * V(next)` at every round end, which is correct for a timeout and
  wrong for a termination: behind a knockdown the next observation is a fresh
  standing round, so the terminal step was credited roughly 26 discounted reward
  *for falling*. brax gets a zero terminal value for free from its pipeline; here it
  has to be masked, which is why the env now returns `timed_out` as a metric and the
  loss uses `cont = 1 - done * (1 - timeout)`.

### The measurement that closed the door

Two arms (the fourth and fifth MJX ones) ended with **0.000 punches to the head** and
nobody could say whether the fists never arrived or arrived and were refused. The
falsifier read each gate factor off the state the env returns: 0.00% of 8,000 sampled
steps had raw contact, the rearm counter never once reset, and 72% of the steps were
spent glove-on-glove at 0.72 m. So it was arrival, not the counter.

`tools/reach_sweep.py` then measured the geometry instead of arguing about it: 8,194
uniform arm poses per arm and per separation, pushed through the env's own contact
buffer with blue holding the guard and with blue's arms at zero.

| separation | poses that score, guard | poses that score, arms down |
| --- | --- | --- |
| 0.90, 0.75, 0.65 m | 0.0% | 0.0% |
| 0.55 m | 0.2% | 0.2% |
| 0.45 m | 1.3% | 1.3% |

Three conclusions, all of which changed the code: the guard seals nothing (arms-down
scores the same, so the opponent was never the wall); the arm is ~0.11 m short of the
torso at the 0.75 m fighting distance the reward was trained on; and the only band
where a punch can land statically was the band the clinch penalty punished, while
`in_range` paid its peak at 0.70 m. That is why the warp env's bands are
`target_range 0.55 / punch_reach 0.45 / clinch_range 0.35` -- the paid optimum has to
sit inside the reachable set -- and why a punch became a force-threshold event rather
than a contact margin.

### The measurement that reopened the head question

The same instrument, seven arms later, on the policy that finally punches: `tools/warp_head_gap.py`
rolls a checkpoint out and tracks the fist's distance to each target surface against the
geometry that decides contact -- a fist capsule of radius 0.035 touching the head sphere of
radius 0.06 means a centre distance of 0.095 m. On warp6's final checkpoint (128 worlds,
deterministic, 122,880 corner-steps):

| | fist-to-head | fist-to-torso |
| --- | --- | --- |
| median distance | 0.361 m | 0.206 m |
| closest ever | 0.092 m | 0.024 m |
| corner-steps inside touching range | 1 | 1,241 (1.0%) |
| scoring events landed there | 0 of 2,004 | 1,968 |

and the head sits **+0.319 m** above the fist at the median (the torso, −0.036 m). So the
answer is **AIM, not REACH**: the elevation exists -- the fist got inside touching range,
once, in 122,880 chances -- and nothing has ever bought it. That is why the head's damage
weight matters now (warp7): a head shot retires a boxer in three hits where the body takes
six, which is the first time in this project that aiming high is worth more than hitting
often. What this measurement deliberately did *not* do is move `target_range`: the sweep
above already says the paid band is at the edge of what the arm can reach, and changing two
things at once would have made warp7 unreadable.

**Corrected on 2026-10-21, by the experiment this paragraph asked for.** "AIM" was too
generous a reading of one corner-step in 122,880: `tools/warp_head_sweep.py` drives every
arm pose in a 1,260-point grid at the head on purpose, closing at up to 11.0 m/s against a
1.0 m/s gate, with the palm site inside the head's touching distance at 0.52 m -- and scores
**nothing** on the head at any distance, while the same sweep lands 129-184 torso hits. The
elevation the arm can buy by itself is not enough to hit the head; see item 4 of "What is
still open" for what that leaves unanswered.

### The first league table (2026-10-21)

`boxer/warp_arena.py` scores a matchup on WarpBoxing itself: whoever is out on the step a
round closes decided it, corners exchanged and pooled, 256 worlds x 480 steps per roll and
two rolls per pairing (~4 min on the 3060). Every fraction below is over the rounds that
closed. **strong** is a win against a loser that had absorbed at least 1.0 of damage when it
went down -- a head shot, or two torso shots -- which is above what a body leaning on an arm
gives for free (0.35). **long** is the win rate inside the rounds that lasted at least 150
control steps (3 s), and `long_share` says what fraction of the rounds those were: where
`long_share` is near zero, the long columns are a handful of rounds and mean nothing.

| A vs B | A wins | B wins | A strong | B strong | `long_share` | A in long | B in long | round length | `end_ko` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warp6 vs warp4 | 0.238 | **0.740** | 0.013 | **0.109** | 0.003 | 0.250 | 0.750 | 1.35 s | 0.000 |
| warp6 vs warp5 | 0.297 | **0.676** | 0.124 | **0.384** | 0.561 | 0.173 | 0.788 | 3.71 s | 0.126 |
| warp5 vs warp4 | 0.114 | **0.871** | 0.011 | **0.078** | 0.006 | 0.855 | 0.145 | 1.34 s | 0.000 |
| warp7 vs warp6 | 0.384 | **0.544** | 0.325 | **0.467** | 0.820 | 0.366 | 0.551 | 5.12 s | 0.585 |
| warp7 vs warp5 | 0.282 | **0.553** | 0.189 | **0.334** | 0.729 | 0.234 | 0.555 | 5.39 s | 0.156 |
| warp7 vs passive | 0.498 | 0.450 | 0.053 | 0.000 | 0.002 | 0.417 | 0.583 | 1.30 s | 0.000 |

**The order is the reverse of the chronology, and it is transitive**: warp4 beats warp5 and
warp6, warp5 beats warp6 and warp7, warp6 beats warp7. Four changes that each moved a
training metric in the right direction each lost the previous arm's fight. **The two stricter
readings do not rescue the newest arms** -- the strong-win column gives the same order, and by
a wider margin than the win rate does (warp5 outscores warp6 3.1x on strong wins against 2.3x
on wins). What the columns change is what a win *means*, and that is a property of the pair.

**`long_share` is the most informative number in the table and it belongs to no arm.** warp4
drags every pairing it is in down to 1.35 s -- 0.3% of its rounds with warp6 last past 3 s,
0.6% with warp5 -- so its 0.109 strong wins are damage concentrated inside a shove, and its
0.855-in-long against warp5 is roughly twenty rounds, not a result. warp6 and warp7 are the
only pair that fights: 82% of their rounds pass 3 s, 58.5% end on the damage cap, and there
the strong column says what the win column says -- **warp6 boxes better than warp7** (0.467
against 0.325). So the honest summary is not "the arm that deals most damage" versus "the arm
that wins most": the arms that win are the ones that end the fight before damage can decide
it, and the one that boxes best is warp6, which is only visible in the single pairing where a
fight happens at all.

The single most uncomfortable number: **warp5 flails 34.1 attempts a round at 0.049 accuracy
and still beats warp7 in 55% of rounds, while warp7 lands twice the punch value (1.62 against
0.84)** -- landing more, and far more accurately, loses, because the loss is a fall and not a
score. And the scene's old attractor sits underneath all of it: the arena's own self-check
roll -- two passive boxers, nobody attacking -- has one of them down inside 200 steps on its
own, so "somebody is on the floor at 1.3 s" is partly the stance rather than the fight. See
"The falls are not about the opponent" above.

What the two stricter readings settle is the decision, not the ranking. The ranking is robust
-- warp4 > warp5 > warp6 > warp7 on wins, on strong wins, and on the win rate inside long
rounds wherever enough long rounds exist to say so. But that means **a rule which scored only
damage-borne wins would still crown warp4**, so "make the scoreboard measure boxing" does not
select for boxing on this scene. The thing that would has to make rounds last, and that is the
stability problem, not the metric.

### Traps in the warp path, one run each

- **`put_data` sizes the constraint buffers from the host `MjData` you hand it, which
  is empty before any forward pass.** In overflow the solver drops constraints
  **silently**, and a dropped contact reads 0 N -- indistinguishable from a punch that
  never landed. The first scene check measured force ratios of 0.64-1.56 against the
  host until `nconmax`/`njmax` were sized by hand (384/2,048 per world; both are
  per-world, not global). `opt.warn_overflow` is zeroed in the env, so the only
  standing defence is the self-check's scripted jab, which must peak above 0.05.
- **A rising hit count is not evidence until the policy's scale is inside its bounds.**
  The first warp1 run took std 0.71 → 27.4 and entropy +33 → 110 while its "hits" rose
  fivefold: a tanh-saturated action is a coin flip, and a coin flip that swings a 29
  joint arm into a body scores on the force threshold. The scale is now
  `clip(softplus(raw) + 0.001, 0.3, 2.0)` -- the floor because a smaller scale cannot
  move a 29-joint arm off its mean, the ceiling because past ~2 the mean stops
  mattering. The completed warp1 then froze the other way (std 0.22, entropy −4.9).
- **Arena numbers do not cross the two stacks unchanged.** The guard-error block of the
  observation is computed from site frames in the MJX env and from sensor reads in the
  warp env: the same quantity measured along two paths, near-identical but not bit
  identical. warp1 proved it was not a pedantic distinction -- the same weights read as
  an upright punching boxer in warp and as two collapsed bodies in the MJX env, which is
  why `boxer/warp_arena.py` scores the warp arms in warp and the MJX numbers on them are
  simply gone.
- **A metrics dict that holds a view of live state reports the next round.** `step()`
  builds the metrics and *then* respawns the worlds that closed a round, and `_randomize`
  zeroes `self._damage` in place. `damage=` handed the caller the live array, so every
  round that ended on the damage cap read 0.00 damage and the first matchup reported
  `a_win_ko 0.000` while `end_ko` in the same table said 63% of rounds ended on the cap.
  The env's own health test could not see it because it calls `_reward` directly, so the
  self-check now asserts the reported array is not the live one.
- **A re-run of the same matchup is not the same number.** warp7 vs warp6, same seed, same
  code, four separate runs: `a_win` 0.384-0.395, `b_win` 0.538-0.560, 672-687 rounds. Warp's
  contact reductions are not bit-reproducible on the GPU and two bodies in contact amplify the
  difference, so the arena's own run-to-run spread is about one point of win rate. Gaps below
  ~3 points are noise; the 14-16 point gaps between arms, and the 3x gaps on strong wins, clear
  it by an order of magnitude.
- **The motion reference has four ways to lie quietly.** CMU's Vicon-dialect ASF/AMC
  pins elbow deflection at 90° whatever the subject did, which is why the pipeline
  reads BVH; BVH channel values must be sliced in pre-order DFS or the fists come out
  at 132-185 m/s; the leading T-pose frame wins any hand-speed threshold, so rest
  frames are trimmed before a window is chosen; and a session contains many punches,
  so the strike window is a contiguous fast run measured on the *outward* path, not
  the fastest single frame.

- **A round-end flag is not a reason.** The first version of this logged `ko_ends`,
  the fraction of closed rounds where the truncation flag was set -- which was 1.00 by
  construction, because the flag *is* the union of "someone fell", "someone left the
  ring" and "the corners drifted apart". It let me tell you 100% of rounds ended on a
  knockdown before anything could distinguish them. The env now returns the three
  reasons separately (`end_fell`, `end_ring`, `end_far`), and the claim was re-measured:
  falls 1.00, ring 0.00, separation 0.00.
- **An instrument that reads after the respawn cannot see the fall it is measuring.**
  With per-world autoreset, `step()` puts the fallen world back on its feet before
  returning, so recomputing the down flags from the post-step sensors reported a floor
  fraction of 0.001 where the truth was 0.006 -- a round every 87 steps implies ~0.011,
  and the fallen state survives exactly one step by design. The probe now reads the
  env's own pre-respawn `on_floor`, and the posture columns are labelled as
  post-respawn because they are biased toward whoever is standing.

Caveat that applies to every video here: renders use the policy's deterministic mode
(`tanh(loc)`), while training samples its actions, so a clip understates hit counts
relative to the log.


### What is still open

The intro promises this list, so here it is -- each item with the measurement that would
close it.

1. **~~A run without the termination-bootstrap bias.~~ Answered by warp5.** With the mask,
   round length went 78 → 382 control steps and punch value per round 0.70 → 3.17 on the
   same 200M budget. So the wall was the objective, not the balance physics: the boxers
   could stand and fight the whole time and were being paid ~26 discounted reward to stop.
   The question that replaces it is whether 382 of 480 is where this reward plateaus, and
   what the remaining 42% of fall-endings are made of.
2. **~~No win rate exists for any warp arm.~~ Answered on 2026-10-21 by `boxer/warp_arena.py`**,
   which scores a matchup on WarpBoxing itself: the corner out when a round closes decided it,
   rounds segmented per world because `step()` respawns inside the call, corners exchanged and
   pooled because red and blue are not interchangeable experience. The first head-to-head put
   warp7 at **0.395 of rounds won against warp6's 0.538** (676 closed rounds), and a second
   roll at double the sample and another seed put it at 0.421 against 0.520 (1,347 rounds).
   The books balance the way they have to: `a_win_ko + b_win_ko` = 0.568 against `end_ko`
   0.569, and the two fall columns 0.366 against `end_fell` 0.375. The league table that
   followed is above, and the ordering it gives is the reverse of the chronology on every
   criterion the arena now reports -- wins, strong wins, and wins inside rounds long enough to
   be called fights. So what is open is no longer which metric to read but why the newest arms
   lose a fight they win on punch quality: they are the ones on the floor at 1.3 s, and a rule
   that scored only damage-borne wins would still crown the scrum (that was checked, not
   assumed). warp1-warp3 have still never been scored -- warp1 trained at a 1.3 m spawn and the
   arena refuses to mix geometries on purpose.
3. **~~Nobody has ever run out a clock.~~ Answered by warp5.** `end_fell` fell from 1.00 to
   0.58 over its own run: 42% of rounds now end because the 480 steps ran out. In the
   deterministic probe a round lasts 476 of 480 steps and only 12% of closures are falls.
4. **~~Head versus torso: measured, priced, and still zero.~~ The scripted sweep ran on
   2026-10-21 (`tools/warp_head_sweep.py`), and the head is not an aim problem.** 1,260 poses
   of red's four left-arm joints, swept over their model ranges, one pose per world, driven
   from the stance in 0.16 s -- 877 to 926 of them closing faster than the 1.0 m/s the
   scorecard asks for, up to **11.0 m/s** -- at 0.45, 0.52 and 0.60 m, with blue's arms down
   so nothing blocks the line. At every distance: **zero head contacts over the force
   threshold, armed or raw** -- the two channels agree exactly, so the rearm gate is not
   hiding them -- while the same sweep scores 129-184 torso hits at the expected 0.50, which
   is the control that says the instrument works. The palm site does reach 0.088 m from the
   head site at 0.52 m, inside the 0.095 m touching distance, and still never puts 50 N on
   the head geom.
   What it does *not* settle is the rest of the body. The sweep moves the arm only, and live
   fights do land head shots, rarely -- warp6 at 0.0025 a round, 0.003 head value a round
   against warp7 in the league table -- so the head is reachable with the torso and the step
   in it, which no arm-only grid supplies. The question is now narrower than "aim or reach":
   do the rare live head hits coincide with a torso lean, and would sweeping the waist and
   knees make one scorable on purpose. Until that is answered, the head's 2x damage weight is
   paying for something this arm cannot deliver by itself.
5. **The imitation channel is still a null** (0.0007-0.0023 across three arms). Either it
   fires in a run that actually punches, or it comes out.
6. **The policy's scale is creeping, and warp7 is the worst of it.** warp4 ended at 1.63
   against a 2.0 ceiling, warp5 at 1.58, warp6 at 1.39 -- and warp7, the calmest fight of
   the four, ended at **1.75** with entropy 56. The ceiling is what stops this becoming
   warp1's 27.4, but a run that presses it has its hit counts back in tanh-saturated noise.
   Nobody has yet explained why the scale grows while the reward is flat: it is the one
   trend in these arms that none of the five changes has touched.
7. **The mill: priced by warp6, halved by warp7.** The falsifier had two branches and warp6
   resolved neither -- accuracy rose 0.062 → 0.370 while the swing rate went *up* to 2.79/s,
   because a swing returning 0.37 × 4.9 = +1.81 against a 0.5 charge is always profitable.
   warp7's consequence did what the price could not: the rate fell to **1.39 swings/s** and
   accuracy reached 0.433, with 5-9% of rounds finishing on a knockout. The cost is
   explicit: punch value per round dropped to 2.17 from warp6's 3.65, and value per *second*
   to 0.006 from 0.010 -- the stun removes scoring windows on both sides, so the fight is
   calmer, not bigger. That makes "punch value per round" a metric that cannot compare the
   two regimes, and it is stated here rather than quietly redefined. What calmer and more
   accurate buys has now been measured, and the answer is not what the log suggested: against
   warp6 in `boxer/warp_arena.py`, **warp7 wins 0.395 of rounds to warp6's 0.538** (0.421 to
   0.520 on a second seed at double the sample), and the
   gap is in conversions to retirements -- 0.402 of warp6's wins put its opponent out on
   damage against 0.166 of warp7's, even though warp7 is the arm that introduced the damage
   cap. Being harder to hit did not survive being hit.


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

### Where the motion reference comes from

The imitation channel tracks a recorded human jab, retargeted onto the G1's arm
joints by `tools/retarget_dense.py`. The recording is the **CMU Graphics Lab
Motion Capture Database** — its FAQ states the data "may be copied, modified, or
redistributed without permission" (`tools/cmu_fetch.sh` pulls it; `LICENSES.md`
carries the wording and the acknowledgement CMU asks for). The earlier reference
was a Mixamo clip, kept locally for the measurement it produced: Adobe's General
Terms section 3.6 allow using and modifying their Content Files inside a product
you author but forbid distributing them stand-alone, so a Mixamo clip or any
curve dump derived from it cannot live in this repo. Reproducing the pipeline
means running the fetch script and the two converters — no motion file is
committed, and none is needed from anyone else.

### The hands are grafted, not swapped

The video that prompted the scorecard also showed open hands: `g1_mjx` has one rigid
rubber-hand mesh where a hand should be, and at any camera distance it reads as a flat
palm. The same Menagerie folder ships `g1_with_hands`, the articulated Dex3-1 hand, and
swapping to it is the obvious fix. It is also not available: MJX's collision driver has
no cylinder-versus-mesh function and the hand variant's colliders are exactly that pair,
so `mjx.put_model` raises `NotImplementedError` before a single step runs
(`tools/hand_cost.py`). Making it simulable would mean authoring new colliders for the
robot -- the one thing this project's brief rules out -- on top of paying its 2 ms step
and 100 solver iterations against this scene's 4 ms and 5.

So the fist is grafted instead. `_graft_fists` takes the articulated hand's own meshes --
the palm plus seven finger links -- at a closed pose and attaches them as *visual* geoms
to the wrist yaw link of the tuned robot, in place of the rubber hand. The pose is
written as joint angles of the articulated model and read back through its forward
kinematics, mirrored in sign for the right hand because that hand's joint ranges are the
negation of the left's. `tests/test_scene.py` pins the result geometrically -- distal
links 56-61 mm from the palm against 94-111 mm with the fingers straight -- because a
wrong sign leaves a hand open and nothing else in the build would notice.

The property that makes this safe is that the graft is visual *only*. The grafted geoms
carry the same signature the rubber hand had -- contype 0, conaffinity 0, no pair -- so
`compiler.discardvisual` drops them from the visual-free build the trainer compiles
exactly as it drops the hand they replace. Measured: that build is unchanged in
everything the policies touch (55 geoms, 208 pairs, nq 72, nu 58, 66.6823 kg), so every
checkpoint trained before the graft still means what it meant, and only the render gains
fingers.

One trap worth recording for whoever touches this next: MuJoCo's compiler rewrites the
pose of a mesh geom as it loads the asset -- the palm's declared offset of 0.0415 m reads
back as 0.0838 m with a 122 degree turn -- so copying transforms out of a *compiled*
model applies that rewrite twice and the hand arrives splayed at twice the distance. The
graft therefore takes body transforms from kinematics and geom offsets from the spec,
which is declared coordinates on both sides.

## Architecture

```
boxer/scene.py     composes the two-boxer MuJoCo scene with mujoco.MjSpec
boxer/env.py       BoxingEnv: Brax PipelineEnv on the mjx backend
boxer/warp_env.py  WarpBoxing: the same scene on mujoco_warp, numpy, with the
                   force-threshold scorecard and per-world round truncation
boxer/selfplay.py  multi-agent bookkeeping + the shared-policy and frozen-opponent
                   wrappers
boxer/policy.py    checkpoint loading (brax and warp checkpoints) and the scripted
                   opponents
boxer/arena.py     fixed-matchup evaluation (win rate, punches landed)
boxer/warp_arena.py  the same evaluation on the warp env, which is where the warp
                   arms actually live
boxer/train.py     PPO training entry point, MJX/brax
boxer/warp_train.py  PPO training entry point, warp GPU physics + jax CPU learner
boxer/bench.py     throughput measurement
boxer/rollout.py   renders a trained policy to video / HTML / stills
tools/warp_video.py  renders a warp checkpoint from the warp sim itself
tools/warp_probe_policy.py  what a checkpoint actually does, measured in its engine
tools/reach_sweep.py, tools/warp_hit_directions.py, tools/fist_probe.py,
tools/warp_head_sweep.py
                   the measurements the design decisions rest on
tools/             WSL2 GPU bootstrap, dependency scripts and health checks
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

**The warp trainer has none of these modes.** `boxer.warp_train` is self-play only:
`--mode`, the frozen-opponent wrappers and the alternating league live in the MJX path
and were not ported. That is the reason no warp arm has a win rate attached to it, and
it is the biggest thing missing from the newer stack.

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

**Two gates live in this repo, and the newer one is the definition.** The bullets below
are the MJX env's gate -- margin and torso-relative speed -- because MJX cannot read
force at all. The warp env scores with the metric the published recipe uses: a contact
of at least **50 N** on a weighted target surface (head 1.0, torso 0.5,
pelvis/hip/thigh 0.35, arms 0), credited **by role** -- the geom carrying a fist or
forearm surface of one corner, landing on a scoring surface of the other -- and gated on
that striker's palm closing on the opponent at >= 1.0 m/s. Since warp6 two more
conditions apply, both added because warp5 showed the scorecard alone rewards a mill:
the swing is **charged** as an attempt whether or not it lands (0.5 against the +10 hit),
which is the denominator RoboStriker's hit-rate ratio has, and the fist must have
**returned to its guard** since it last touched the opponent before its next contact can
score. Attribution by role is not
decoration: the first warp arm credited hits by the order the two geoms happen to appear
in the contact buffer, red's geoms sort first, so a chest pressed against a chest paid
red 0.50 and a full blue jab paid blue nothing. `tools/warp_hit_directions.py` is the
three-case truth table that killed it.

The MJX-era gates, each added because a measurement or a render killed the pair before
it. Measured: two boxers holding a
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

Contacts remain available as the separate `contacts` metric, so "touched" and "hit"
stay distinguishable -- as *presence* of contact per step, not a substep-summed count,
which is what made the ratio between them meaningless the first time around.

**Counting a punch as an event turned out to be three bugs, not one.** The rising edge
landed, and the numbers still made no sense, so each layer got measured separately:

- **an event paid ~17 punches.** `raw_landed` sums how many contact slots crossed the
  margin across the five substeps, so one landed punch scored once per geom that
  touched. The trace showed only 3.2% of control steps carried a counted punch --
  1.6/s, a boxing rate -- while the metric reported 28/s. Hits and head shots are
  indicators now; damage still scales with how much connected, because a punch that
  catches shoulder and chest did more work than one that grazes a knuckle.
- **the retraction gate was decoration.** It measured the fist-to-target distance to
  the torso *site*, which sits at the body's centre, so a fist resting on the chest was
  already 0.16 m "retracted" and 99.3% of steps counted as armed. What replaces it is
  the invariant the trace actually showed violated: every counted step had contact
  present. A punch comes from outside, so `rearm_steps` consecutive steps off the
  opponent are required, and contact itself spends the window. That alone takes 22.4
  hits a round down to 7.85.
- **an impact threshold cannot exist against a welded target.** Scoring on contact force
  is what the closest published work does --
  [RoboStriker](https://arxiv.org/abs/2601.22517) (Yin, Cao et al., 2026), which
  simulates the same robot, "Unitree G1 robots with 29 degrees of freedom", and scores
  with "the proportion of contacts exceeding the force threshold to the total number of
  offensive attempts" -- but MJX exposes no force on its contact struct, and the proxy
  that seemed equivalent (how much approach speed the fist loses) pays a vibrating fist
  for its own reversal. It was removed rather than tuned. RoboStriker publishes a
  [project page](https://yinkangning0124.github.io/RoboStriker/) and no code repository,
  so every mechanism quoted from it here is this repo's own implementation of the
  described metric, checked against the sentences above rather than run from their code.

**What the 200M arm in this instrument actually is.** 256 rounds, full 10 s length,
against an opponent that cannot fall and throws nothing: **19.4 punches, 1.96 a second,
and 0.000 to the head.** No knockdowns either way. The head is not out of reach -- the
fist can be raised to +0.55 m above the torso against the head's +0.43 m -- but only
with the shoulder near vertical, and the head bonus does not pay for lifting an arm that
far while staying upright and re-cocking. So "never aims high" is a reward-shaping result
here, not a kinematic one, and it is the clearest thing this run says about what to
change next.

**And rendered, it is not boxing.** Contact sheets of the finished arm against the
pinned opponent, from a front camera and with the round run to its full length
(`boxer.rollout <ckpt> --opponent passive --anchor blue`): the learner walks in, closes
to chest contact, drops its own guard, and flaps both hands around the opponent's torso.
The hands-together guard visible in those frames is the statue's, not the learner's. So
the 1.96/s is real contact events with a fast-enough, aimed-enough fist, and it is
produced by behaviour nobody would call punching. Two readings remain open, and the next
run is the one that separates them: the gate is still too loose (1 m/s of torso-relative
speed and a 0.16 s re-cock window are both generous), or 200M steps against an opponent
that cannot retaliate converges on pressing-and-flapping whatever the counter says. This
arm was trained under the 17x-inflated counter, so retraining it under the corrected one
is the test -- which is what `tools/long_run_event.sh` now runs.

### What watching the video changed in the reward

Rendered and watched back, the finished third arm does three things that are not boxing:
it walks to one spot and stays there, it flaps a hand up from the guard at roughly 6 Hz --
which reads as a greeting rather than a jab -- and both boxers spend the round slapping
each other's hands. Each of those is a number in the reward, and each number moved:

* **the arms are worth nothing.** A glove was worth 0.25, which made hand-slapping the
  cheapest scoring contact in the scene. Arms stay collision geoms -- blocking has to be
  a real physical event -- but they now score 0.0, and a zero-weight target is masked out
  of the *hit count* as well as out of the damage, so the metric and the reward cannot
  disagree about what happened in a round.
* **the scorecard is ordered**: head 1.0 > torso 0.5 > legs 0.35 > arms 0.0, which is what
  a boxing scorecard does and what the render said was missing.
* **the re-cock window is a jab's, not a vibration's**: `rearm_steps` 8 -> 20, i.e. 0.4 s
  off the opponent at 50 Hz. 8 steps (0.16 s) was short enough to permit the 6 Hz flap.

Presence and score are two different quantities, and each has exactly one job. Presence -- any
striker inside the margin of any target, weighted or not -- drives the rising edge and the
`contacts` metric: a punch has to arrive from off the opponent's body, and `hits_per_contact`
needs a denominator that a glove touch still counts against. Score drives the damage, the hit
count, and the rearm window.

The window is the part that took two attempts, and both are worth keeping because the first
sounds stricter and is not usable. Spending it on *any* touch means that at fighting distance
-- 0.9 m apart, where the corners are in contact on well over half the control steps and the
longest clean run is shorter than the window -- it never reopens: 0 scored steps in 200 for a
jabber throwing 1.6 times a second (`tools/fist_probe.py`). Spending it on a *scored* contact
says what the window was always for, "0.4 s since this boxer last put a fist on something that
counts", and the same jabber in the same round scores again. A glove resting on a glove does not
silence a boxer; a fist vibrating on a chest does, which is the flicker the window exists to
stop.

**What the new scorecard says about the third arm.** Re-measured in it, the finished arm
lands **0.000 punches across 256 full-length rounds** while still making 126 contacts a
round -- the same 126 it made when the old counter reported 31.3 punches in 10 s. Nothing
about the policy or the physics moved between those two numbers; only what a contact is
allowed to be worth did. The video was right and the counter was wrong: every counted
"punch" of the third arm was a hand on a hand. The window is not what zeroed it -- 8 and
20 steps both read 0.000 -- which isolates the arms-at-zero as the whole of the difference,
and makes the fourth arm a clean test of whether a boxer can be taught to aim at something
the scorecard buys.

**And what training under that scorecard did.** The fourth arm was trained with the glove
worth nothing, the head worth most and a 0.4 s window. It reached 161M steps and was
stopped by decision once 50M of them had produced nothing new. In the arena it lands
**0.000 punches across 256 full-length rounds** and makes **403 contacts a round** -- three
times the third arm's, with the same zero head shots. Rendered, it walks in and presses
both gloves against the statue's gloves for the whole round: no wind-up, no retraction, no
punch shape. What it did learn is the thing that was actually being paid for -- it fell in
100% of episodes early in training and in 1% by the end, and rounds now run the full 10 s.

So the scorecard was right and still produced no boxing, and that is what pointed at the
next bug -- same family as the others. The hit channel pays `damage`, the **depth** of a
contact: `clip(-dist / 1 cm)` times the target weight, summed over the slots that crossed
the margin. A fist that touches without sinking a centimetre prices at a fraction, and this
arm's measured `damage` was **0.013 per episode against a round return of 829**, while
merely standing upright pays **1.659 every step**. Grinding gloves was the correct reading
of those numbers. The clean event indicator the arena reports (`landed_hits`) already
existed in the code and was never what the agent was paid on. The fifth arm changes that one
thing: the hit channel pays the event (`w_hit` 3.0 -> 10.0 on `landed_hits`) with depth kept
as a small extra (`w_damage` 1.0), which puts a landed punch at about six steps of standing.
`tests/test_env.py` asserts that ratio, so the channel cannot quietly sink back under the
noise floor of existing.

**Then there is the shape of the motion, which was two bugs wearing one costume.** The
scripted "jabber" that every punch fixture here leaned on was a sine on shoulder pitch
alone. That rotates the whole arm about the shoulder, so the fist travels an arc that is
mostly *up*: the metronome was throwing a greeting and arriving on the opponent's gloves,
which is also why a scripted jab scored nothing once gloves stopped being worth anything.
It is now an elbow extension with the upper arm brought round at the same time, which is
what sends the fist along the axis the boxer faces.

And the gate gained a fourth condition for the same reason. Speed, aim and uprightness
together still pass a hand *lifted* from the chest into the head -- and on this robot that
is the path of least resistance to the head, since the head sits 0.43 m above the torso
centre while the straight-wristed guard tops out at +0.05 m. A fist's velocity now has to
lie along the facing direction (full credit within ~25 degrees), and the same factor is
applied to the dense shaping, because paying for a lift near the head while refusing to
count it as a punch is exactly how the fourth arm learned to wave. The cost is deliberate
and worth stating: an uppercut cannot score here, because this robot cannot throw one
without first raising the arm through the same arc as the wave.

**A confident claim that turned out to be false.** While making the arms score zero the
glove test would not pass, and reading the contact buffer seemed to show that MJX's
slot -> pair mapping depends on the state: 284 slots for 208 declared pairs, every
head-involving slot same-body or floor, and no cross-body fist-on-head slot anywhere. That
reading was wrong and the error was mine. `contact.geom` indexes the geoms of the model
the *env* compiled, which is built without visuals -- 55 geoms -- while the names were
being resolved through a fresh `scene.build_model()`, which carries 125. Every id named a
different geom, so a declared pair looked undeclared. `tools/slot_audit.py` reads the
buffer through the env's own model and reports what is actually there: the mapping is
byte-identical at reset, after a step, and with the corners teleported into each other;
the eight fist-on-head slots exist; and the 76 extra slots are pairs that fall into two
condim groups, none of them weighted, so damage cannot be summed twice. The optimisation
that folds per-geom roles into per-slot constants stands, and so does the test that pins
it. What does not stand is the generalisation from one misread diagnostic.

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

### Training on mujoco_warp

The warp path needs its own venv (mujoco 3.16 + mujoco-warp + warp-lang, and jax
pinned to CPU so the learner never competes with warp for the card):

```bash
bash tools/warp_deps.sh                      # creates /home/pedro/venvs/warp_test
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_env_bench.py  # steps/s
cd /mnt/e/boxer-agent && PYTHONPATH=. JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/pedro/venvs/warp_test/bin/python -u -m boxer.warp_train \
    --worlds 2048 --imitation runs/reference/jab_targets_cmu.json --output runs/warp5
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_env   # self-check
```

`python -m boxer.warp_env` is the test suite for this env: it checks sensor parity
against the host, the scripted jab's force threshold, both arms' imitation peaks, the
knockout's sign in each direction, and that a terminated round respawns inside `step()`.
Leave `MUJOCO_GL` unset in this venv — `egl` errors and `osmesa` has no `libGL` here.

### Watch a fight

```bash
PYTHONPATH=. .venv/bin/python -m boxer.rollout runs/boxer/checkpoint --format mp4
PYTHONPATH=. .venv/bin/python -m boxer.rollout runs/boxer/checkpoint --format html
# a warp checkpoint, rendered from the warp sim itself (no MJX round-trip):
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_video.py runs/warp5 \
  --length 300 --output runs/warp5_fight.mp4
# and what that checkpoint actually does, as numbers from its own engine:
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_probe_policy.py runs/warp5
# and the one number a reward log cannot give: a win rate, measured in the engine
# the arms were trained in. Two rolls (corners swapped and pooled) of --worlds
# parallel rounds; 256 worlds x 480 steps takes about 4 minutes on the 3060.
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_arena \
  runs/warp7 runs/warp6 --worlds 256
PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python -m boxer.warp_arena --self-check
```

`--format sheet` writes a single contact-sheet PNG of evenly spaced frames,
which is the quickest way to sanity-check behaviour without a player.
`warp_probe_policy.py` prints the fraction of the rollout each corner spends on the
floor and the fraction with both standing — the two numbers a reward log cannot
substitute for, and the reason two arms were misread as converged.

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

A **warp** run writes four things, and `runs/` is gitignored because the motion
reference and the checkpoints both live under it:

- `warp_config.json` -- the settings the run was actually built with: worlds,
  `obs_size`/`action_size`, round cap, spawn distance and action scale, the
  imitation file and its weight, the layer widths, and the whole PPO block. A run
  is reproducible from this file plus the commit, not from the CLI history.
- `checkpoint/step-<N>.pkl` and `checkpoint/latest.txt` -- a plain pickle of numpy
  leaves (`params_p, params_v, opt_state, running_norm, step`). Flax's msgpack
  cannot serialise the tuple or optax 0.2.8's state, so this is deliberately dumb.
  `boxer.policy.load_warp_policy` reads them, and `make_opponent` dispatches to it
  on the presence of `warp_config.json`, so the arena and both renderers take a
  warp checkpoint without being told.
- `progress.jsonl` -- one record per iteration: reward per corner, hits per corner,
  distance, imitation, `carry`, `down_red`/`down_blue`/`both_up`, `round_length`,
  `end_fell`/`end_ring`/`end_far`, `punch_value_red`/`punch_value_blue`, and from warp6
  the punch structure itself: `head_r`/`torso_r`/`body_r` per world-round (and the
  other corner's `head_b`/`torso_b`/`body_b`), `attempts_r`/`attempts_b`, `accuracy_r`,
  `at_home` -- plus `clip_frac`,
  `entropy`, `std`, `value_loss`, learner steps/s. Two things a parser has to survive:
  the schema grew while arms were running (warp4's early records have no posture or
  round fields, and its late ones still say `ko_ends`, the flag that could not tell a
  fall from a walk-out), because a live process runs the code it started with; and
  `hits_red`/`hits_blue` are weighted values, not counts.
- `<arm>.log` -- the same numbers as the line-per-iteration stream the trainer
  prints, which is what gets tailed while a run is live.

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
- **A reward bug shipped in `warp4` and was found by reading the loss, not by
  watching the run.** Once rounds truncate, `delta = r + gamma * V(next) - V` at a
  round end credits the terminal step with the value of the respawned round --
  roughly 26 discounted reward for falling -- because the env returns a fresh
  observation where brax's pipeline would have returned a zero terminal value. The
  arithmetic was visible before the effect was: `round_length` was crawling (44 to
  63 steps over 228 iterations) while the survival incentive said it should climb
  faster. Fixed by masking the bootstrap on termination
  (`cont = 1 - done * (1 - timeout)`), which is why the env now returns
  `timed_out`. `warp4`'s own curve is therefore measured under a bias toward
  finishing rounds on the floor, and is quoted as such.
- **Two arms were read as converged because the log could not see posture.** warp1
  and warp2 both plateaued at a reward that looked like a policy, and the number
  that would have said "they are lying on the floor for 87% of the round" was not
  recorded. `down_red`, `down_blue`, `both_up`, `round_length` and `ko_ends` are in
  the log now, and `tools/warp_probe_policy.py` exists so a checkpoint's behaviour
  can be measured in its own engine instead of inferred from its reward.
- **The imitation channel has never done anything.** It measured 0.00000 for all 611
  iterations of warp1 at `sigma = 0.3` rad, and 0.0007-0.0023 at the widened
  `sigma = 0.6` used since. It is kept because it is cheap and because a shaped term
  that never fires is evidence about the arm's reachable set, not because any result
  in this README depends on it.
- A short run does not produce a boxer. The gauntlet is posture first (staying
  upright on 29 position-controlled joints is the hard part), then footwork,
  then punching. Expect the first tens of millions of steps to be mostly about
  not falling over.
- The `strike` term is dense shaping and *can* be farmed by waving a fast hand
  near the opponent without contacting. The signed contact term is weighted 10x
  higher so that landing dominates once the policy finds it, but this is the
  term to watch if training produces flailing.
- Ring boundary is a termination condition, not a physical rope.
- **The whole test suite does not finish on this machine, and that is a limit of the
  instrument, not of the code.** Run as one job it kills the WSL VM at 20-25 minutes
  every time (four reboots in one night; `uptime` reads 0 min and the log stops
  mid-progress with no summary), while a trainer ran 20.6 h uninterrupted on the same
  card. So the suite is run in halves and each half is reported separately:
  `tests/test_env.py` on its own, and the remaining files as a second job. A green
  "73 passed" line has been produced this way repeatedly but never in a single pass,
  and anything that gates a launch on a single-pass suite would never launch here.
  `tools/verify_then_train.sh` captures pytest's own exit status for the halves it can
  run; the trainer's wait loop checks for a free card, which says nothing about tests,
  so it must not be treated as the gate.
- Renders are only as good as the policy. See the run artefacts under `runs/`.

## License

The code here is MIT (`LICENSE`). That grant deliberately does not reach the third-party
assets the project consumes, and the difference is not cosmetic:

- the Unitree G1 description is installed from PyPI as `mujoco-menagerie`
  (Apache-2.0/BSD upstream) and is never vendored into this repo;
- the motion reference is CMU Graphics Lab MoCap data, pulled by `tools/cmu_fetch.sh`
  under its own "may be copied, modified, or redistributed without permission" terms,
  with the acknowledgement CMU asks for;
- the Mixamo clip that preceded it **cannot** be redistributed, so neither it nor any
  curve dump derived from it is committed -- reproducing that measurement means running
  the fetch and the two converters, which is why `runs/reference/` is gitignored.

`LICENSES.md` carries each of those wordings. Everything generated under this license --
checkpoints, videos, the `runs/` tree -- is yours to do with as you like.

