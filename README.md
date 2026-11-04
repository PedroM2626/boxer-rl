# boxer — two humanoids learning to fight each other in JAX

![Striker (red fists) boxing Champion (blue fists) in native MuJoCo](fights/striker_vs_champion_preview.gif)

**Striker (red) vs Champion (blue).** A 12-second native MuJoCo bout with
learned controllers, free legs and no pelvis support. Both fists match each
boxer's corner color. [Watch the preview video](fights/striker_vs_champion_preview.mp4)
or the [full 60-second bout](fights/striker_native_legs_vs_champion_60s.mp4).

## Where things stand (2026-11-03)

Two simulated Unitree G1 humanoids box each other in native MuJoCo with no
physical assistance: free legs and roots, no pelvis springs, welds or
gravity compensation. Every policy is learned except the scripted punch
paths of the champion stack and a scripted sparring brain. A punch scores
only as a physical impact of a fist on the opponent's head or torso (native
impact scorecard: impulse, closing speed and fist speed). Bouts of 120 s run
inside a 4 m roped ring under knockdown rules: a fall is counted, both robots
are reset standing (they cannot get up by themselves), and a third knockdown
is a TKO.

Two complete boxers exist, each a stack of learned parts:

| Stack | Legs | Brain and arms | Style |
| --- | --- | --- | --- |
| **Striker** (strongest) | `checkpoints/spar-native` | `checkpoints/striker`: one policy for footwork and both arms' joint targets, no scripted moves | in-fighting, short hook-like punches |
| **Champion** (reference) | `checkpoints/spar` | `checkpoints/tactics` (footwork, punch choice, guard height) + scripted straights and hooks (`boxer/moves.py`) bent by `checkpoints/arm-skill` | longer straights, high guard |

Each on its own legs, over 32 bouts of 120 s (16 seeds, corners swapped), the
striker beats the champion **32–0**, landing **256 clean punches per bout to
87**, with **1 knockdown to 4**
([details](docs/striker.md#legs-trained-in-two-robot-bouts-2026-11-03);
[60 s video at 60 fps](fights/striker_native_legs_vs_champion_60s.mp4)).

```bash
bash tools/run_spar.sh boxer.spar_match --ring --seconds 120 --knockdowns 3 --seeds 0 1 2 3 --swap \
  --checkpoint checkpoints/spar-native --red-tactics checkpoints/striker \
  --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill \
  --video runs/fight.mp4
```

**Limits, measured:**

* **Falls in striker brawls.** Two strikers fighting chest to chest fall
  about 20 times in 32 bouts of 120 s; more leg training did not reduce it
  ([spar motor](docs/spar-motor.md#legs-trained-in-two-robot-bouts-2026-11-03)).
* **Close to a clinch.** The striker wins by fighting at very close range;
  nothing in the rules separates clinched boxers.
* **No getting up.** After a knockdown the referee resets both robots standing.
* **Parts are trained separately.** Legs, tactics, arm skill and striker are
  separate policies trained in turn; changing one (e.g. a striker's arm style)
  can break another (its legs), as several experiments below show.
* **Turning is automatic.** Brains choose advance and circling; facing the
  opponent is a fixed rule (`tactics_env.decode`).
* **Simulation only**, against a fixed pool of policies; no sim-to-real work.
* **Tests:** all 83 test files (409 tests) pass when run one file per
  process (`tools/test_all.sh`, 2026-11-03); the whole suite in one process
  exhausts the WSL VM's memory.

**Checkpoints.** Current: `spar`, `spar-native`, `tactics`, `arm-skill`,
`striker`. Earlier, kept for comparison: `striker-v1` (first striker),
`spar-v2` (rapid-punching legs), `spar-v1`, `spar-shorthook`,
`spar-straight` (earlier spar motors), `tactics-arm-v1`, `tactics-ring-v1`,
`tactics-highguard-v1`, `tactics-levelhook-v1`, `tactics-guard-v1`,
`tactics-native-v1`, `tactics-batched` (earlier tactics champions; play each
on the legs it was trained on, see [tactics](docs/tactics.md)),
`curriculum` (certified balance/walk/approach foundation), `skills_seed_2m`
and `skills_200m` (archived assisted boxing), `striking-pilot` and
`striking-validation` (archived paired striking pilot).

## Progress timeline

Newest last. Every row links to the report with protocols and full results;
[experiment history](docs/experiment-history.md) has one row per experiment
since 2026-10-30, including the rejected ones.

| Dates | Phase | What was tried and measured | Outcome |
| --- | --- | --- | --- |
| 10-06 – 10-08 | MJX boxing environment | Two G1s in one MJX scene, Brax PPO; symmetric self-play, frozen opponent and league compared ([results](docs/archive/mjx-warp-experiments.md#results)). Arena numbers before 10-08 used the wrong action scale; corrected and two conclusions withdrawn. | Robots mostly fall; best policy wins 64 % against a passive opponent by staying up. |
| 10-09 – 10-16 | Scoring and form | A punch counted as one event with reload and forward travel; closed-fist visuals; guard refit. | Reward changes did not stop the falls ([results](docs/archive/mjx-warp-experiments.md#reweighting-the-reward-changed-nothing)). |
| 10-17 – 10-21 | `mujoco_warp` arms warp1–warp7 | Per-contact force scorecard, termination-bootstrap fix, costed punches; first win rates in `warp_arena`. | Rounds of 364–382 of 480 steps; warp6 beats warp7 0.538–0.395; no head shots. |
| 10-21 | Assisted boxing (archived) | Hierarchical skills policy with pelvis springs holding the robots up. | 3.4 clean punches per round; not unassisted ([notes](docs/assisted-boxing.md)). |
| 10-21 – 10-22 | Unassisted foundation | Curriculum balance → walk_easy → walk → approach, each passing its physical gate twice. | Certified, 475M transitions ([curriculum](docs/curriculum.md)). |
| 10-22 – 10-29 | Paired striking pilot (archived) | Arm residuals over the frozen approach motor, PPO v12–v17, imitation, many audits, a batched CUDA backend. | Isolated real strikes, no reliable reciprocal boxing; robots fell when arms punched ([pilot](docs/striking-pilot.md), [history](docs/experiment-history.md#decisions-and-evidence)). |
| 10-30 | Spar motor | Diagnosis: the leg motor survived 6 % of solo trials while its arms punched. Legs retrained under generated punches and shoves. | 100 % solo survival; first unassisted two-robot bouts: 8/8 go 20 s, 18.6 clean punches ([spar motor](docs/spar-motor.md)). |
| 10-30 – 10-31 | Learned tactics | Batched self-play PPO, then fine-tuning on native MuJoCo workers. | Batched self-play degrades natively; native fine-tuning wins 42–2–4 ([tactics](docs/tactics.md)). |
| 10-31 | Moves, ring, rules | Hooks and high guard, level IK hook, scoring audit, held-high-guard training, long rounds, knockdown rules, roped ring, faster native stepping. | Each new champion beats the last (27–17–4, 37–9–2, 27–16–5 over 48 bouts) ([moves](docs/moves.md), [tactics](docs/tactics.md)). |
| 10-31 | Arm skill | Learned per-step correction (≤ 0.35 rad) of each scripted punch; tactics retrained with it. | 39–4–5 with identical tactics; landing 41 → 50 % ([arm skill](docs/arm-skill.md)). |
| 11-01 | Knockdowns | Census: falls follow shin contact. Motor with phantom shins; leg-contact penalty for tactics. | Both rejected: knockdowns did not drop ([spar motor](docs/spar-motor.md#phantom-shins-and-real-falls-2026-11-01)). |
| 11-02 | Rapid-punching legs | Motor trained with near back-to-back punches; tactics retrained on it. Arm skill retrained on it (rejected: more falls). | New stack 22–9–1, knockdowns 2 to 13 ([spar motor](docs/spar-motor.md#legs-for-rapid-punching-2026-11-02)). |
| 11-03 | Striker | One policy for footwork and both arms: cloned from the champion, then native PPO. | 16–0 in 120 s bouts, 64 % hook-like and 21 % new punch shapes, but 14 knockdowns to 1 ([striker](docs/striker.md)). |
| 11-03 | Legs for the striker's arms | Motor replays recorded striker arm actions in half its episodes. | Striker knockdowns 14 → 1; promoted as `spar` ([details](docs/spar-motor.md#legs-for-the-strikers-arms-2026-11-03)). |
| 11-03 | Arm-motion cost | Striker pays per radian of arm motion; legs retrained solo on its recording. | Per clean punch it already moved as much as the champion; with the cost it fell 29, then 18 times; rejected ([striker](docs/striker.md#arm-motion-cost-2026-11-03)). |
| 11-03 | Legs trained in two-robot bouts | Native PPO for the motor in two-robot ring bouts (`boxer.motor_native`), opponents from strikers, champion and scripted brain. | Low-motion striker knockdowns 29 → 2; promoted (`spar-native`, `striker`). Champion loses 4–28 on these legs, keeps its own. Striker-heavy continuation: no gain ([details](docs/spar-motor.md#legs-trained-in-two-robot-bouts-2026-11-03)). |

## Milestones since unassisted two-robot bouts (2026-10-30 onward)

The paragraphs below were written as each result came in; checkpoint names
are given as they are now.

**Unassisted two-robot bouts (2026-10-30).** The paired falls had one cause.
The frozen leg controller had been trained with its arms locked in guard.
While its arms punched, it survived only about 6 % of solo 10 s trials. The
new [`spar` stage](docs/spar-motor.md) trains the legs and waist while a
generator throws randomized punches with the arms and the body gets shoved.
After 10.8M fine-tuning steps (≈ 8 min), solo survival reaches 99.4 %. At
39.3M steps (≈ 30 min) it reaches 100 % and passes the full gate. In
two-robot native MuJoCo bouts with scripted tactics, **8/8 bouts go the full
20 s with no falls**, averaging 18.6 clean punches per bout between both
corners ([video](fights/spar_g1_vs_g1.mp4)).

**Learned tactics (2026-10-30/31).** [Self-play PPO](docs/tactics.md) chooses
footwork, punches and guard height. It is pretrained in a batched GPU
two-robot environment, then **fine-tuned on native MuJoCo rollouts** from 8
parallel CPU workers. Batched self-play beyond the first policy kept getting
worse in native physics; native fine-tuning improves steadily (first native
champion, now `checkpoints/tactics-native-v1`: 42–2–4 against the batched one).

**Hooks and a high guard (2026-10-31).** [New moves](docs/moves.md) are
fitted by forward kinematics and shared by every backend. The motor was
retrained to balance through them. Native fine-tuning taught the champion to
keep a high guard a third to half of the time. After diagnostics showed the
first hook could not land, the hook became a level IK arc with straight-like
reach. A [scoring audit](docs/moves.md#scoring-audit-toolsscore_auditpy)
showed the scorecard is fair to hooks; they rarely reach the target. Fine-tuning
against scripted opponents that hold the high guard produced a new
champion (now `checkpoints/tactics-highguard-v1`): **27–17–4 over 48 native bouts** against
the level-hook champion, and 113 clean punches to 16 in 8 bouts against a held
high guard (the previous champion: 43 to 9). It breaks the guard with
straights and has stopped throwing hooks
([video](fights/tactics_guard_vs_native_v1.mp4) shows an earlier champion).

**Roped ring (2026-10-31).** Bouts can take place inside a 4 m roped ring
scaled to the G1 (`--ring`). The ring champion (`checkpoints/tactics-ring-v1`)
was fine-tuned inside it and sees where the ring centre is; it beats the
previous champion 27–16–5 over 48 native bouts and 14–2 in 120 s ring bouts
([details](docs/tactics.md#roped-ring-2026-10-31)). Native bouts now replay
identically on GPU and CPU hosts, and native fine-tuning runs 2.1× faster.

**Learned punches (2026-10-31).** An [arm skill](docs/arm-skill.md) adds
learned, real-time corrections to each punch, aimed at where the target
actually is (`--red-arm checkpoints/arm-skill`). With identical tactics on
both sides it wins 39–4–5 over 48 native bouts and 16–0 in 120 s ring
bouts; punches land about 50 % of the time instead of 41 %. Tactics
fine-tuned with the arm skill attached (now `checkpoints/tactics-arm-v1`)
beat the ring champion 35–11–2, though more knockdown-prone over long bouts
([details](docs/arm-skill.md#tactics-trained-with-the-arm-skill)).

**Long rounds and knockdowns (2026-10-31).** Bouts of 120–180 s hold up after
a footwork fix that keeps the boxers facing each other. `--knockdowns 3` adds
a referee: a fall is counted and both robots restart standing (they cannot
get up unaided), and a third knockdown is a TKO
([details](docs/tactics.md#knockdown-rules-2026-10-31);
[3-minute video](fights/tactics_knockdown_rules_180s.mp4) with two knockdowns,
the high-guard champion in blue).

**Knockdowns diagnosed (2026-11-01).** Falls follow the robots' shins
tangling; neither legs trained against phantom opponent shins nor tactics
paying for leg contact reduced real knockdowns
([details](docs/spar-motor.md#phantom-shins-and-real-falls-2026-11-01)).

**Legs for rapid punching (2026-11-02).** Learned tactics punch about twice a
second, twice as often as the leg motor was trained for. The leg motor
(now `checkpoints/spar-v2`) was trained with near back-to-back punches and
arm-skill-sized corrections (`curriculum_train --rapid-punches`), and the
champion tactics (`checkpoints/tactics`) were fine-tuned on top of it. Over 32
bouts of 120 s against the previous stack, each robot on its own legs, the
new stack wins **22–9–1 and suffers 2 knockdowns to 13**
([details](docs/spar-motor.md#legs-for-rapid-punching-2026-11-02);
[2-minute video](fights/rapid_champion_120s.mp4): the new champion in blue
wins 126–106 on points, red is knocked down once). Tactics older than these
play on their own motor: `--checkpoint checkpoints/spar-v1`.

**One policy for footwork and both arms (2026-11-03).** The
[striker](docs/striker.md) (now `checkpoints/striker-v1`) replaces tactics,
scripted moves and arm skill with a single policy that sets both arms' joint
targets every step: no move list, no fixed paths. Imitating the champion
stack first and then trained with native PPO, it beats that stack **28–4**
in 20 s bouts and **16–0** in 120 s bouts (269 clean punches to 72). It
learned to fight at close range with short punches, mostly hook-like and a
fifth on paths no scripted move takes, but on the legs of the time it fell
far more (14 knockdowns in 16 long bouts to 1)
([30 s video](fights/striker_vs_champion_30s.mp4)).

**Legs for the striker's arms (2026-11-03).** The leg motor
(`checkpoints/spar`, previous `checkpoints/spar-v2`) also trains replaying
arm actions recorded from the striker's own bouts (`striker_train arms`,
`curriculum_train --striker-arms`). On it the unchanged first striker beats
the champion **32–0** over 32 bouts of 120 s, 309 clean punches to 141, with
**1 knockdown to 0** instead of 14 to 1; the champion stack falls less on it
too and plays on it since
([details](docs/spar-motor.md#legs-for-the-strikers-arms-2026-11-03);
[60 s video at 60 fps](fights/striker_vs_champion_new_legs_60s.mp4);
`spar_match` videos are 60 fps by default now, `--fps`).

**Arm-motion cost (2026-11-03).** `spar_match` reports how far each corner's
arm targets move. Per clean punch the first striker moves its arms as much as
the champion (21.4 against 21.7 rad). A striker fine-tuned to pay for arm
motion (`native_finetune --arm-motion-cost`) moved 11 % less but fell 29
times in 32 long bouts on legs trained under the old arm style, and 18 times
on legs retrained solo on its own recording
([details](docs/striker.md#arm-motion-cost-2026-11-03)).

**Legs trained in two-robot bouts (2026-11-03).** The motor also learns in
native two-robot ring bouts (`boxer.motor_native`), where pushing and
leaning at close range happen. On those legs (`checkpoints/spar-native`) the
striker that pays for arm motion (now `checkpoints/striker`) beats the
champion stack on its own legs **32–0** over 32 bouts of 120 s, 256 clean
punches to 87, with **1 knockdown to 4**. The champion loses 4–28 to itself
on these legs and keeps `checkpoints/spar`. Training on more striker brawls
did not reduce their falls
([details](docs/striker.md#legs-trained-in-two-robot-bouts-2026-11-03);
[60 s video](fights/striker_native_legs_vs_champion_60s.mp4)).

## Archive

Earlier phases, moved out of this README on 2026-11-03 and kept as written:

* [Paired striking pilot log](docs/archive/striking-pilot-log.md) (2026-10-22 to 10-29)
* [Assisted boxing reference](docs/archive/assisted-boxing-reference.md) (2026-10-21)
* [MJX and warp direct-control experiments](docs/archive/mjx-warp-experiments.md)
  (2026-10-06 to 10-21): results, architecture, training modes, quickstart and
  the honesty notes of those stacks

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
boxer/curriculum_env.py    UnassistedG1: one G1 on MJX/Warp, the curriculum stages and the
                           spar stage (generated, rapid or replayed striker arm motions)
boxer/curriculum_train.py  GPU PPO for the leg motor (balance ... spar), with physical gates
boxer/curriculum_eval.py   the solo physical gate (survival under shoves and punching arms)
boxer/paired_foundation.py two robots in one native MuJoCo model, ring, impact scorecard, falls
boxer/moves.py             scripted straights, hooks and guards, fitted by forward kinematics
boxer/tactics_env.py       tactics observation and decode (footwork, punch, aim, guard)
boxer/tactics_train.py     batched GPU self-play for tactics
boxer/native_finetune.py   PPO on native two-robot bouts in CPU workers (tactics or striker)
boxer/tactics_gate.py      native selection and confirmation gates
boxer/arm_skill.py, boxer/arm_finetune.py   learned corrections of scripted punches
boxer/striker.py           one policy for footwork and both arms' joint targets
boxer/striker_train.py     striker imitation data, cloning, and arm recordings for the legs
boxer/motor_native.py      PPO for the leg motor in native two-robot bouts
boxer/spar_match.py        native bouts between any two stacks, knockdown rules, video
tools/punch_shapes.py      which punch shapes a corner throws
tools/knockdown_causes.py  what happens in the second before each knockdown
tools/score_audit.py       every fist-on-target contact and why it did or did not score
tools/run_spar.sh          runs any of these with the curriculum runtime on WSL
```

Each stack runs at 50 Hz: a brain (tactics or striker) sets the footwork
command and the arms; the leg motor turns the command into leg and waist
targets while keeping balance; the arm targets replace the motor's own arm
outputs. Details: [spar motor](docs/spar-motor.md), [tactics](docs/tactics.md),
[moves](docs/moves.md), [arm skill](docs/arm-skill.md), [striker](docs/striker.md).
The original MJX and warp stacks are described in the
[archive](docs/archive/mjx-warp-experiments.md#architecture-of-the-mjx-and-warp-stacks).

## Quickstart

Linux or WSL2 with an NVIDIA GPU; one runtime, `$HOME/venvs/boxing_curriculum`
([setup](docs/curriculum.md#operation-and-published-artifacts)):

```bash
bash tools/setup_curriculum.sh
# watch the striker against the champion, each on its own legs
bash tools/run_spar.sh boxer.spar_match --ring --seconds 60 --knockdowns 3 --seeds 1 \
  --checkpoint checkpoints/spar-native --red-tactics checkpoints/striker \
  --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill \
  --video runs/fight.mp4
# tests, one file per process (the whole suite in one process runs out of memory on WSL)
bash tools/test_all.sh
bash tools/test_all.sh tests/test_spar.py tests/test_striker.py   # or only some files
```

Training commands for each part are in its report: the leg motor
(`boxer.curriculum_train --stage spar`, [spar motor](docs/spar-motor.md)),
tactics (`boxer.native_finetune`, [tactics](docs/tactics.md)), the striker
(`boxer.striker_train`, `boxer.native_finetune --striker`,
[striker](docs/striker.md)) and the legs in two-robot bouts
(`boxer.motor_native`). Quickstart for the original MJX and warp stacks:
[archive](docs/archive/mjx-warp-experiments.md#quickstart-for-the-mjx-and-warp-stacks).

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

