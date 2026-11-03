# Sparring motor: balance that survives its own punches

> Update 2026-10-31: `checkpoints/spar` is now the motor retrained for hooks
> and the high guard ([moves](moves.md)). The straight-only motor described
> here is `checkpoints/spar-straight`.

Date: 2026-10-30.

## Diagnosis

From 10-21 to 10-29 the striking work kept the certified `approach` motor
frozen and asked PPO arm residuals to add punches. Every variant (v12–v17,
GPU batching, KL guards, GAE, opponent pools) ended with the same failure:
pairs fall once both robots act. The leg controller was the cause. It had
been trained with its arms held in the guard pose, so arm motion and
received punches were dynamics it had never seen.

Measured directly with the new `spar` stage (below), the certified motor
alone, with no opponent, survives only **0–19 %** of 10-second trials once its
arms throw punches (mean survival 2–5 s). Its own `approach` gate is ≥ 90 %. No
arm-residual objective can fix that from above.

## Change

`UnassistedG1("spar")` (in `boxer/curriculum_env.py`) is the walking task
with these changes:

* `punch_arms` replaces the 14 arm actions with randomized straight punches.
  Each punch picks a hand and a height between body and head, with ±0.12 rad
  endpoint jitter and 0.8–1.25× speed, using the skills controller's
  FK-checked targets and 0.18 s / 0.26 s extension/retraction. Pauses of
  0.1–1.2 s allow combos. The policy still controls legs and waist and sees
  the arm state through its normal joint observations and last action.
* Shoves every 1–3 s at 0.1–0.5 m/s stand in for received punches. The
  walk stage uses 3–5 s at 0.1–0.3.
* Footwork commands come from the walk ranges, with 50 % standing. The
  evaluation uses half-speed walking commands and the balance stage's push
  trial.

Fine-tuning starts from `checkpoints/curriculum/approach` with the unchanged
curriculum trainer:

```bash
bash tools/run_spar.sh boxer.curriculum_train --stage spar \
  --init-from checkpoints/curriculum/approach --output runs/spar/train \
  --steps 40000000 --worlds 1024 --checkpoints 40 --adaptive-gates --gate-every 10000000
```

## Results

Solo `spar` gate (2 seeds × 7 commands × push on/off, 32 worlds, 10 s):

| Motor | Mean survival | Worst trial | Gate |
| --- | --- | --- | --- |
| certified `approach` (frozen) | 0.06 | 0.00 | fail |
| `spar` @ 10.8M (≈ 8 min of training) | 0.994 | 0.969 | fail: standing drift 0.21 m/s vs 0.20, weak left step |
| `spar` @ 21.6M | 0.999 | 0.969 | **pass** |
| `spar` @ 32.4M | 1.000 | 1.000 | fail: one tracking error 0.216 |
| `spar` @ 39.3M (final, now `checkpoints/spar-straight`) | **1.000** | 1.000 | **pass** |

Two-robot bouts (`boxer/spar_match.py`): native MuJoCo, free roots, no pelvis
support, scripted ring tactics, 20 s or first knockdown, seeds 0–7:

| Motor | Went the distance | Mean length | Clean punches / bout | Thrown / bout |
| --- | --- | --- | --- | --- |
| certified `approach` (15 s, seeds 0–3) | 0/4 | 3.0 s | 0.75 | 3 |
| `spar` @ 10.8M | 8/8 | 20.0 s | 8.9 (both corners score) | 58.5 |
| `spar` @ 39.3M (final) | **8/8** | 20.0 s | **18.6** (red 2 wins, blue 5, 1 draw) | 64.6 |

Video: [`fights/spar_g1_vs_g1.mp4`](../fights/spar_g1_vs_g1.mp4) (final motor, seed 0), with a [JSON sidecar](../fights/spar_g1_vs_g1.json). Both robots approach, exchange punches and
stay up. Weakness: from about 8 s they drift into a chest-to-chest clinch and
rotate, so the scripted range control needs work.

```bash
bash tools/run_spar.sh boxer.spar_match --checkpoint checkpoints/spar-straight \
  --seeds 0 1 2 3 4 5 6 7 --seconds 20 --video runs/spar/match.mp4
```

## What this is and is not

* It is unassisted, two-sided physical boxing. Both bodies are free, every
  joint is driven by its actuator, and contact and scoring use the existing
  impact scorecard.
* Punch *selection* is scripted (`RingBrain`). The motor is learned; the
  punches are fixed primitives. The next step is to replace `RingBrain` with
  a self-play PPO policy that has the 6-decision interface of the archived
  assisted mode (lead, rear, aim, advance, circle, guard). That mode learned
  real exchanges in 2M transitions once balance was not the bottleneck.
* Blocking, slipping and punch types other than straight punches are not
  trained yet. Adding hooks and guard movements to `punch_arms` is the same
  kind of change as this one.

## Phantom shins and real falls (2026-11-01)

With the arm-trained tactics, 16 native ring bouts of 120 s had 13
knockdowns. `tools/knockdown_causes.py` records what happened in the last
second before each one, next to how often it happens at any moment:

| Before a knockdown | Share of knockdowns | Base rate (any 1 s) |
| --- | --- | --- |
| rope contact | 8 % | 12 % |
| clean punch taken | 38 % | 81 % |
| body contact between the robots | 54 % | 25 % |

Every knockdown after body contact involved the robots' **shins** touching;
none followed upper-body contact. Body contacts are short bumps (median
40 ms, at most 0.24 s, 30–190 N), about the size of the shoves the motor
already trains on.

**Phantom shins.** `curriculum_train --phantom-shins` adds a mocap pair of
opponent shins to the spar scene (`curriculum_env.PHANTOM_*`). Every 2–6 s
they appear 0.45 m from the pelvis, mostly in front, sweep in to 0.18–0.30 m
and back out over 0.3–2 s, colliding only with the robot's shins and feet.
In this test (256 worlds, 10 s) the current motor survives 98 % without
them and 83 % with them.

| Motor | 10 s survival without / with phantom shins | Standard spar gate |
| --- | --- | --- |
| `checkpoints/spar` | 98.0 % / 83.2 % | pass |
| phantom 10.8M (`runs/spar-phantom`) | 99.6 % / 91.4 % | pass (at 0 and 10.8M) |
| phantom +29.5M (`runs/spar-phantom2`) | 99.2 % / 94.1 % | **fail**: standing drift 0.20, sideways tracking 0.21–0.23 |

**Most "knockdowns" were not falls.** `PairedFoundation.fallen` keeps the
foundation's self-trip rule: a robot whose foot touches its own other foot
or shin counts as fallen. Of the 13 knockdowns, 7 were that rule and only 6
were real falls (pelvis below 0.55 m). Under knockdown rules
(`spar_match --knockdowns N`, N > 1) a knockdown now counts only real falls;
a robot whose feet touch is still standing and boxes on. With one fall
ending the bout (the gates and every earlier result), nothing changes.

Under the new rule, the same 16 bouts of 120 s:

| Motor | Real knockdowns | After shin contact |
| --- | --- | --- |
| **`checkpoints/spar`** | **6** | 5 |
| phantom 10.8M | 9 | 6 |

The self-trips did not turn into real falls (13 → 6). The phantom-trained
motor is sturdier in the synthetic test but not in real bouts, where the
tactics and the arm skill were trained with the current motor. The motor
stays `checkpoints/spar`; `--phantom-shins` remains available for later
experiments.

**Tactics retrained on the phantom motor.** To rule out that the phantom
motor only lost because the tactics above it were trained on the old one,
the champion tactics were fine-tuned again on the phantom motor (3M native
steps, same settings as the champion, `native_finetune --motor
runs/spar-phantom/checkpoint`; the gate now plays on the training motor).
The gate barely passed: 24–22–2 against the champion tactics on the same
motor, with 15 falls in 48 bouts of 20 s (the champion's own gate had none).
The two complete stacks, each with its own legs (`spar_match
--blue-checkpoint`), 16 ring bouts of 120 s, three knockdowns stop a bout:

| Stack | Real knockdowns (vs ring-v1, phantom/old motor) | Head-to-head | Knockdowns suffered head-to-head |
| --- | --- | --- | --- |
| **`checkpoints/spar` + `checkpoints/tactics`** | **6** | **15–1** | **1** |
| phantom motor + retrained tactics | 16 | 1–15 | 21 (2 TKO) |

Every phantom-stack knockdown was a real fall (pelvis low), 11 of 16 after
leg contact. Retraining the tactics did not help: the phantom motor is
sturdier against the synthetic shins but falls more in real bouts, whatever
drives it. The motor stays `checkpoints/spar`.

**Tactics that keep their legs apart.** `native_finetune
--leg-contact-penalty P` takes `P` reward per second the robots' legs (hips
to feet, `PairedFoundation.legs_touching`) touch. Fine-tuned from the
champion with P = 20 (3M steps, otherwise the champion's settings), leg
contact in training fell from 2.4 % to 1.7 % of steps. The gate selected the
1.5M checkpoint, which only drew with the champion (15–16–17, with about 10
clean punches per bout instead of 18–21): it avoids contact by boxing less.
Same 16 ring bouts of 120 s as above:

| Tactics | Real knockdowns | Falls after leg contact | Falls on the ropes | Steps with leg contact |
| --- | --- | --- | --- | --- |
| **champion** | **6** | 5 | 1 | 1892 |
| leg penalty 1.5M (gate pick) | 8 | 1 | 3 | 899 |
| leg penalty 3.0M | 9 | 3 | 5 | 1526 |

Leg contact halves and almost disappears before falls, yet the robots fall
as often (6 → 8–9 is within the noise of 16 bouts); the falls move to the
ropes and to moments without contact. Shin contact was a trigger, not the
cause: the motor is close to its balance limit while it punches about twice a
second, and any disturbance tips it over. Not promoted; the penalty
defaults to 0.

## Legs for rapid punching (2026-11-02)

Every one of the knockdowns above came while the falling robot's own punch
was in flight. Learned tactics throw about 2.2 punches per second per boxer,
nearly back to back, and the arm skill bends each one by up to 0.35 rad. The
spar generator the motor trained with waits 5–60 steps between punches
(about one per second, an arm in motion 40 % of the time) and never bends a
punch. `curriculum_train --rapid-punches` (`curriculum_env.RAPID_PUNCH_*`)
waits 0–6 steps instead and adds a random correction of up to ±0.35 rad to
the punching arm's shoulder and elbow targets for each punch (an arm in
motion 81 % of the time). Default training is unchanged.

10 s survival, 256 worlds, as for the phantom shins:

| Motor | Normal punching | Rapid punching | Standard spar gate |
| --- | --- | --- | --- |
| `checkpoints/spar-v1` (previous) | 98.4 % | 93.8 % | pass |
| **rapid 10.8M (`checkpoints/spar`)** | **99.2 %** | **97.7 %** | pass (at 0 and 10.8M) |

With the old tactics on top, the new motor is worse: the champion tactics
fell 19 times in 16 ring bouts of 120 s against ring-v1 (4 on the old motor).
The tactics had learned commands that suit the old legs. Fine-tuned again on
the new motor (3M native steps, the champion's settings, `native_finetune
--motor`), the 3.0M tactics beat the previous champion tactics 30–13–5 in the
native gate on the new motor. Whole stacks, each robot on its own legs
(`spar_match --blue-checkpoint`), ring bouts of 120 s, three knockdowns stop
a bout:

| Bouts | New stack's record | Knockdowns, previous / new stack |
| --- | --- | --- |
| head-to-head, seeds 0–7 swapped | 9–7 | 5 / 1 |
| head-to-head, seeds 8–15 swapped | 13–2–1 | 8 / 1 |
| **head-to-head, 32 bouts** | **22–9–1** | **13 / 2** |
| against ring-v1 on the previous motor | 16–0 (previous stack: 14–2) | 4 / 4 (each stack's own) |

Promoted: the motor is `checkpoints/spar` (previous: `checkpoints/spar-v1`;
since 2026-11-03 this motor is `checkpoints/spar-v2`),
the tactics `checkpoints/tactics` (previous: `checkpoints/tactics-arm-v1`;
their `native-gate.json` calls it `checkpoints/tactics`, its name then).
Against a common opponent the new stack falls as often as the old one; it
falls far less in the head-to-head. Tactics trained on the previous motor
should be played on it (`--checkpoint checkpoints/spar-v1`); with both
corners on the new motor, ring-v1 fell 8 times in 16 bouts instead of 2.
The arm skill was trained on the previous motor and is used unchanged.

## Legs for the striker's arms (2026-11-03)

The [striker](striker.md) sets both arms' joint targets itself and fell 14
times in 16 bouts of 120 s against the champion's 1. Its arms move in ways
the generator never produced: both arms at once, held out near the opponent,
short repeated motions instead of punches that return to the guard.

`striker_train arms` records the 14 arm actions the striker sent in native
ring bouts (200,000 control steps, 8 workers, against the champion pool and
scripted opponents, the same mix it trained in). `curriculum_train
--striker-arms <npz>` replays them in half the spar episodes
(`curriculum_env.STRIKER_ARM_FRACTION`), each from a random point in the
recording, in place of the generated moves; the other half keeps the
generator. The motor was fine-tuned from `checkpoints/spar` with
`--rapid-punches --striker-arms` (seed 44) and passed the spar gate at 0 and
10.8M steps. The recording is kept as `checkpoints/spar/striker-arms.npz`.

10 s survival, 256 worlds; "striker arms" replays the recording in every
episode:

| Motor | Normal punching | Rapid punching | Striker arms | Standard spar gate |
| --- | --- | --- | --- | --- |
| rapid (`checkpoints/spar-v2`) | 99.6 % | 98.0 % | 85.2 % | pass |
| **striker arms 10.8M (`checkpoints/spar`)** | **100 %** | **100 %** | **95.3 %** | pass (at 0 and 10.8M) |

Native ring bouts of 120 s, three knockdowns stop a bout, corners swapped:

| Bouts | Record | Clean punches per bout | Knockdowns |
| --- | --- | --- | --- |
| striker vs champion, both on the previous legs (before, 16 bouts) | 16–0 | 269 : 72 | 14 : 1 |
| striker on the new legs vs champion on the previous legs (32) | 32–0 | 317 : 123 | 2 : 2 |
| **striker vs champion, both on the new legs (32)** | **32–0** | **309 : 141** | **1 : 0** |
| champion on the new legs vs champion on the previous legs (32) | 23–9 | | 6 : 29 |

The striker was not retrained: the new legs alone take its knockdowns from
14 in 16 long bouts to 1 in 32. The champion stack falls less on them too.
A striker fine-tuned on the new legs (3M native steps, the striker's
settings; the gate picks 2.25M, 48–0 against the striker on the same legs)
lands more, 429 clean punches per bout to 110 against the champion, but
suffers 8 knockdowns in 32 bouts to 0. Not promoted.

Promoted: the motor is `checkpoints/spar` (previous: `checkpoints/spar-v2`).
The champion tactics, the arm skill and the striker are unchanged and play
on it.

## Legs trained in two-robot bouts (2026-11-03)

Solo training stands in for an opponent with replayed arm motions and random
shoves. A striker fine-tuned with an arm-motion cost
([striker](striker.md#arm-motion-cost-2026-11-03)) kept falling (29
knockdowns in 32 bouts of 120 s against the champion, 18 on legs retrained
solo on its own recording, whose solo survival under its arms was 97.7 %):
what remains is two bodies leaning and pushing at close range.

`boxer.motor_native` fine-tunes the motor with PPO in native two-robot ring
bouts, the physics of evaluation. Both robots run the learner motor and both
robots' transitions train it. Each corner's commands and arms come from a
brain drawn per bout: the striker, the low-motion striker, the champion
tactics with the arm skill, or the scripted ring brain. Only the 15 leg and
waist outputs carry log-probabilities (the arms are always replaced). The
reward keeps the solo task's main terms per step (velocity tracking 1.0 and
0.75, upright torso −2, tilt rate −0.15, action rate −0.01, times the
control interval) and charges 10 for a fall; a fall ends the bout, terminal
for the robot that fell. The motor's critic sees privileged simulator state
that native bouts lack, so a new critic (two layers of 256) sees the policy
observation, where the opponent is and which brains fight; the policy waits
8 updates for it. Policy learning rate 2e-5, updates stop early beyond a KL
of 0.01.

From `checkpoints/spar`, 3M robot transitions (8 workers, 27 minutes): falls
in training fell from 0.073 to about 0.012 per 1000 transitions. The
standard spar gate passes (100 % survival, like `checkpoints/spar`).

| Bouts of 120 s (32 each) | Record | Clean punches per bout | Knockdowns |
| --- | --- | --- | --- |
| low-motion striker vs champion, both on `checkpoints/spar` | 29–3 | 300 : 65 | 29 : 0 |
| the same on the solo retrain with its recording | 30–2 | 303 : 51 | 18 : 0 |
| **the same on the two-robot legs** | **32–0** | **276 : 60** | **2 : 1** |
| striker-v1 vs champion on the two-robot legs | 32–0 | 325 : 213 | 0 : 0 |
| champion on the two-robot legs vs champion on `checkpoints/spar` | 4–28 | | 1 : 12 |

The champion falls far less on them but loses 4–28 to itself on the
current legs: its tactics learned commands that suit those legs (as in
[legs for rapid punching](#legs-for-rapid-punching-2026-11-02)).

Kept as `checkpoints/spar-native`, the legs for strikers; the champion keeps
`checkpoints/spar`. Play each on its own legs with `spar_match --checkpoint`
and `--blue-checkpoint`.

**More striker brawls.** Continued from `checkpoints/spar-native` for 3M
more transitions with the pool weighted toward strikers (`--weights 1 2 1
.5`, BRAINS order; about 44 % of bouts striker against striker). Training
falls stayed at 0.02–0.03 per 1000 transitions. 32 bouts of 120 s:

| Bouts | `spar-native` | continued |
| --- | --- | --- |
| striker vs striker-v1, both on the legs tested | 32–0, knockdowns 15 : 5 | 32–0, knockdowns 9 : 12 |
| striker on the legs tested vs champion on `checkpoints/spar` | 32–0, knockdowns 1 : 4 | 31–1, knockdowns 0 : 7 |

The brawls fall as often as before (21 knockdowns against 20); the falls
only change sides. Not promoted.
