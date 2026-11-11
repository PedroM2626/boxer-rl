# Striker: one policy for footwork and both arms (2026-11-03)

## Why

The champion stack splits a boxer into separately trained parts: tactics
choose when, with which hand, how high and which **scripted move** to throw
([tactics](tactics.md)); fixed joint paths (`boxer/moves.py`) execute it; an
[arm skill](arm-skill.md) bends the path by a degree or two. Two limits
follow: the parts never learn together, and the robot can only throw the
punches someone wrote down (in practice it throws straights, rarely a hook,
never anything else).

The striker (`boxer/striker.py`) replaces tactics, scripted moves and arm
skill with **one policy**. Every control step (50 Hz) it outputs:

* footwork: advance and circle (turning toward the opponent stays
  automatic, as in `tactics_env.decode`);
* the shoulder pitch, roll, yaw and elbow targets of **both arms** (the right
  arm in the left arm's convention).

There is no punch request, no move list and no fixed path. A punch is
whatever arm motion the policy produces, and it scores under the same
native impact scorecard as any other (fist contact on head or torso with
enough impulse, closing speed and fist speed relative to the own torso).
Arm targets stay in a box around every scripted pose widened by 0.35 rad
and move at most 0.3 rad per step (scripted swings: up to about 0.22). The
leg motor keeps balance (`checkpoints/spar-v2` when the striker was first
trained, `checkpoints/spar-native` since 2026-11-03): the legs remain a
separate layer.

Observation (74): the tactics observation (32), and per arm the joint angles
and velocities, the fist relative to the opponent's head and torso, and the
fist velocity relative to the torso (all in the torso frame), plus both
arms' current targets.

## Training

1. **Imitation** (`boxer/striker_train.py collect`, `clone`). 640k native
   ring steps of the champion stack (champion tactics, scripted moves, arm
   skill, rapid-punching motor) against the champion pool and scripted
   opponents, recording the striker observation and what the stack did: its
   advance/circle outputs and the arm targets it actually sent. A striker
   fitted to them by least squares (held-out squared error 0.003–0.009 for
   footwork, 0.0001–0.01 for arm targets, in [−1, 1] units), with a value
   function fitted to discounted native rewards. The clone boxes but is
   weaker than the original: **2–14** against it, 10.5 clean punches per
   bout to 17.7 (small errors compound).
2. **Native PPO** (`native_finetune --striker`), from the clone: the
   champion's reward and opponents (pool of champion and highguard-v1
   tactics with the arm skill, half scripted opponents, 60 % of those on a
   high guard), 3M steps, exploration floor 0.1. Clean punches per 1000
   training steps rose from 15.9 (first quarter) to 35.6 (last quarter);
   the champion tactics' training ran at about 19. The gate picked the
   3.0M checkpoint: **41–7** over 48 bouts against the clone.

## Results

Native ring bouts against the champion stack (champion tactics + arm
skill), both on the rapid-punching motor, corners swapped:

| Bouts | Striker's record | Clean punches per bout, striker : champion | Knockdowns suffered, striker / champion |
| --- | --- | --- | --- |
| 32 of 20 s | **28–4** | 29.6 : 8.3 | 4 / 4 |
| 16 of 120 s (TKO at 3) | **16–0** | 269 : 72 | **14 / 1** |

[30 s video](../fights/striker_vs_champion_30s.mp4) (striker in red, 72–15).

**What it learned is in-fighting.** The striker closes to chest-to-chest
range and keeps both fists high and busy, throwing short punches at the
opponent's head from point-blank range, about two clean punches a second.
At that range the champion's long scripted punches do not fit.

**Repertoire** (`tools/punch_shapes.py`, 4 bouts of 30 s against the
champion, red corner). A punch is a stretch where a fist is over 12 cm from
both guards. It is named by its peak (the point farthest from the guard):
straight or hook when the peak lies within 8 cm of a scripted straight's
path or a hook's swing, otherwise new, by the peak's main direction.

| Red corner | Punches | Straight | Hook | New (down / up / forward) | Mean reach | Mean rise |
| --- | --- | --- | --- | --- | --- | --- |
| champion (scripted + arm skill) | 310 | 91 % | 4 % | 1 % / 0 / 4 % | 0.18 m | 0.06 m |
| clone | 149 | 78 % | 17 % | 4 % / 0 / 1 % | 0.18 m | 0.08 m |
| **striker** | 282 | 15 % | **64 %** | **15 % / 6 %** / 0 | 0.13 m | 0.11 m |

The champion is almost all straights. The striker has moved away from them:
its punches are shorter and most peak along a hook's swing (short, curved
punches from the side), and a fifth reach points no scripted move reaches,
low and high. These are trajectories the policy produced itself.

## Limits

* **Falls (fixed in the legs).** On the motor it was trained with
  (`checkpoints/spar-v2`), 14 knockdowns in 16 bouts of 120 s against 1 for
  the champion: that motor never trained under these arm motions. The leg
  motor now trains replaying the striker's own arm actions; on it the
  striker wins 32–0 over 32 bouts of 120 s, 309 clean punches to 141, with
  1 knockdown to 0 ([legs for the striker's arms](spar-motor.md#legs-for-the-strikers-arms-2026-11-03)).
* **Close to a clinch.** Fighting chest to chest is legal under these rules
  (no referee breaks clinches) but is not what boxing looks like at its best.
* **Legs are still separate.** The striker merges tactics, moves and arm
  skill; the leg motor is still its own policy, now trained under the
  striker's recorded arm actions.
* **Punching costs almost nothing.** The only cost is 0.02 per launch, and a
  striker launch is an arm leaving the guard; it keeps its arms out and
  launches about 3 times per 120 s bout. Missing is free. An arm-motion cost
  did not help (below).

## Arm-motion cost (2026-11-03)

`spar_match` now reports `arm_motion_rad`, the radians each corner's
shoulder and elbow targets move over a bout (both arms, 8 joints), whatever
drives the arms. `native_finetune --arm-motion-cost c` takes `c` reward per
radian from a striker learner.

The striker's arm targets move about 1 rad per control step, so its arms
look busy all the time. Yet per clean punch it spends what the champion
does: in 32 bouts of 120 s on the current legs it moves 55 rad/s and lands
309 clean punches per bout; the champion's scripted arms move 26 rad/s and
land 141; that is 21.4 and 21.7 rad per clean punch.

Fine-tuned from `checkpoints/striker` (now `striker-v1`) with `--arm-motion-cost .05` (about
half its hit reward at its own motion; otherwise the striker's settings,
3M native steps on `checkpoints/spar`), training motion fell from 1.39 to
about 1.1 rad per step (exploration included). The gate picks 1.5M (48–0
against the striker). Against the champion, 32 bouts of 120 s:

| Striker | Record | Clean punches per bout | Arm motion | Rad per clean punch | Knockdowns (striker : champion) |
| --- | --- | --- | --- | --- | --- |
| `checkpoints/striker` (now `striker-v1`) | 32–0 | 309 : 141 | 55 rad/s | 21.4 | 1 : 0 |
| arm-motion cost 0.05, 1.5M | 29–3 | 300 : 65 | 49 rad/s | 19.2 | **29 : 0** |

It moves 11 % less and keeps the champion to half the clean punches, but
it falls 29 times: its arms no longer move like the recording the leg
motor trained under ([legs for the striker's arms](spar-motor.md#legs-for-the-strikers-arms-2026-11-03)).
Not promoted.

**Legs retrained on its recording.** `striker_train arms` recorded the
low-motion striker too (200,000 steps, 0.89 rad of arm-target motion per
step against the striker's 0.97). The motor was fine-tuned from
`checkpoints/spar` with both recordings (`--rapid-punches --striker-arms`,
400,000 steps, seed 45) and passed the spar gate at 0 and 10.8M. 10 s solo
survival, 256 worlds, replaying each recording in every episode:

| Motor | Normal | Rapid | Striker arms | Low-motion striker arms |
| --- | --- | --- | --- | --- |
| `checkpoints/spar` | 100 % | 98.4 % | 96.1 % | 91.8 % |
| both recordings, 10.8M | 99.6 % | 97.3 % | 94.1 % | 97.7 % |

Against the champion, everyone on the retrained legs, 32 bouts of 120 s:

| Striker | Record | Clean punches per bout | Arm motion | Knockdowns (striker : champion) |
| --- | --- | --- | --- | --- |
| `checkpoints/striker` (now `striker-v1`) | 32–0 | 346 : 126 | 55 rad/s | 2 : 0 |
| arm-motion cost 0.05, 1.5M | 30–2 | 303 : 51 | 49 rad/s | 18 : 0 |

The legs take the low-motion striker's knockdowns from 29 to 18, still far
more than the striker's 2. Solo survival under its arms is already higher
than under the striker's, so its falls come from what solo training does
not reproduce, most likely the body contact of its fighting at closer
range. Neither this motor nor the striker was promoted then.

## Legs trained in two-robot bouts (2026-11-03)

The motor then trained where those falls happen: native two-robot ring
bouts (`boxer.motor_native`, [spar motor](spar-motor.md#legs-trained-in-two-robot-bouts-2026-11-03)),
with both strikers, the champion and the scripted brain in its opponent
pool. On those legs (`checkpoints/spar-native`), 32 bouts of 120 s:

| Bouts | Record | Clean punches per bout | Knockdowns |
| --- | --- | --- | --- |
| low-motion striker vs champion, both on the new legs | 32–0 | 276 : 60 | 2 : 1 |
| striker-v1 vs champion, both on the new legs | 32–0 | 325 : 213 | 0 : 0 |
| low-motion striker vs striker-v1, both on the new legs | 32–0 | 367 : 229 | 15 : 5 |
| **low-motion striker on the new legs vs champion on its own** | **32–0** | **256 : 87** | **1 : 4** |
| striker-v1 on the new legs vs champion on its own | 32–0 | 267 : 211 | 0 : 0 |

The low-motion striker's knockdowns against the champion fall from 29
(and 18 on legs retrained solo on its recording) to 2. Its brawls with
striker-v1, two in-fighters chest to chest, still bring falls.

Promoted: the low-motion striker is `checkpoints/striker` (previous:
`checkpoints/striker-v1`), and strikers play on `checkpoints/spar-native`.
The champion stack (`checkpoints/tactics` + `checkpoints/arm-skill`) keeps
`checkpoints/spar` and stays the reference built from scripted moves. Each
on its own legs:

```bash
bash tools/run_spar.sh boxer.spar_match --ring --checkpoint checkpoints/spar-native --red-tactics checkpoints/striker \
  --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill \
  --seconds 120 --knockdowns 3 --swap --seeds 0 1 2 3
```

`spar_match` recognizes a striker checkpoint by its sizes (74 → 10) wherever
it takes tactics.

## Long-bout gate and alternating rounds (2026-11-11)

**Gate on long bouts.** The native gate (`boxer/tactics_gate.py`) chose
checkpoints from 20 s bouts, but falls show up in 120 s bouts. The striker
with the arm-motion cost had 0 knockdowns in the gate and 29 in long bouts.
`native_finetune --long-gate N` now runs a long check on the top N
candidates before confirming one: 8 bouts of 120 s with knockdown rules
(seeds 24–31, corners swapped) against the gate's champion and against every
`--long-gate-opponents`. A candidate is stable when it suffers at most
`opponent knockdowns + max(2, opponent knockdowns)` against every opponent.
If none is stable, the gate keeps the champion (`native_gate_rejected`).

Checked against known cases:
- **Positive control:** tactics vs tactics-arm-v1 on `spar-v2` is stable
  (9 : 5).
- **Negative control:** the low-motion striker on the old `spar` legs vs the
  champion tactics is unstable (17 : 1).
- The opponent matters. Against striker-v1 that same striker looked stable
  (2 : 1), so the champion tactics is a standing long-gate opponent.

**Alternating rounds** (`python -m boxer.co_train`). Each round has two
phases:
1. Legs: `motor_native`, 1M transitions, with the current striker in its
   pool.
2. Striker: `native_finetune --striker`, 750k transitions on those new legs,
   gated with the long check against the current striker and the champion
   tactics.

New legs always carry over; the striker carries over only when the gate
selects it. Three rounds, each about an hour:

| Round | Striker long check (knockdowns, candidate : opponent) | Gate |
| --- | --- | --- |
| 0 | 375k: 13 : 10 vs striker, 10 : 0 vs champion; 750k: 18 : 13, 21 : 0 | rejected |
| 1 | 375k: 6 : 3 vs striker, 1 : 0 vs champion | selected (44–4 in confirmation) |
| 2 | 750k: 20 : 26 vs round-1 striker, 4 : 0 vs champion; 375k: 14 : 4 | rejected |

The final pair is the round-1 striker on the round-2 legs. Its result in
32 bouts of 120 s (seeds 0–15, corners swapped):

| Bouts | Record | Clean punches per bout | Knockdowns |
| --- | --- | --- | --- |
| co-trained pair vs champion on its own legs | 32–0 | 273 : 103 | 2 : 2 |
| co-trained pair vs itself | — | 232 : 207 | 28 in all |
| co-trained pair vs current striker on `spar-native` | 32–0 | 280 : 188 | 12 : 4 |
| current striker vs itself on `spar-native` (baseline) | — | 234 : 252 | 20 in all |

The rounds did not reduce brawl falls. Two co-trained strikers fall 28
times in 32 bouts, against 20 for the current striker. Against the current
striker, the co-trained pair goes down three times as often. Against the
champion it is level with the current striker (2 : 2 against 1 : 4). Not
promoted. Each phase adapts to the other, but the striker phase still
learns to hit harder in 30 s episodes, and its legs pay for it in long
bouts. The [whole-body policy](whole-body.md) removes the two phases
altogether.
