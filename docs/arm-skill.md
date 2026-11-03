# Learned punches: the arm skill (2026-10-31)

Until now every punch followed a fixed trajectory (`boxer/moves.py`): the
arm swung along a scripted joint path to a fixed pose, wherever the opponent
actually was. Tactics chose when to punch, with which hand, how high and
which move, but not the path.

The arm skill (`boxer/arm_skill.py`) is a third learned layer, between
tactics and the scripted moves. It does not replace the hierarchy. On every
20 ms step of a punch it sees:

* the punching fist relative to the point it was asked to hit (the
  opponent's torso centre for aim 0, the head for aim 1);
* the fist's and the target's velocity relative to its own torso;
* the punching arm's four joint angles and velocities;
* how far the punch has progressed.

It adds a residual of up to ±0.35 rad to the punching arm's shoulder pitch,
roll, yaw and elbow targets. The right arm is mirrored into the left arm's
frame, so one policy serves both hands. A fresh arm skill outputs exactly
zero: its last layer starts at zero, and a test and a native bout check it
reproduces the scripted punches bit for bit.

## Training (`boxer/arm_finetune.py`)

Native-physics PPO, with the same workers as the tactics fine-tune. Both
corners box with frozen tactics (the current champion and highguard-v1,
30 % scripted opponents) and the frozen leg motor, inside the roped ring.
One corner per bout has its punches steered by the learner. Every punch is
one episode, from launch to the end of the retraction:

* reward +1 per clean punch (native impact scorecard);
* −0.02 × squared residual (in units of 0.35 rad);
* −5 and the end of the episode if the boxer falls.

Run `runs/arm-skill`: 3M punch steps (about 70 minutes at about 690 steps/s),
exploration floor 0.1, learning rate 3e-4. Clean punches per punch in
training rose from about 0.40 to 0.44–0.45.

## Gate

Native bouts in the ring: champion tactics with the arm skill against the
same champion tactics with scripted punches, corners swapped. The only
difference between the corners is the arm skill.

| Checkpoint | W–L–D (16 bouts, seeds 0–7) | Clean punches per bout | Landing rate: arm skill / scripted |
| --- | --- | --- | --- |
| **0.75M** | **14–1–1** | 21.9 : 17.8 | **47.9 % / 39.0 %** |
| 1.5M | 13–0–3 | 21.9 : 17.7 | 47.8 % / 38.7 % |
| 2.25M | 13–3–0 | 21.1 : 16.2 | 48.4 % / 36.8 % |
| 3.0M | 9–4–3 | 21.4 : 18.5 | 46.9 % / 40.4 % |

Confirmation on fresh seeds and long ring bouts:

| | 0.75M | 1.5M |
| --- | --- | --- |
| 32 bouts, seeds 8–23 | **25–3–4**, landing 49.8 % / 41.0 % | 25–7–0, landing 52.7 % / 44.7 % |
| 48 bouts in total | **39–4–5** | 38–7–3 |
| 16 bouts of 120 s (TKO at 3) | **16–0**, landing 56.2 % / 47.6 % | 13–3, landing 55.6 % / 46.0 % |
| Knockdowns suffered in those (arm skill / scripted) | **2 / 3** | 6 / 1 |

The 1.5M checkpoint lands slightly more but falls more over long bouts. The
0.75M checkpoint has no balance cost and becomes `checkpoints/arm-skill`
(its selection and confirmation records sit next to it).

## What it learned

Over three 20 s bouts the residuals are small: mean absolute 0.018 /
0.015 / 0.018 / 0.026 rad (shoulder pitch, roll, yaw, elbow) for 0.75M,
about 1–1.5°, well inside the ±0.35 rad allowed. Corrections of that size,
aimed at where the target actually is, raise the share of punches that
score from about 41 % to about 50 % (a relative gain of roughly a fifth).
These are learned real-time corrections around the scripted paths, not
whole new trajectories. The fall penalty and the motor's training
distribution keep them small, and the 1.5M checkpoint shows that larger
ones start to cost balance.

## Using it

```bash
bash tools/run_spar.sh boxer.spar_match --ring \
  --red-tactics checkpoints/tactics --red-arm checkpoints/arm-skill \
  --blue-tactics checkpoints/tactics --swap --seeds 0 1 2 3
```

`--red-arm` / `--blue-arm` attach an arm skill to a corner; the bout
results name it after the tactics ("tactics + arm …").

## Tactics trained with the arm skill

`native_finetune --arm checkpoints/arm-skill` steers every corner's punches
with the arm skill during training, and the native gate then gives it to
both corners so only the tactics differ. Run `runs/tactics-arm`: from the
ring champion, 3M native steps inside the ring, same pool and opponents as
before.

Gate against the ring champion, both corners with the arm skill:

| Candidate | W–L–D (16 bouts) | Clean punches per bout |
| --- | --- | --- |
| 0.75M | 9–5–2 | 21.6 : 19.0 |
| 1.5M | 11–3–2 | 21.4 : 17.6 |
| **2.25M** | **13–1–2** | **21.1 : 16.4** |
| 3.0M | 4–8–4 | 15.7 : 15.9 |
| **2.25M, 48 bouts with fresh seeds** | **35–11–2** | **20.9 : 17.1**; no falls in the 32 confirmation bouts |

16 bouts of 120 s in the ring (TKO at 3 knockdowns), new tactics with the
arm skill:

| Opponent | Result | Clean punches per bout | Landing rate | Knockdowns suffered (new / opponent) | Ropes (s per boxer) |
| --- | --- | --- | --- | --- | --- |
| ring champion + arm skill | **13–3** | 132.2 : 114.2 | 50.4 % / 42.0 % | 6 / 7 | 2.2 |
| ring champion, scripted punches | **14–2** | 131.0 : 100.4 | 49.3 % / 36.9 % | 7 / 3 | 1.4 |

The 2.25M tactics become `checkpoints/tactics`; play them with
`--red-arm checkpoints/arm-skill` (or `--blue-arm`). The ring champion moves
to `checkpoints/tactics-ring-v1`.

**The trade-off:** the new tactics win every comparison but box more
aggressively. Over long bouts they spend more time on the ropes and fall
more: 6–7 knockdowns in 16 bouts of 120 s, against 2 for the ring champion
pair without the arm skill. No bout ended in a TKO, and the 20 s gate saw
almost no falls.

Follow-up (2026-11-01): 7 of 13 of those knockdowns were the self-trip
rule, not real falls. With knockdown rules now counting only real falls,
the same bouts have 6 knockdowns. A motor retrained against phantom
opponent shins did not reduce them
([spar motor](spar-motor.md#phantom-shins-and-real-falls-2026-11-01)).

## Retrained on the rapid-punching motor (2026-11-02)

The arm skill was trained on the previous leg motor (`checkpoints/spar-v1`).
On the rapid-punching motor (`checkpoints/spar` from 2026-11-02, now
`checkpoints/spar-v2`) it still
works: in the arm gate (champion tactics with the arm skill against the same
tactics with scripted punches, 16 bouts) it goes 10–4–2, landing 48.4 % /
44.0 %. Fine-tuned from it on the new motor (`arm_finetune --init-from
checkpoints/arm-skill --tactics checkpoints/tactics`, 1.5M punch steps), then
matched directly against it, identical tactics on both sides:

| Checkpoint | Arm gate (16 bouts) | vs current arm skill, 32 bouts of 20 s | vs current, 16 bouts of 120 s | Knockdowns in those, new / current |
| --- | --- | --- | --- | --- |
| 0.75M | 13–3, landing 54.6 % / 47.2 % | 18–11–3 | 11–4–1, landing 54.0 % / 50.9 % | 8 / 3 |
| 1.5M | 10–6, landing 47.6 % / 42.1 % | 16–12–4 | 8–8, landing 51.4 % / 50.8 % | 7 / 4 |

Both retrained arms land a little more but fall more over long bouts, the
same trade-off that ruled out the original 1.5M checkpoint. Not promoted:
`checkpoints/arm-skill` stays.
