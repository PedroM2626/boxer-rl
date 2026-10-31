# Hooks and the high guard

Date: 2026-10-31. Builds on the [sparring motor](spar-motor.md) and
[learned tactics](tactics.md).

## Why

The second native self-play round plateaued at parity with the champion
(18–28–2). With only straight punches and footwork, the tactics looked
saturated. More moves give the tactics something new to learn.

## Moves (`boxer/moves.py`)

All poses are left-arm (shoulder pitch, roll, yaw, elbow); the right arm
mirrors roll and yaw. Hook and guard poses were fitted by forward kinematics
(`tools/fit_moves.py`, L-BFGS on fist and elbow positions in the torso frame).
One module serves the spar generator (JAX), batched tactics (JAX) and native
play (NumPy), so the three cannot drift apart. Tests check NumPy/JAX parity.

| Move | Path | Timing | Fist in torso frame |
| --- | --- | --- | --- |
| Straight (unchanged) | guard → strike | 0.18 s out, 0.26 s back | x 0.41–0.44 m |
| Hook | guard → wind-up (fist 0.30 m out to the side, 40 % of the extension) → strike across the centre line, elbow wide (y ≈ 0.20 m) and level | 0.26 s out, 0.28 s back | x 0.32–0.36 m, y ≈ 0 |
| High guard | both fists in front of the face | held while chosen | z 0.43 m (normal guard 0.26 m) |

Raised fists and forearms collide with incoming punches through the
existing fist-vs-hand/wrist/elbow contact pairs, so the high guard blocks
physically. Only head and torso contacts score.

## Motor: balance through the new moves

The spar generator now throws 40 % hooks and switches between the normal
and high guard every 0.5–3 s (40 % of periods high). The straight-only motor
(`checkpoints/spar-straight`) survives only 58 % of solo spar trials under the
new generator (worst trial 0.375). Fine-tuned from it (`runs/spar-moves`), the
motor passes the full spar gate at 10.8M and 21.6M steps with 100 % survival,
two consecutive passes, and becomes `checkpoints/spar`. In a native mirror of
the tactics champion with the new motor, all 16 bouts go the full 20 s with
no falls (31.4 clean punches per bout).

## Tactics interface

Tactics grow from 5 to 7 actions: hook instead of straight when output 6 is
positive, high guard when output 7 is positive. Observations grow from 26
to 30: both corners' "hook in flight" and "high guard" flags, appended at
the end so the first 26 entries keep their meaning. Older 26/5 checkpoints
still run: `tactics_policy` feeds them the first 26 inputs and pads their
actions with "straight, normal guard". `widen_legacy` turns such a checkpoint
into an equivalent 30/7 network: zero weights on the new inputs, new action
means at 0 and a moderate exploration scale. A test checks that it acts
identically. Native fine-tuning then learns when the new moves pay off.

## Training

```bash
bash tools/run_spar.sh boxer.native_finetune --init-from checkpoints/tactics \
  --output runs/tactics-moves --steps 3000000 --checkpoint-every 750000 \
  --workers 8 --pool checkpoints/tactics --native-gate 4 --seed 2
```

## Results (native MuJoCo, new motor for both corners)

3M native steps from the widened champion (`runs/tactics-moves`, 2.2 h at
about 380 steps/s; extra arm contacts make it slower than straight-only
training's 630). Gated against the previous champion
(`checkpoints/tactics-native-v1`, which keeps straights and the normal guard):

| Checkpoint | vs previous champion, seeds 0–7 | Clean punches / bout |
| --- | --- | --- |
| 0.75M | 12–3–1 | 17.9 : 14.9 |
| 1.5M | 6–8–2 | 16.1 : 15.3 |
| 2.25M | 11–5–0 | 15.8 : 13.8 |
| **3.0M** | **12–3–1** | 11.8 : 7.6 |
| **3.0M, 48 bouts with fresh seeds 8–23** | **33–12–3** | **12.4 : 8.9** |
| 3.0M vs scripted (16 bouts) | 15–1–0 | 12.3 : 1.75 |

How it boxes: acting deterministically, the new champion holds the **high
guard about a third of the time** (33–34 % of bout time), throws about as many
punches as before (44 per bout), and **never chooses hooks**. The guard is
the advantage. Opponents' clean punches fall from about 15 to 8.9 per bout
(previous champion) and from 5.3 to 1.75 (scripted brain). Its own scoring
also drops, since punches launched from the high guard start farther from
the target. In the videos the boxers now stay at punching range instead of
drifting into the chest-to-chest clinch.

Limits:

* **Hooks went unused.** They were explored during training (the new
  action started at 50 % hooks) but dropped. A likely reason is reach: 0.32–0.36 m
  against 0.41–0.44 m for straights, at the range these boxers keep. Hooks may
  need closer footwork, or a reward for angle and variety, to become useful.
* **It falls a little more:** 6 falls in 48 bouts against the previous
  champion's 2, and 1 in 16 against the scripted brain.

Published as `checkpoints/tactics`
([gate record](../checkpoints/tactics/native-gate.json)). The video
[`fights/tactics_guard_vs_native_v1.mp4`](../fights/tactics_guard_vs_native_v1.mp4)
(seed 3) shows it winning 11–4 with 5.7 s of high guard.

## Why hooks go unused (diagnostics, 2026-10-31)

`tools/move_effectiveness.py` pits a scripted attacker that throws only one
punch type against the scripted brain in the normal or high guard, in native
physics, with 6 bouts per row. It reports clean punches per punch thrown:

| Attacker / defender range | Punch | vs normal guard | vs high guard |
| --- | --- | --- | --- |
| 0.62 / 0.62 m | straight | 16.8 % | 8.8 % |
| 0.62 / 0.62 m | short hook (active) | 3.1 % | 1.5 % |
| 0.52 / 0.62 m | short hook | 6.1 % | 0 % |
| 0.62 / 0.62 m | long hook (candidate) | 4.5 % | 3.8 % |
| 0.57 / 0.62 m | long hook | 9.6 % | 1.5 % |
| 0.52 / 0.52 m | straight | 25.7 % | 7.4 % |
| 0.52 / 0.52 m | long hook | 3.7 % | 8.2 % |

* The active short hook ends 0.32–0.36 m out at the centre of the face,
  exactly where a high guard sits. The policy was right to ignore it.
* The opponent's guard fists are about 0.27 m in front of their torso, so a
  hook gets around them only at close range. At 0.52 m the long hook matches
  the straight against a high guard, on small counts.
* Tracing the fist with forward kinematics shows that straight-line
  interpolation of joint angles from the long wind-up to the strike lifts the
  head hook to z 0.535 m (an overhand arc over the opponent's arms) instead
  of sweeping level at 0.42 m. Video frames confirm this. The body hook stays
  level (z 0.18–0.20 m).

### Level long hook (v2, active since 2026-10-31 evening)

`tools/fit_moves.py` now solves IK at 6 waypoints on a horizontal arc around
the shoulder (each warm-started from and kept near the previous solution),
and `moves.punch_pose` swings through them. Traced with forward kinematics,
the head hook stays at z 0.426–0.431 m and the body hook at 0.182–0.184 m. It
sweeps from 0.31 m out to the side to the centre line and reaches 0.40–0.43 m
forward, about the straight's reach. The short hook is kept as `SHORT_*`.

A level arc alone did not raise the landing rate: 6.6 % / 1.4 % against the
normal / high guard at 0.62 m, and 2.6 % / 4.5 % at 0.52 m. A contact census
(`tools/fist_contacts.py`, 3 bouts per row, red throwing only one move at
0.55 m) shows what punches actually touch:

* **The head is essentially out of reach.** Hooks never touched it, and
  straights touched it in 55 substeps against 191 on the torso. The front of
  the head sits at the limit of the G1's reach at these distances.
* **Gloves meet gloves.** About 1,300–1,500 substeps per hand of fist-on-fist
  contact: both guards touch constantly and the arms stay tangled.
* **Head hooks run into arms and shoulders** (150–500 shoulder-contact
  substeps).
* **Body hooks get around a high guard.** Against a high guard, hooks touched
  the torso in 140 substeps against 63 for straights. The raised fists leave
  the body open from the side.

The motor (`runs/spar-levelhook`) passed the full spar gate with the new hook
already at step 0 and again at 10.8M steps. It becomes `checkpoints/spar`; the
short-hook motor is kept as `checkpoints/spar-shorthook`.

### Native fine-tune with the level hook

3M native steps from the guard champion with the new motor
(`runs/tactics-hooks`). The champion's hook output averages −0.02 with scale
0.23, so training sampled hooks about half the time. Gated against the guard
champion:

| Checkpoint | vs guard champion, seeds 0–7 | Clean punches / bout |
| --- | --- | --- |
| **0.75M** | **14–2–0** | 12.1 : 8.6 |
| 1.5M | 12–3–1 | 9.3 : 5.3 |
| 2.25M | 11–3–2 | 7.4 : 3.6 |
| 3.0M | 10–4–2 | 12.6 : 8.6 |
| **0.75M, 48 bouts with fresh seeds** | **37–9–2** | **11.7 : 8.8**; falls 1 vs 4 (confirmation) |
| 0.75M vs scripted (16 bouts) | 14–2–0 | 11.4 : 2.8 |

It became `checkpoints/tactics` (now `checkpoints/tactics-levelhook-v1`); the
guard champion is kept as `checkpoints/tactics-guard-v1`. Both use the high guard heavily (35–41 % and
50–56 % of bout time). **Hooks are still rare: about 2 % of the new champion's
punches**, up from 0 %. Its edge comes mainly from falling less and landing a
few more straights, not from hooks.

Where this leaves hooks:

* The level arc fixed the motion, and the motor balances through it. The
  census shows body hooks get around a high guard more often than straights.
* The policy uses hooks about 2 % of the time. The scoring audit below
  shows the scorecard is not the reason.

### Scoring audit (`tools/score_audit.py`)

Does the scorecard under-credit hooks? Its closing and fist-speed gates
measure speed along the fist -> target-centre line, which might reject a
fist that arrives from the side. The audit logs every contact onset of a fist
on a legal target (head or torso) in native bouts, with the move in flight,
the gate values, the same speeds along the contact normal, and the first
gate that rejected it. Run with 8 seeds, 20 s per bout:

| Bouts | Move | Target contacts | Scored | Rejected by |
| --- | --- | --- | --- | --- |
| scripted straight vs normal guard | straight | 64 | 60 (94 %) | closing 4 |
| scripted straight vs high guard | straight | **0** | — | — |
| scripted hook vs normal guard | hook | 8 | 7 (88 %) | closing 1 |
| scripted hook vs high guard | hook | 5 | 4 (80 %) | closing 1 |
| learned vs learned (champion) | straight | 221 | 194 (88 %) | fist speed 14, closing 11, impulse 2 |
| learned vs learned (champion) | hook | 0 | — | — |

Medians: hook contacts close at 0.8–0.9 m/s and push 1.7–2.3 N s, against
the 0.3 m/s and 0.5 N s thresholds. Across all rows, only 4 of the 31 gate
rejections would pass if speeds were measured along the contact normal.

What the audit shows:

* **The scorecard is fair to hooks.** Once a hook touches the head or torso
  it scores 80–88 % of the time, like a straight. Lateral impacts are not
  penalized, so the scoring rule stays as it is.
* **The bottleneck is reaching the target.** A scripted hook touches the
  target about 8 times in 160 s, against 64 for a straight.
* **A high guard stops straights completely** (no torso contact in 160 s),
  while hooks still land around it. That is the only matchup where a hook
  beats a straight.
* In learned bouts the champion's straights still land, and it throws no
  hooks that touch a target, even though it holds the high guard 35–56 % of
  the time. A straight's arm leaves the high guard while it punches, and the
  guard toggles often, so the straight still finds openings and the policy
  has little reason to hook.

### Fine-tune against a held high guard

The audit suggested an opening: the scripted attacker's straights never touch
a held high guard, but its hooks do. `boxer.native_finetune
--high-guard-fraction` makes that share of the scripted opponents hold the
high guard all bout. Run: `runs/tactics-highguard`, from the level-hook
champion, 3M native steps, pool = level-hook and guard champions, half the
bouts scripted, 60 % of those with a held high guard (30 % of all bouts).

The learner's hook share fell instead of rising: about 6 % of punches in the
first 0.5M steps, 1–2 % by 1.5M, 0–1 % at the end.

Native gate against the level-hook champion:

| Candidate | W–L–D (16 bouts) | Clean punches per bout |
| --- | --- | --- |
| 0.75M | 6–8–2 | 10.8 : 11.7 |
| 1.5M | 7–4–5 | 12.4 : 12.0 |
| 2.25M | 9–5–2 | 13.1 : 13.2 |
| **3.0M** | **10–4–2** | **17.1 : 13.2** |
| **3.0M, 48 bouts with fresh seeds** | **27–17–4** | **16.0 : 14.0**; falls 2 vs 2 (confirmation) |

Against scripted opponents (8 bouts of 20 s each, learner in red):

| Policy | vs held high guard: W–L, clean, hooks | vs normal guard: W–L, clean, hooks |
| --- | --- | --- |
| level-hook champion | 8–0, 43 : 9, 5 hooks | 8–0, 93 : 25, 3 hooks |
| high-guard 0.75M | 8–0, 73 : 17, 0 hooks | 8–0, 100 : 26, 0 hooks |
| **high-guard 3.0M** | **8–0, 113 : 16, 0 hooks** | **8–0, 117 : 25, 0 hooks** |

The 3.0M checkpoint becomes `checkpoints/tactics`; the level-hook champion
moves to `checkpoints/tactics-levelhook-v1`.

**The learner beats the held high guard with straights, not hooks.** The
scripted attacker's straights never reach that guard, but the learned
policy's straights land 113 times in 160 s. A straight reaches its target in
0.18 s, a hook in 0.26 s, so once footwork finds an opening the straight is likely
the better punch. These runs did not measure how its footwork and aim open
the guard. Under this scorecard and these moves, hooks are not worth
learning. Making them matter would mean changing the game: e.g. a guard that
actually covers the torso front so only angled punches get through, or a
richer target (head shots within reach at closer range).
