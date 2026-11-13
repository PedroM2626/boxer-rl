# Whole-body policy (2026-11-11)

**Current strongest boxer:** `checkpoints/whole-body`, run 5 (see
[self-play with a heavier fall penalty against stacks](#self-play-with-a-heavier-fall-penalty-against-stacks-run-5-promoted)).
The run-3 policy it replaced is kept as `checkpoints/whole-body-v1`
([long episodes, tilt cost and leg anchor](#long-episodes-tilt-cost-and-leg-anchor)).

Every boxer before this was a hierarchy of separately trained parts. A brain
(tactics or striker) sent a footwork command to a leg motor and set the
arms, and the motor turned the command into leg and waist targets. Each part
trained with the others frozen, and changing one broke another: when the
striker's arm style changed, its legs began to fall
([striker](striker.md#arm-motion-cost-2026-11-03)). The whole-body policy
(`boxer/whole_body.py`) removes the hierarchy: one network sets every joint
target the robot has, every control step.

## Interface

| | Size | Contents |
| --- | --- | --- |
| Observation | 174 | the striker observation (74: opponent, ring, both arms' motion) and the motor's proprioception without the footwork command (100: pelvis velocity, gyro, gravity, joint positions and velocities, last actions, gait clock) |
| Action | 23 | 15 leg and waist targets in the motor's units, and 8 shoulder and elbow targets in the striker's units (the pose box around every scripted pose, moved at most `striker.RATE` per step, wrists held) |

The network is 512–256–256 for both the policy and the value function.
`spar_match` recognizes a whole-body checkpoint by its sizes (174 → 23)
wherever it takes tactics. Such a corner has no motor: its leg and waist
targets replace the motor's row for that robot. `native_finetune --whole-body`
trains one, and a whole-body policy can also be an opponent in the native
pools.

## Imitation: one network for striker and legs

`python -m boxer.whole_body_train collect` plays 800k native ring steps with
the striker (`checkpoints/striker`) on its legs (`checkpoints/spar-native`). The
opponents are the champion, the high-guard champion, both strikers, and a
scripted brain in 30 % of bouts. At every step it records the whole-body
observation and what the two layers did together: the motor's 15 executed
leg targets and the striker's 8 arm targets. `clone` fits the policy mean to
those labels, with leg errors weighted 4× because they cost falls, and the
value function to discounted native returns. It runs for 40 epochs, starting
the policy scale at 0.1. Held-out mean squared error is 0.0014 for the legs
and 0.0010 for the arms.

The clone (`checkpoints/whole-body-clone`) boxes end to end with no motor.
In 32 bouts of 120 s (seeds 0–15, corners swapped), the clone on no motor
against each opponent on its own legs:

| Opponent | Record | Clean punches per bout | Knockdowns (clone : opponent) |
| --- | --- | --- | --- |
| champion stack | 32–0 | 259 : 88 | 6 : 2 |
| striker (its teacher) | 16–16 | 256 : 252 | 13 : 8 |

The clone matches its teacher's punching but falls more often. The striker
itself goes 1 : 4 against the champion and suffers 6 knockdowns in 32
striker brawls, as measured below. Imitation errors in the legs add up over
120 s.

## Native PPO: more punches, more falls

Two runs, each 2M native transitions starting from the clone. Opponents:
champion, high-guard champion, both strikers, and scripted bouts (30 %).
Arm-motion cost 0.05, minimum policy std 0.05, gate on the last four
checkpoints. The long check ran against the clone, the champion tactics and
the striker.

| Run | Settings | Long check (120 s, seeds 24–31) | Gate |
| --- | --- | --- | --- |
| 1 | lr 5e-5, 4 epochs, fall penalty 30 | 1M: 12 : 5 knockdowns vs the clone; 1.5M: 11 : 10 vs the clone, then 15 : 0 vs the champion (300 clean punches per bout to 20) | rejected |
| 2 | lr 1.5e-5, 2 epochs, fall penalty 60 | 1M: 7 knockdowns vs the clone (431 clean punches per bout to 109); 1.5M: 16 vs the clone | rejected |

In run 1 the updates were large: KL 0.1–0.25 per iteration, with 64 % of
probability ratios outside the PPO clip. Run 2 brought both down (KL 0.02,
21 % clipped). Both runs show what the arm-only experiments showed. PPO in
20 s training episodes raises hits by 1.5–2× and wins the 20 s gate bouts
(14–2, 15–1 against the clone), and the falls appear only in long bouts.
The long-bout gate is what caught it. Neither run was promoted.

At 50 Hz a 1000-step training episode lasts 20 s, and with γ = 0.97 the
return looks about 0.7 s ahead. A fall builds up over longer than that, so
the only signal against it was the terminal fall penalty.

## Long episodes, tilt cost and leg anchor

Run 3 kept run 2's settings and changed what training sees and values:

- `--episode-seconds 120`: training episodes as long as the judged bouts.
- `--gamma .99`: returns look about 2 s ahead instead of 0.7 s.
- `--tilt-cost 5`: every step the learner loses `dt · 5 · (2 |g_xy|² + 0.15 |ω_xy|²)`.
  `g_xy` is the horizontal part of gravity in the torso frame and `ω_xy` the
  roll and pitch rate. These are the leg motor's own upright terms
  (`motor_native.REWARD`), 5 times stronger (`native_finetune.tilt_penalty`).
- `--leg-anchor 20`: the PPO loss adds 20 × the mean squared gap between the
  leg outputs' means and those of the starting clone
  (`native_finetune.leg_drift`). The arms and every scale are free.

During training the leg drift stayed around 0.001 and KL around 0.005–0.03.
Hits per 1000 steps rose from 23 to 66 while the tilt penalty fell from 0.14
to 0.13 per step.

The gate selected the 1M checkpoint, the first whole-body checkpoint it
accepted. Its long check (120 s, seeds 24–31) found it stable against all
three opponents: 1 : 1 knockdowns vs the clone, 0 : 0 vs the champion
tactics, 2 : 2 vs the striker. Confirmation: 47–0–1 over 48 bouts.

Evaluation, 32 bouts of 120 s (seeds 0–15, corners swapped). The
whole-body policy has no motor; each opponent plays on its own legs:

| Bouts | Record | Clean punches per bout | Knockdowns (whole body : opponent) |
| --- | --- | --- | --- |
| **vs champion stack** | **32–0** | **361 : 81** | **1 : 0** |
| vs striker (`checkpoints/striker` on `spar-native`) | 31–1 | 362 : 114 | 2 : 2 |
| vs itself | — | 389 : 391 | 4 in all |
| *for comparison:* striker vs champion | 32–0 | 256 : 87 | 1 : 4 |
| *for comparison:* striker vs itself | — | 234 : 252 | 20 in all |

It lands 41 % more clean punches than the striker against the same
champion. Its own brawls fall 4 times in 32 bouts, where two strikers fall 20
times. Promoted as `checkpoints/whole-body`
([60 s video](../fights/whole_body_vs_champion_60s.mp4)).

The changes were made together; the ablations below separate them.

Commands:

```bash
python -m boxer.whole_body_train collect --output runs/whole-body-data
python -m boxer.whole_body_train clone --data runs/whole-body-data --output runs/whole-body-clone
python -m boxer.native_finetune --whole-body --init-from runs/whole-body-clone/checkpoint   --output runs/whole-body-native3 --motor checkpoints/spar-native --ring --arm checkpoints/arm-skill   --pool checkpoints/tactics checkpoints/tactics-highguard-v1 checkpoints/striker checkpoints/striker-v1   --scripted-fraction .3 --high-guard-fraction .6 --steps 2000000 --checkpoint-every 500000   --min-std .05 --learning-rate 1.5e-5 --epochs 2 --fall-penalty 60 --arm-motion-cost .05   --episode-seconds 120 --gamma .99 --tilt-cost 5 --leg-anchor 20   --native-gate 4 --long-gate 2 --long-gate-opponents checkpoints/tactics checkpoints/striker
bash tools/run_spar.sh boxer.spar_match --ring --seconds 120 --knockdowns 3 --seeds 0 1 2 3 --swap   --checkpoint checkpoints/spar-native --red-tactics checkpoints/whole-body   --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill
```

In `spar_match`, `--checkpoint` still names a motor for that corner. A
whole-body corner overrides the motor's leg targets every step, so the motor
does not move it.

## Ablations: the leg anchor and the horizon carry it

Each run below drops one of run 3's four changes and trains 1M steps from
the clone with run 3's seed (302); `full-seed303` keeps all four with
another seed. No gate: every run is judged at 1M steps over 32 bouts of
120 s (seeds 0–15, corners swapped) against the champion and against
itself. Training hits and tilt are the last 20 progress rows.

| Run | Dropped | vs champion | Clean per bout | Knockdowns (wb : champ) | Own brawls: knockdowns | Training hits / 1000 |
| --- | --- | --- | --- | --- | --- | --- |
| run 3 (promoted) | — | 32–0 | 361 : 81 | 1 : 0 | 4 | 66 |
| full-seed303 | — (other seed) | 31–0–1 | 376 : 81 | 0 : 0 | 2 | 69 |
| no-long | 120 s episodes (20 s) | 32–0 | 352 : 58 | 2 : 2 | 4 | 56 |
| no-tilt | tilt cost | 32–0 | 382 : 113 | 6 : 5 | 10 | 67 |
| no-gamma | γ 0.99 (0.97) | 32–0 | 326 : 74 | 7 : 5 | 38 | 66 |
| no-anchor | leg anchor | 31–1 | 404 : 28 | 14 : 1 | **108** | 65 |

- **The leg anchor is essential.** Without it the legs drift from the
  clone's gait. The policy lands the most punches against the champion (404
  to 28) and loses one bout on knockdowns (14 : 1); two of them fall 108
  times in 32 bouts, and the brawls barely produce clean punches (139 per
  bout).
- **The longer horizon matters next.** With γ 0.97 the own brawls fall 38
  times.
- **The tilt cost helps a little** (10 knockdowns without it, against 4).
- **Long episodes do not matter** once the other three are in place: 20 s
  episodes give the same 4 knockdowns, with fewer training hits (56 against
  66).
- The second seed reproduces the result (2 knockdowns in own brawls).

## Continued training with itself in the pool (rejected)

Run 4 started from `checkpoints/whole-body`, with the same settings as run 3
and the policy itself added to the opponent pool. The leg anchor moved with
it: it now holds the legs near the promoted policy's legs. 2M steps, seed 304.

Every checkpoint beat the promoted policy in the 20 s gate (13–2–1 at 0.5M,
then 15–0–1, 15–0–1, 15–1). Over 120 s it beat it 16–0 (443 : 280 clean
punches per bout at 1.5M) with no falls. But against the champion stack it
fell 9 times in 16 long bouts at 1M and 3 times at 1.5M, where the champion
fell 0 times; the gate allows 2. Nothing was promoted.

Training against itself makes it stronger against itself and less steady
against the champion's different style. Keeping the champion stacks and
the strikers in the pool did not prevent this within 2M steps.

## Self-play with a heavier fall penalty against stacks (run 5, promoted)

Run 4 fell against the champion because nothing in self-play made a fall
against a stack cost more than a fall against itself. Run 5 adds that:
`native_finetune --stack-fall-penalty 150` charges 150 for the learner's fall
against an opponent that is not a whole-body policy (the champion stacks and
the strikers). Falls against the whole-body policy still cost the usual 60
(`NativeBout.learner_fall_penalty`). The other settings are run 3's, with
three changes:

- 20 s episodes, since the ablation showed 120 s episodes add nothing;
- 4 workers;
- seed 305, starting from run 3.

The opponent pool is run 3 itself, two champion stacks and two strikers.
Training ran 2M steps.

Gate (`runs/whole-body-native5/gate`):

| Step | 20 s gate vs run 3 | 120 s check vs run 3 (own : its knockdowns) | 120 s vs champion | 120 s vs striker |
| --- | --- | --- | --- | --- |
| 0.5M | 13–0–3 | 16–0, 418 : 159 clean, 0 : 0 | 16–0, 1 : 0 | 15–1, 2 : 0 |
| 1M | 16–0 | 16–0, 422 : 209, 5 : 1 (rejected) | — | — |
| 1.5M | 14–2, 2 : 0 | not checked | — | — |
| 2M | 12–4, 4 : 1 | not checked | — | — |

The 0.5M checkpoint passed every long check and the 48-bout confirmation
(43–1–4, 2 falls). Beyond 0.5M the policy got steadily worse again.

Full evaluation (32 bouts of 120 s, seeds 0–15 with corners swapped, ring,
knockdown rules), against run 3's numbers:

| Opponent | Run 3 (`whole-body-v1`) | Run 5 (`whole-body`) |
| --- | --- | --- |
| champion stack | 32–0, 361 : 81 clean, knockdowns 1 : 0 | 32–0, 419 : 96, knockdowns 8 : 3 |
| striker | 31–1, 362 : 114, 2 : 2 | 32–0, 445 : 50, 4 : 0 |
| itself | 389 : 391, 4 knockdowns | 473 : 472, 0 knockdowns |
| run 3, head to head (gate, 16 bouts) | — | 16–0, 418 : 159, 0 : 0 |

Run 5 wins everything, lands 15–25 % more clean punches, and beats run 3
16–0. It was promoted as the strongest boxer. The cost is falls against the
champion stack: 8 knockdowns in 32 bouts, against run 3's 1. The heavier
penalty kept those falls within the gate's long check (1 in 16 bouts on
seeds 24–31), but not on seeds 0–15. Against itself and the strikers it is
steadier than run 3. With the [get-up policy](getup.md) a knockdown no longer
needs a restart, so these falls cost points rather than the bout.

```bash
bash runs/wb_ppo5.sh     # native_finetune --whole-body ... --stack-fall-penalty 150 --episode-seconds 20 --workers 4
```

**Episode length.** 20 s episodes are the recommended default for native
fine-tuning, and `native_finetune --episode-seconds` defaults to 20. The
ablation (no long episodes: 4 knockdowns, the same as 120 s) and run 5 both
show that the long episodes are not what keeps the legs steady. The 120 s
gate check is what catches long-bout falls.

## What is open

- Self-play that keeps the champion steady over more seeds: the fall penalty
  against stacks helped (run 5 passed the gate where run 4 did not), but run 5
  still falls 8 times in 32 bouts against the champion. Next to try: the long
  check inside the training loop, or a larger gate sample.
- Getting up inside the whole-body policy itself. Now a separate get-up
  policy takes over while the boxer is down ([get-up](getup.md)).
