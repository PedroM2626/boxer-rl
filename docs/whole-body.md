# Whole-body policy (2026-11-11)

**Current strongest boxer:** `checkpoints/whole-body`, run 6 (see
[starts from the end of a referee's count](#starts-from-the-end-of-a-referees-count-run-6-promoted)).
Run 5, which it replaced, is kept as `checkpoints/whole-body-v2`, and run 3
as `checkpoints/whole-body-v1`
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

## Starts from the end of a referee's count (run 6, promoted)

With the [referee's count](getup.md#in-bouts-the-referees-count), boxers
often fell again just after resuming: they restart from where the get-up
policy left them, a stance no training bout began from. Run 6 trains from
there. `python -m boxer.getup_starts` builds a bank of such states in native
MuJoCo: one robot lying (the get-up stage's lying starts), the other
standing in guard on its motor, the get-up policy run through the 8 s
count, then both set 2 m apart facing each other
(`PairedFoundation.separate`), as `spar_match --getup --count-separate 2`
resumes a bout. 395 of 405 counts ended with the boxer up and steady; the
bank keeps 400 (`checkpoints/getup-starts/getup_starts.npz`, 0.6 MB).
`native_finetune --start-bank` starts a share of training bouts
(`--start-bank-fraction`, 0.25 here) from a banked state, the learner in the
risen corner three times in four.

Everything else is run 5's (`runs/wb_ppo6.sh`): from run 5, 1M steps, seed
306. The gate picked 0.5M and 0.25M for the long check; 0.5M fell 3 times
against the champion and was rejected, 0.25M passed every check (16–0 against
run 5, the champion and the striker; confirmation 45–1–2, 2 falls).
The training log shows more KL spikes than run 5 (46 of the first 359
iterations above 1, against 48 of 1954), most likely from the unusual
banked states; the clipped updates did not harm the result.

Full evaluation (32 bouts of 120 s, seeds 0–15, corners swapped, ring,
knockdown rules), with the same run without the bank as a control (same
seed and settings; its gate rejected every candidate, so its 0.25M step was
evaluated directly):

| Opponent | Run 5 (`whole-body-v2`) | Run 6 without the bank, 0.25M | Run 6 (`whole-body`) |
| --- | --- | --- | --- |
| champion stack | 32–0, 419 : 96, knockdowns 8 : 3 | 32–0, 417 : 124, 18 : 1 | 32–0, 427 : 143, **2 : 0** |
| striker | 32–0, 445 : 50, 4 : 0 | 32–0, 413 : 74, 1 : 1 | 32–0, 374 : 43, 1 : 0 |
| champion, count and 2 m apart (`--getup --count-separate 2`) | 29–3, 10 : 2, 3 KOs | — | **32–0, 435 : 148, 0 : 0** |
| run 5, head to head | — | — | **32–0, 348 : 199, 7 : 17** |
| itself | 473 : 472, 0 knockdowns | — | 392 : 381, 8 : 8 |

Run 6 falls a quarter as often as run 5 against the champion, and beats
run 5 32–0. The control shows the bank is what did it: without it, the same
seed and step fall 18 times against the champion. Against the champion with
the count, run 6 was never knocked down, so no count happened. The bank did
not teach it to box on after getting up so much as to stay up from the
unusual stances the bank starts it in. The cost is in mirror bouts: 8
knockdowns each against itself, where run 5 had none
([runs 7 to 11](#falls-against-itself-runs-7-to-11-rejected)). Run 6 was promoted as
the strongest boxer; run 5 is kept as `checkpoints/whole-body-v2`.

```bash
bash runs/getup_starts.sh   # python -m boxer.getup_starts --output runs/getup_starts.npz --states 400
bash runs/wb_ppo6.sh        # native_finetune ... --start-bank runs/getup_starts.npz --start-bank-fraction .25
bash runs/wb_eval6.sh       # the evaluation above
```

## Falls against itself (runs 7 to 11, rejected)

Run 6's 16 knockdowns in 32 mirror bouts come late (14 of 16 after 60 s,
beyond the 20 s training episodes), with more body contact than run 5's
(1.5 s per bout against 0.4 s). Mirror bouts, 32 of 120 s (seeds 0–15,
corners swapped), total knockdowns:

| Policy | vs itself | vs champion (its own) |
| --- | --- | --- |
| run 5 (`whole-body-v2`) | 0 | 8 |
| run 6 without the bank, 0.25M | 2 | 18 |
| **run 6** (`whole-body`, bank 25 %) | 16 | 2 |
| run 7, 0.25M | 12 | — |
| run 8, 0.25M (bank 10 %) | 8 | 5 |

The bank that keeps run 6 up against the champion is also what makes it
fall against itself; the control without it hardly falls in the mirror.
Two attempts to keep the first without the second:

- **Run 7**: from run 6, run 6 and run 5 in the pool, fall penalty 150
  against every opponent (`runs/wb_ppo7.sh`, seed 307). Rejected by the
  gate: 14 and 19 knockdowns against run 6 in 16 long bouts at 0.5M and
  0.75M. Its 0.25M step, not checked by the gate, still falls 12 times
  against itself. Self-play in the pool hurt again, as in run 4.
- **Run 8**: run 6 with 10 % of bouts from the bank, not 25 %
  (`runs/wb_ppo8.sh`, from run 5, seed 306). 0.25M passed the gate
  (confirmation 46–1–1) and halves the mirror falls (8), but falls 5 times
  against the champion and 14 against the striker (5 : 9), and loses to
  run 6 0–32 (284 : 382 clean punches). Not promoted.

- **Run 9**: run 6's recipe (from run 5, bank 25 %) with another seed (309),
  8 candidates, and a mirror check added to the long gate
  (`--mirror-gate 2`: at most 2 knockdowns in 16 bouts of 120 s against
  itself, seeds 24–31, corners swapped; `tactics_gate.mirror_stable`). Five
  candidates passed the short gate against run 5 (14–2 to 16–0), and the four
  sent to the long gate all beat run 5 16–0 or 15–1 over 120 s. But all four fell
  16 to 21 times in 16 long bouts against the champion (run 6's selected step:
  2), so none reached the mirror check (`runs/wb_ppo9.sh`). Rejected.
- **Run 10**: from run 6, its recipe with 60 s training episodes instead of
  20 s, since the mirror falls come after a minute (`--episode-seconds 60`,
  `--mirror-gate 4`, seed 310, `runs/wb_ppo10.sh`). It only drew level with
  run 6 in the short gate (best 10–4–2 at 0.5M and 0.625M; 2–13 and 3–12 by
  the end). In 16 long bouts the four candidates fell 5 to 14 times against
  the champion, and 3 to 18 times against run 6 (run 6: 3 to 13). None
  reached the mirror check. Rejected.
- **Run 11**: from run 6, its recipe plus starts taken just before its own
  falls against itself. `boxer.mirror_starts` played run 6 against itself
  (100 bouts of 120 s) and kept the state 0.5 to 3 s before every fall:
  333 states (`checkpoints/mirror-starts`). `--mirror-bank` started 15 % of
  training bouts from one, against run 6, the learner in the corner that
  fell three times in four (`runs/wb_ppo11.sh`, seed 311). The gate
  rejected it: the later candidates fell 14 to 32 times against run 6 in 16
  long bouts; 0.125M passed against run 6 and the champion but fell 6 times
  against the striker. Over 32 bouts that step beats the champion 32–0
  (4 : 2 kd) and the striker 32–0 (2 : 1), but against itself it falls
  far more than run 6 (22 + 34 = 56 knockdowns, against 16), and it falls 21
  times against run 6 (run 6: 7). Learning from those states against a
  frozen run 6 made it a worse partner for itself. Rejected.

Run 9 shows that run 6's low knockdowns against the champion are partly its
seed: the same recipe with another seed falls as often as the control
without the bank. Run 6 stays the strongest boxer: it wins every matchup
that has an opponent other than itself.

## A boxing rhythm (run 12, kept as `whole-body-style`)

Runs 3 to 6 won by volume: from the clone to run 6 the whole body went from
2.2 to 3.6 clean punches a second, and from 29 % to 6 % of the time in pauses
(1 s or more between its clean punches). The champion stack lands 1.2 a
second and pauses half the time, with its high guard up 44 % of it. It looks
like boxing; runs 3 to 6 look like a brawl. The reward caused it
(`REWARD`): a punch landed paid 2 and one taken cost 1.5, so trading blows
paid both boxers, and a bonus paid every second spent at punching range.

Run 12 (`runs/wb_ppo12.sh`, from the clone, run 6's recipe otherwise, seed
312) changes the reward only, not the scorecard:

- `--taken-penalty 2.5`: trading blows no longer pays;
- `--in-range-reward 0`: no pay for standing at punching range;
- `--hit-cooldown .6`: at most one rewarded clean punch per 0.6 s (the bout's
  scorecard still counts every one);
- `--guard-reward 1`: per second, for arm targets near the guard or high
  guard (`guard_closeness`);
- `--style-gate 2.0 .25`: the long gate also rejects a candidate landing
  more than 2 clean punches a second or pausing less than 25 % of the time
  (`tactics_gate.bout_style`).

The gate rejected every candidate: 0.875M kept the style (1.5 and 2.0 a
second, 49 % and 34 % paused) but fell 6 times against the champion in 16
long bouts. That step is kept as `checkpoints/whole-body-style` for its
rhythm. 32 bouts of 120 s:

| Matchup | W–L–D | Clean punches | Knockdowns | Its clean punches a second | Its time in pauses |
| --- | --- | --- | --- | --- | --- |
| vs champion | 32–0–0 | 177 : 52 | 10 : 0 | 1.47 (run 6: 3.56) | 48 % (run 6: 6 %) |
| vs striker | 32–0–0 | 159 : 59 | 13 : 2 | 1.32 (3.12) | 55 % (13 %) |
| vs itself | — | 85 : 92 | 6 : 4 | 0.71 (3.26) | 75 % (13 %) |
| vs run 6 | 0–31–1 | 111 : 329 | 34 : 6 | 0.93 | 77 % |

It boxes at the champion's pace and falls less against itself than run 6
(10 knockdowns in 32 bouts, against 16), but it falls more against everyone
else, and run 6's volume beats it every time. Videos:
`fights/style_vs_champion_60s.mp4` and `fights/style_vs_itself_60s.mp4`.

## Blocks and slips (runs 13 and 14, rejected)

The scorecard now also counts the punches a boxer stops
(`PairedFoundation`'s `blocked`, `defense.MissTracker`):

- *blocked*: a fist lands on the opponent's arms or gloves under the clean
  punch rules;
- *clashed* (since 2026-11-15): the same contact, but the glove it hits is
  itself punching: moving toward the attacker's torso at 0.3 m/s or more,
  relative to its own torso. Clashes no longer count as blocks;
- *missed*: a fist drives at the head or torso (closing at 1 m/s or faster,
  within 0.35 m), comes within 0.10 m of it and pulls back without touching
  a target or the guard;
- *evaded*: a miss during which the target moved away from the fist at
  0.3 m/s or faster (a slip or a step back).

`spar_match` reports them per bout (`defense`), with the defended share =
(blocked + evaded) / (landed + blocked + evaded). Before any training for it
(16 bouts of 120 s, `runs/defense_probe.sh`), boxing the champion, run 6
defends 61 % of the punches thrown at it and evades 1.4 a bout;
`whole-body-style` defends 83 % and evades 4.8.

Run 13 (`runs/wb_ppo13.sh`, from `whole-body-style`, run 12's style terms,
seed 313) pays 0.5 per punch blocked and 2 per punch evaded
(`--block-reward .5 --evade-reward 2`). The gate rejected every candidate:
all fell 4 to 17 times in 16 long bouts against the champion. 0.75M, the
stylish one that fell least, over 32 bouts of 120 s (`runs/wb_eval13.sh`):

| Matchup | W–L–D | Clean punches | Knockdowns | Punches a second / pauses | Taken / blocked / evaded a bout | Defended |
| --- | --- | --- | --- | --- | --- | --- |
| vs champion | 32–0–0 | 187 : 24 | 5 : 2 | 1.56 / 48 % | 24 / 255 / 8.2 | 92 % (champion 55 %) |
| vs striker | 31–1–0 | 147 : 30 | 29 : 3 | 1.25 / 60 % | 30 / 263 / 2.7 | 90 % |
| vs `whole-body-style` | 27–5–0 | 96 : 63 | 19 : 33 | 0.89 / 70 % | 63 / 167 / 2.0 | 73 % (style 61 %) |
| vs run 6 | 0–32–0 | 47 : 133 | 58 : 12 | 0.43 / 90 % | 133 / 139 / 5.8 | 52 % (run 6 82 %) |
| vs itself | — | 37 : 32 | 32 : 60 | 0.48 / 84 % | 32 / 95 / 1.1 | 75 % |

The defense rewards work as far as the scorecard goes: against the champion
it takes 24 clean punches a bout instead of run 12's 52, and evades almost
twice as often, keeping the champion's pace. But it stays up worse: 92
knockdowns in 32 bouts against itself (run 12: 10), 29 against the striker,
58 against run 6. Evasions stay rare (1 to 8 a bout, against 100 to 260
blocks), so the guard does nearly all the defending; slipping or stepping
back at speed seems to cost the legs their balance. Videos:
`fights/defense_vs_champion_60s.mp4` and `fights/defense_vs_itself_60s.mp4`.

Run 14 (`runs/wb_ppo14.sh`, seed 314) pays a block or an evasion only once
the learner has stayed up 1.5 s after it (`--defense-hold 1.5`; a fall first
forfeits it), raises the fall penalty against whole-body opponents from 60
to 100, adds `whole-body-style` to the pool and checks the mirror bout
(`--mirror-gate 6`). Its candidates fell 2 or 3 times in 16 long bouts
against the champion, but the gate still rejected all of them: those that
fell least landed 2.0 to 2.5 clean punches a second, over the style limit.
0.25M (2.02 a second), over 32 bouts of 120 s (`runs/wb_eval14.sh`):

| Matchup | W–L–D | Clean punches | Knockdowns | Punches a second / pauses | Taken / blocked / evaded a bout | Defended |
| --- | --- | --- | --- | --- | --- | --- |
| vs champion | 32–0–0 | 248 : 67 | 6 : 0 | 2.06 / 33 % | 67 / 232 / 8.3 | 78 % |
| vs striker | 31–1–0 | 163 : 46 | 9 : 4 | 1.36 / 55 % | 46 / 235 / 3.1 | 84 % |
| vs `whole-body-style` | 28–3–1 | 129 : 43 | 17 : 4 | 1.10 / 63 % | 43 / 252 / 1.7 | 86 % |
| vs run 6 | 0–32–0 | 121 : 287 | 51 : 7 | 1.07 / 74 % | 287 / 150 / 5.6 | 35 % |
| vs itself | 14–16–2 | 78 : 78 | 20 : 12 | 0.67 / 75 % | 78 / 169 / 0.7 | 68 % |

It falls a third as often as run 13 against itself (32 knockdowns, against
92) and against the striker (9, against 29), but it buys that with volume:
faster and fewer pauses, and fewer of the champion's punches defended (78 %,
against run 13's 92 % and `whole-body-style`'s 83 %). Neither run defends
better and stays up as well as `whole-body-style` (10 knockdowns against
itself). Evasions stay at 1 to 8 a bout.

## What is open

- Mirror bouts: run 6 falls 16 times in 32 bouts against itself (run 5: 0),
  mostly after a minute. Neither self-play (run 7) nor a smaller bank share
  (run 8) fixed it without losing elsewhere. Run 9 repeated run 6's recipe
  with the mirror bout in the long gate (`--mirror-gate`), but with its
  seed it fell 16–21 times against the champion. Run 10 trained on 60 s
  episodes from run 6 and fell 5–14 times against the champion. Run 11
  started from states just before run 6's own mirror falls, against run 6,
  and fell 56 times against itself. Five attempts have not fixed it;
  the mirror falls are left open.
- A boxing rhythm that also stays up: run 12 boxes at the champion's pace
  but falls 10 times against it in 32 bouts, and loses every bout to run 6.
- Defense that also stays up: run 13 blocks and evades more (92 % of the
  champion's punches defended) but falls 92 times against itself in 32
  bouts; run 14, paying defense only after 1.5 s standing, falls 32 times
  but defends less (78 %). Slips are rare. Clashes are now counted apart
  from blocks; re-scored, `whole-body-style` still defends 83 % of the
  champion's punches (238 blocks and 27 clashes a bout).
- Getting up inside the whole-body policy itself. Now a separate get-up
  policy takes over while the boxer is down ([get-up](getup.md)).
