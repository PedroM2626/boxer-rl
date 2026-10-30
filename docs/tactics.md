# Learned boxing tactics (self-play PPO)

Date: 2026-10-30. Builds on the [sparring motor](spar-motor.md).

## Architecture

Each boxer is three layers. Only the top one is trained here.

| Layer | What it does | Source |
| --- | --- | --- |
| Tactics (PPO, 50 Hz) | 5 numbers: left punch, right punch, aim (body ↔ head), advance/retreat, circle left/right | `boxer/tactics_env.py`, `boxer/tactics_train.py` |
| Punch primitives | Launch latches hand, aim, ±0.06 rad jitter and 0.9–1.15× speed; 0.18 s extension, 0.26 s retraction | `punch_arms` / `spar_match.PunchArm` |
| Sparring motor (frozen) | Legs and waist follow footwork commands and keep balance while punching or hit | `checkpoints/spar` |

`decode` (shared by the batched env and native play) turns tactics into a
motor command. Advance is up to 0.4 m/s forward or 0.3 m/s back, lateral up to
0.2 m/s, and turning toward the opponent is automatic. A punch request is a
positive left/right output while the arm is free.

Tactics observation (26 floats, `observe`): opponent position and distance in
the pelvis frame, distance minus reach, which way the opponent faces, both
velocities, both uprightness and height, both punch states (phase, active,
hand, aim: the opponent's punch is visible as it starts), round clock and
bearing.

## Batched training environment

`TacticsEnv` puts both robots in one MJX/Warp world: 72 qpos, 158 contact pairs.
The physics matches native `PairedFoundation` (same model builder, 4 ms
substeps, same motor observation). `build_pair(strike_sensors=True)` adds
read-only contact sensors for fist-on-head, fist-on-torso and chest-to-chest.

* **Clean hit, training proxy:** the first fist-on-target touch of a punch,
  during extension, with the fist closing on the target at ≥ 0.5 m/s. Head
  1.0, torso 0.6. All reported results use the native impact scorecard
  instead.
* **Reward (red learner):** 2 × own hits − 1.5 × hits taken, −10 for falling,
  +5 for the opponent falling, +0.3/s near punching range (0.62 m), −1/s in a
  clinch, −0.02 per punch thrown.
* **Episode:** 20 s or the first fall. Spawns are 0.9–1.7 m apart with random
  global yaw and facing noise, so corner colour carries no advantage.
* **Throughput:** 11.6k world-steps/s scripted-only. PPO with 512 worlds runs
  at about 25k learner steps/s on the RTX 3060 (`training/sps`), roughly
  100× native MuJoCo play.

Generation 0 trains against the scripted `RingBrain` (JAX twin
`TacticsEnv._scripted`). Generation *g* trains against a frozen copy of
generation *g − 1* in 70 % of bouts and the scripted brain in 30 %. Each
generation warm-starts from the previous weights, with fresh Adam:

```bash
bash tools/run_spar.sh boxer.tactics_train --output runs/tactics \
  --generations 3 --steps-per-generation 20000000 --worlds 512
```

## Native evaluation

Every result below is native MuJoCo (`boxer.spar_match`): free roots, no
support, native impact scorecard, 20 s or first knockdown. Each pairing plays
seeds 0–7 twice with corners exchanged (`--swap`), 16 bouts.

```bash
bash tools/run_spar.sh boxer.spar_match --red-tactics checkpoints/tactics \
  --seeds 0 1 2 3 4 5 6 7 --swap --output runs/tactics/eval/final_vs_scripted.json
```

## Results (native MuJoCo, 16 bouts per row)

"First" is the policy named first; wins, losses and draws count across both
corner assignments. Clean punches use the native scorecard.

| First | Second | W–L–D | Clean punches / bout (first : second) | Full 20 s | Falls |
| --- | --- | --- | --- | --- | --- |
| scripted | scripted (8 bouts, [spar motor](spar-motor.md)) | – | 18.6 total | 8/8 | 0 |
| gen‑0 | scripted | **16–0–0** | **20.1 : 5.7** | 16/16 | 0 |
| gen‑1 | scripted | 16–0–0 | 19.1 : 5.8 | 16/16 | 0 |
| gen‑2 | scripted | 15–1–0 | 17.6 : 4.8 | 15/16 | gen‑2 once |
| gen‑3 | scripted | 15–1–0 | 16.8 : 5.0 | 15/16 | gen‑3 once |
| gen‑0 | gen‑0 (mirror) | – | 37.6 total | 16/16 | 0 |
| gen‑1 | gen‑0 | 3–13–0 | 9.4 : 16.2 | 14/16 | one each |
| gen‑2 | gen‑0 | 4–10–2 | 11.8 : 15.8 | 14/16 | one each |
| gen‑3 | gen‑0 | **8–8–0** | 14.2 : 14.0 | 15/16 | gen‑0 once |
| v2 gen‑0 | gen‑0 | 4–12–0 | 10.9 : 17.1 | 15/16 | gen‑0 once |
| v2 gen‑1 | gen‑0 | 3–12–1 | 10.8 : 14.3 | 16/16 | 0 |
| v2 gen‑2 | gen‑0 | 4–12–0 | 10.8 : 14.4 | 15/16 | 1 |
| gen‑3 | gen‑0, **fresh seeds 8–23** (32 bouts) | 10–22–0 | 12.8 : 14.8 | 28/32 | 4 |

What this shows:

* **Learned tactics clearly beat scripted tactics.** Every generation wins
  15–16 of 16 against `RingBrain` and lands 3–4× as many clean punches.
  Learned-vs-learned bouts are twice as busy as scripted ones (37.6 vs 18.6
  clean punches per 20 s) and still go the distance.
* **Self-play progress does not transfer one-to-one from batched to native
  physics.** Gen‑1 trained only against gen‑0. In batched physics it beats
  gen‑0 82–41 over 128 bouts (`tools/batched_league.py`), yet natively it
  loses 3–13. It matches gen‑0 against the scripted brain natively, so it is
  not weaker overall: it exploits a gen‑0 behaviour that exists only in MJX/Warp
  contact dynamics. Codex's earlier backend comparison
  ([GPU recovery](gpu-training-recovery.md)) measured the same kind of
  contact disagreement.
* **Opponent pool and native-style starts close the gap gradually:** gen‑1
  3–13 → gen‑2 4–10–2 → gen‑3 8–8 against gen‑0.
* **Matching the native scoring rules in training (v2) did not help:** 4–12,
  3–12–1 and 4–12 over three generations.
  The remaining gap is more likely the physics backend than the reward.

## Champion: `checkpoints/tactics` (v1 gen‑0)

Gen‑0 (20M steps against the scripted brain, 14 minutes of training) beats
every later generation natively, including on fresh seeds (22–10 against
gen‑3), never falls against the scripted brain, and wins 16–0 against it.
It is published as `checkpoints/tactics`. Video:
[`fights/tactics_selfplay.mp4`](../fights/tactics_selfplay.mp4) (gen‑0 vs
gen‑0, seed 3: 17–24 clean punches, both standing for the full 20 s).

```bash
bash tools/run_spar.sh boxer.spar_match --red-tactics checkpoints/tactics   --blue-tactics checkpoints/tactics --seeds 3 --video runs/tactics/selfplay.mp4
```

Runs: `runs/tactics` holds v1 gen 0–3 (`gen-2-oom-killed` is an interrupted
attempt that ran out of host memory when a GPU evaluation ran alongside
training). `runs/tactics-v2` holds the native-scoring league. Native evaluation
JSONs are in `runs/tactics/eval/`.

## Limitations and next steps

* **Batched → native transfer of policy-vs-policy skill is the bottleneck.**
  Options, in order of expected value:
  1. Domain randomization in `TacticsEnv` (friction, motor strength, link
     masses, contact softness) so a policy cannot rely on one solver's
     contact details.
  2. Fine-tune with a slower native or `mujoco_warp` collector, the
     backend Codex already optimized for this model, as a final stage.
  3. Choose checkpoints by native evaluation, not by training reward.
     `spar_match --swap` already provides this.
* **Close-range rotation.** Robots drift chest to chest late in the round.
  The clinch sensor rarely fires (torso capsules barely touch), so the clinch
  penalty never acts. A distance floor or a penalty on arm/shoulder contacts
  would.
* **Move set.** Only straight punches and footwork. Hooks, uppercuts and
  guard/block primitives must first go into the `spar` generator, so the motor
  learns to balance through them.
* **Bouts end at the first knockdown.** Standing back up is not trained.
