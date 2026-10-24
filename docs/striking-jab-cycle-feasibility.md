# Explicit jab-cycle physical feasibility, version 1

**Completed review:** [all sixteen conditions failed](results/striking-pilot/jab-cycle-v1/README.md).
The prototype consumed 4,048 controls, with zero qualifying complete episodes
and no learner updates. It is rejected as a positive training target. Formal
diagnostic consumption is 2,030,927; unused allowance is not an automatic retry.
The prospective parameters/criteria below are preserved without relaxation.

Declared after reviewing all retained demonstrations and before new collection.
The [retained-target audit](results/striking-pilot/demo-motion-2080768/README.md)
exactly reproduced fifteen complete trajectories; all fifteen learner contacts
failed both inherited speed and forward references. The old generator stops
correcting after its first event. Continuing identical imitation does not supply
a validated positive target for the intended strike cycle.

Test one explicit approach/guard/extend/retract/rest prototype through the
original 17-action interface. `boxer/jab_feasibility.py` uses only seven bounded
left-arm residual slots; all command offsets, right-arm and opposing residuals
are zero. The frozen motor still walks both free bodies toward .45 engagement.
At root distance <=.75 m, cycle guard .35 s, extend .18 s, retract .25 s, rest
.30 s, repeating after contacts. A fixed own-torso guard target [.16,.16,.12] m
and opponent torso center +.10 m world height define the motion. A damped
Jacobian correction capped at .16 rad/control step is inverted through the same
.8 rad authority and .8/.2 action filter, then clipped to [-1,1]. No joint state
is set, root/leg actuator replaced, contact changed or support/force applied.

This is a scripted diagnostic controller with privileged geometry, not a trained
boxer or an adopted agent. It does not restore or update any learner checkpoint.
The goal is a physically executable positive motion target before teaching it.
The parameters are prospective and stay fixed across all trials; no seed fitting.

Allocate **at most 8,000 diagnostic controls**, sixteen 500-control episodes:
seeds 7/23/31/47, both acting corners and both spawn swaps. These known seeds
are feasibility conditions, not fresh promotion evidence. Zero PPO/supervised
updates. Start formal consumed simulation at **2,026,879**, ceiling **2,034,879**.
Diagnostic PPO remains 2,163,968; all main/BC counters remain unchanged.

Record native punch traces and all failures. Keep original independent impact
scoring; apply the separate version-1 motion references descriptively. For this
specific straight-cycle design, prospectively count a candidate only if a scored
left-hand event has >=1 m/s torso-relative onset speed, forward cosine >=.7,
>=.12 m pre-window hand excursion, a complete 500 ms recovery window with >=.06 m
return toward its pre-window hand position and <=.10 s of later contact. Travel
and return thresholds test the intended explicit stroke/retraction design; they
are not universal boxing thresholds. A feasible full episode must survive all
ten seconds and contain at least two such events. One selected successful episode
would establish only existence, not robustness or a learned policy. No gate is
weakened and no promotion follows automatically.

Stop on nonfinite/constraint/contact overflow, unbounded actions, provenance or
accounting mismatch or unexpected native stderr. Count attempts before physics;
preserve complete partial progress before recovery. A stopped/failed prototype
is not retried or modified silently. Inspect the hidden persistent WSL job at
`/home/pedro/boxing-runs/striking-jab-cycle-v1/audit.json` and `progress.json`.
The existing boxing/diagnostic locks and monitoring schedule remain in force.
Judge all sixteen conditions before any bounded redesign or new training target.
