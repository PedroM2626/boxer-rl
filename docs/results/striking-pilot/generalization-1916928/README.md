# Bounded independent reset and cross-policy audit

See `docs/striking-generalization-audit.md` for reviewed evidence and exact cap.
Latest source 670bed08 has two original complete passes but fails fresh 20M
neutral-0. No training/promotion is allocated. Four disjoint 21M full groups
(16 seeds, both swaps, all three profiles) plus two saved opposing policies
6bf3273a/8003a7bf on 22M seeds with both evaluated-corner assignments produce
128 episodes / maximum 64,000 evaluation controls. Each actor uses its own
weights/normalizer over unchanged unassisted physics. Mixed-policy reports
cannot certify ordinary mirror gates. All group outcomes must be archived.
Diagnostic PPO stays 1,918,208; formal evaluation ceiling 1,360,474.

Live state: `/home/pedro/boxing-runs/striking-generalization-1916928/audit.json`.
Launcher: `tools/start_striking_generalization_audit.ps1`. Preserve monitoring.

## Native evaluation integration

Fourteen tests and Python/Bash/PowerShell syntax checks pass. Same-policy
opponent routing preserves native observations, actions, rewards, qpos/qvel and
scores exactly; the evaluated corner's own action remains intact. Nonfinite or
incompatible policies are rejected, and mixed-policy reports cannot certify
ordinary mirror gates. Two initial complete cross-policy reports (16 episodes)
are independently checked across both corner assignments, with exact source/
opponent hashes, finite states, original controller markers and empty native
logs. They are archived with validation and a historical running snapshot.
These partial integration results do not replace the full remaining audit.

## Completed independent review

`analysis.json` and `audit.json` now archive the full 128-episode result.
All four disjoint full groups fail; neutral-0 / neutral-1 / mirror totals are
25/32 / 26/32 / 24/32. Surviving mirror episodes have 23 red and 29 blue valid
events and 12 two-way exchanges. Cross-policy pair survival is 24/32, with three
evaluated-policy and five opposing-policy first falls. Original repeatability
remains two distinct passes; there is no promotion or main adoption.

Actual evaluation is 55,067 controls, bringing formal evaluation to 1,351,541.
All 16 native error logs are verified empty; reports retain every failed trial.
Foundation histories/payloads, main hash, latest/frozen hashes and finite learned
state are independently rechecked. `validation.json` and `launch-state.json`
above are historical integration snapshots, not current job status. The next
bounded operation is documented in `docs/striking-motor-floor-audit.md`.
