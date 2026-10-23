# Completed matched actor-rate comparison

See `docs/striking-policy-update-investigation.md` for the predeclared contract.
Both branches start from the latest fully evaluated 8c99c2df final; the separate
1,024-control geometry probe remains consumed diagnostic compute. Reduced-rate
version 7 preserves all state and uses actor Adam 0.00001 versus control 0.0001.
Critic rate, rewards, physics and full gates remain unchanged. Maximum new
PPO is 32,768 learner/world controls, with 84,000 evaluation controls separately.
Live output: `/home/pedro/boxing-runs/striking-actor-rate-1835008/comparison.json`.
No actor is adopted and no phase passes from a short integration check.

The reduced-rate first block is verified and included in the arm allocation:
exact source leaves retained, finite state, Adam 11,160/34,832, payload SHA
b0f0155901f006458b3b5336826c2b06164531179bd548f8d5b82fab1deeadb6.
Mean KL is 0.00291492 and ratio clip fraction 3.125%, lower than the completed
unchanged-rate probe (0.0379053 / 52.34375%). `validation.json` and full 18-row
measurements retain evidence; this is no physical improvement claim. Initial
and first-block model payloads are archived under the matching checkpoint folder.
The source fresh evaluation consumes 10,395 controls and fails complete gates;
all three full reports are archived. `launch-state.json` is a historical running
snapshot, not a live status endpoint. Inspect the live path before interventions.

## Independent complete review

The comparison completed with 24,576 learner/world controls and 62,822 evaluation
controls (144 complete episodes), not the maximum reserved compute. The reduced
rate stopped after 8,192 controls due to 3/8 neutral-role screen survival. Original
rate consumed its full 16,384. All finite payloads, three complete screen sets,
three matched fresh sets, 24 finite metric rows, empty native error logs and
retained main/foundation prefixes were verified by `tools/review_striking_actor_rate.py`.
Full reports and exact gate models are archived alongside `analysis.json`.

No complete gate passes. Fresh source / reduced / original survivors total
16/24 / 13/24 / 19/24; surviving active red-blue event counts are 2-2 / 2-3 /
8-9. Original-rate active survival is 6/8 versus 7/8 source, despite four two-way
surviving episodes versus two source episodes. Treat this as mixed small-sample
evidence, not a generalization guarantee. Reject the reduced-rate proposal.
The review permits one 16,384-credit ordinary original-rate confirmation from
its latest final only; see the policy-update investigation for stops and ceilings.
Global diagnostic PPO is 1,836,288, formal evaluation 1,052,177. Historical
launch/first-block snapshots above describe the integration, not current status.
