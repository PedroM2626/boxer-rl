# Static workspace at recorded terminal-dwell states

**Completed and independently reviewed:** see [results and limitations](results/striking-pilot/terminal-workspace-v1/README.md).
All 24 queries are reviewed, consuming 1,890 producer plus 24 review forwards;
cumulative cloned total 6,253. No new physical controls. The prospective protocol
below is retained; unused compute expires at review.

Declared 2026-10-24 after the independently reviewed dwell comparison. All eight
pairs survive, but no left-fist target contact occurs. Median full-target error
within dwell is about 32–36 cm by case. Before changing authority, timing or the
motor, test whether the endpoint can be matched inside the original motor-centered
joint box at those actual recorded poses. This is geometry, not a physical gate.

Choose at most three snapshots per case: last control of the first dwell, the
dwell control with lowest recorded Cartesian error and last control of the last
dwell. Remove duplicate indices. At most 24 snapshots / target queries; archive
source case, control index, exact qpos, motor baseline, target and source hashes.
Use the existing two-start bounded static solver, original seven left-arm joints,
baseline +/- .8 rad intersected with physical joint limits and max 40 function
evaluations per start. Hold other coordinates only in disposable cloned MjData.
No live root/leg constraints, mj_step, integration, inference or optimizer updates.

Declare a new **6,000 cloned forward-evaluation** allocation, including at most
24 independent review forwards. Prior cloned consumption 4,339; new cumulative
ceiling **10,339**. This is not reuse of the expired previous 20,000 allocation.
Physical controls remain **2,051,072**, PPO 2,163,968 and main/BC counters unchanged.
Persist query progress, stop before exceeding the ceiling, on nonfinite state,
unexpected native stderr or source mismatch. Existing global/per-output locks and
hidden persistent WSL launch remain mandatory; inspect
`/home/pedro/boxing-runs/striking-terminal-workspace-v1/audit.json` before recovery.

Independently verify sources, snapshot selection, actual source telemetry,
joint bounds and each best position with a fresh cloned kinematic call, separately
counted. No solver rerun on review and no physical replay. A locally matched
endpoint does not establish dynamic, collision-safe or balanced reachability.
Failure of two local starts does not prove global impossibility. Report actual
versus local-best errors and bounds, not a retroactive pass threshold. Use these
results to decide whether to investigate controller tracking or terminal target/
authority geometry; no automatic learning extension or policy adoption.
