# Post-PPO saved-Gaussian diagnostic

**Completed:** all 72 episodes consumed 34,237 world controls. Raw profiles,
`completed-sampling.json` and `analysis.json` are archived here. Neither final PPO
actor recovers stable strikes; no surviving two-way episode occurs. All results
are finite and native error logs empty. See the [completed review and next
continuation](../../../striking-retention-continuation.md). The launch snapshot
and notes below are retained as historical evidence.

This is a launch snapshot, not final physical evidence. See
[the completed PPO review](../../../striking-ppo-anchor-review.md) for the bounded
72-episode / 36,000-world-control allocation, unchanged physical criteria and
rejected PPO extensions. No optimizer update or policy adoption is authorized by
this audit. Inspect live durable state before another launch:
`/home/pedro/boxing-runs/striking-postppo-sampling-bc5000/sampling.json`.

The parent and one native evaluator were verified active after the source neutral
profiles completed. Native error logs were empty. Existing full PPO comparison
integrity/accounting checks were rerun successfully, including all 128 finite
block losses per arm, matching auxiliary streams, checkpoint hashes and complete
foundation certification histories. Both new Python tools compile and the LF
WSL shell wrapper passes `bash -n`. No training implementation or physics changed.

Archive complete sampled results and actual consumed controls after completion;
this snapshot does not include later attempted evaluation controls.
