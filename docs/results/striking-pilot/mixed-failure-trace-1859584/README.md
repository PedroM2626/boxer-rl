# Passive matched mixed-confirmation failure trace

The complete confirmation stagnated. Replay source/final on unchanged complete
neutral-0, neutral-1 and self-play profiles, seeds 16,000,003..16,000,006, both
swaps: 48 episodes / maximum 24,000 world controls. No optimization or control
changes. Require exact agreement of every original physical trial field, checked
finite traces and native error logs. All consumed simulation is retained even
on failure. Temporal association of contact, posture, commands and falls is not
a causal explanation; do not choose a controller modification solely from it.
Diagnostic PPO stays 1,844,480, formal evaluation ceiling 1,108,975. Main and
standalone supervised counters remain unchanged. No adoption or promotion.

Live state: `/home/pedro/boxing-runs/striking-mixed-failure-trace-1859584/trace-review.json`.
Launcher: `tools/start_striking_mixed_failure_trace.ps1`. Preserve monitoring.

## Integration and verified recovery

Eleven focused tests pass, including exact native passive/untraced trajectory
agreement, contact-timing summaries and complete-screen early-stop behavior.
Python/Bash syntax and PowerShell parsing pass. The first full native profile
(source neutral-0) verifies all eight original trial fields exactly, all finite
traces and checked hashes: 4,000 world controls. The reports and eight raw traces
are archived here; launch state remains a historical snapshot of the active job.

After that replay completed, a missing archived source-reference report stopped
the analyzer. The confirmation reviewer now archives the full source reports.
The failed state is preserved in `failure-before-resume.json`. Explicit recovery
reuses only completed reports, verifies exact original trial agreement and
preserves the 4,000 consumed controls; it does not rerun that physics. A partial
report is rejected rather than silently replayed. The resumed process continues
remaining profiles with unchanged policies and the same 24,000-control total cap.
No training, optimizer reset or physical change was introduced by this repair.
