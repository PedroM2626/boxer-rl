# Matched motor-only versus static-guard comparison

**Completed and independently reviewed:** see the
[full result and decision](results/striking-pilot/guard-control-v1/README.md).
Actual consumption is 6,155 physical controls, cumulative 2,041,130. The original
prospective protocol below is retained; its unused controls expire at review.

Declared 2026-10-24 before collection, following the
[reviewed first-fall/workspace audit](results/striking-pilot/jab-failure-workspace-v1/README.md).
Question: does the exact static guard already disrupt the frozen motor, or is
the failed extension/retraction cycle necessary for the observed failures?
This is a scripted physical diagnostic, not training or punch certification.

## Matched intervention and physical criteria

Run two modes: all-zero boxing-head actions (motor-only), and the original v1
guard controller with phase fraction fixed at zero after the unchanged .75 m
engagement trigger. Preserve the guard point, left-arm Jacobian, damping,
correction cap, .8 rad residual authority, filter and action clipping. No direct
root/leg actions, forces, assistance, posture gains or contact changes.
Both modes retain the frozen certified approach motor and .45 m automatic
approach/engagement profile; both bodies otherwise use the original zero head.

Each mode uses seeds 7/23, both acting corners and both reset swaps: eight cases
per mode, **sixteen new episodes**, at most 500 controls / ten seconds each.
Corner labels intentionally duplicate the motor-only physics and are not eight
independent motor-only reset samples: there are four unique reset conditions.
Use all eight corresponding already-recorded jab v1 cases as a historical third
arm, without spending new controls to repeat them again. Compare full-state
approach prefixes against those exact archived cases before guard activation.

Record paired ten-second survival, exact first-fall flags/time by body, contact
events, range, full qpos/qvel, actions and guard clipping/tracking. The independent
review must check every trial and all files, finite states, unchanged source
hashes, native stderr, zero external forces, full-state continuity, intended
channels, original filtered target agreement and equal approach prefixes.
These sixteen cases cannot certify generalization, full boxing or a phase pass.
Report effect patterns and matched cases; do not invent new promotion thresholds.

## Bounded accounting and stop rule

Formal diagnostic physical controls begin at **2,034,975**. Allocate at most
**8,000 additional controls**, new ceiling **2,042,975**, justified by the need
to distinguish static guard disruption from cycle-specific failure. Diagnostic
PPO remains **2,163,968**; all main/BC/optimizer counters and policies unchanged.
The separately completed cloned geometry remains 4,339 evaluations; no new
geometry allocation. No new imitation targets or learner updates.

Stop on first fall per episode, native/overflow warning, nonfinite values,
unexpected stderr, source/profile mismatch, unequal approach prefix or compute
ceiling. Preserve partial attempted-control accounting. No blind retry or reuse
of old unused budgets. Use existing global/per-output locks and hidden persistent
WSL launcher. Inspect the process and
`/home/pedro/boxing-runs/striking-guard-control-v1/audit.json` before recovery.

After both arms complete and independent review, archive every case and decide:
if guard-only already falls where motor-only survives, redesign guard/posture
coordination first; if guard-only is stable but jab v1 fails, investigate cycle
timing/targets; if motor-only fails too, investigate base/range/contact behavior
before attributing those matched failures to the arm. Mixed results require
case-specific interpretation. No automatic PPO extension or threshold changes.
