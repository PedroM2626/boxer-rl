# Completed frozen-policy backend comparison

Reviewed 2026-10-29. Actual final v17 actor/optimizer remains unchanged. All
profiles use the same four fresh seeds, both placements, unassisted free physics
and unchanged complete-punch rules. Native 3/5 is the historical baseline;
the other arms are separately declared numerical diagnostics.

| Engine / solver / line search | Neutral-0 survivors | Neutral-1 survivors | Mirror survivors | Valid mirror cycles by corner |
| --- | --- | --- | --- | --- |
| Historical native / 3 / 5 | 5/8 | 7/8 | 3/8 | 4 / 1 |
| Native / 30 / 20 | 6/8 | 4/8 | 3/8 | 2 / 4 |
| CUDA / 3 / 5 | 5/8 | 4/8 | 5/8 | 3 / 5 |
| CUDA / 30 / 20 | 6/8 | 4/8 | 3/8 | 0 / 0 |

No arm passes complete unchanged criteria. CUDA 3/5 has reciprocal valid cycles
in two surviving fresh seed conditions, but fails required survival. Higher solver
convergence does not improve learned boxing consistently. This small sample does
not establish a superior backend, numerical trajectory equivalence or a reason
to change production physics. Keep native gates authoritative; no promotion,
rollback or new PPO.

Source/report/full/native trace hashes, finite arrays and event windows were
rechecked in `completed-review.json`. GPU output is exported into the unchanged
native measurement API. Source actor SHA-256:
`f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a`.

New formal controls **24,149**, cumulative **2,295,150**. Unused **11,851** of
the declared 36,000 allowance expire. All six profiles completed before the archive
interruption were retained/verified; only the three remaining profiles resumed.
No repeated evaluation controls or discarded accounting.

Decision: continue the declared own-engine Warp6/7 comparison; then select the
architecture/learning continuation from complete evidence. The fast batched CUDA
prototype still needs complete scorecard, motion history, reward/reset parity
and PPO integration. Stable repeated reciprocal fighting and broader/longer
fresh validation remain required for completion.
