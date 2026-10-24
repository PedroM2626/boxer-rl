# Evaluation-only automatic approach-yaw comparison

See `docs/striking-approach-yaw-review.md` for rationale, exact source, cap and
limitations. Latest final 6bf3273a is unchanged. Compare automatic base-yaw scale
0.5 against 1.0; preserve learned offsets/arms and all physics/rewards/thresholds.
Reuse verified original 16M control, run modified 16M and both fresh 17M profiles:
72 new episodes / at most 36,000 controls, no optimization or adoption. Modified
reports are explicitly ineligible for ordinary promotion. Global PPO stays
1,844,480; formal controls before job 1,106,474, maximum after 1,142,474.

Live state: `/home/pedro/boxing-runs/striking-base-yaw-1859584/comparison.json`.
Launcher: `tools/start_striking_base_yaw_compare.ps1`. Preserve monitoring.

## Completed review

All 72 new episodes / 33,287 controls are verified with finite states, exact
source provenance and empty native logs. The full three new report sets and
independent `analysis.json` are archived. Existing 16M control is reused without
counting its simulation again. Neutral survival improves, but active survival
regresses 6/8 to 5/8 on 16M and 7/8 to 5/8 on fresh 17M. Reject global yaw
scaling; no controller adoption or optimization for it. Diagnostic PPO remains
1,844,480; formal evaluation becomes 1,139,761. Complete foundation histories
and main hashes are reverified. Next contract: `docs/striking-fixed-opponent-comparison.md`.
