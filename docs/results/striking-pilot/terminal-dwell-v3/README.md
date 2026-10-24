# Terminal dwell: stability retained, no physical strikes

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-terminal-dwell.md). Add 120 ms holding
the original full-extension target after the unchanged 180 ms outbound motion;
preserve the learned guard, motor/filter return, free legs and unassisted physics.
The new cycle is 1.20 seconds instead of 1.08; other phase durations are unchanged.

| Matched diagnostic | Ten-second pair survivors | Legal scored impacts | Candidate punches |
| --- | --- | --- | --- |
| Prior v2 immediate motor return | 8/8 | 0 | 0 |
| New v3 terminal dwell | 8/8 | 0 | 0 |

All eight cases have zero acting-left-fist contact substeps. No first-fall
predicates, nonfinite state or native errors occur. Each case includes 42 dwell
controls; median full-target error within dwell is 31.6–35.7 cm by case.
The combined outbound/dwell median is 27.6 cm and action-channel clipping 20.9%.
That combined median covers different phase fractions than v2's outbound-only
14.7 cm and must not be called a matched tracking regression by itself.

The full v1 candidate conjunction remains unchanged: onset speed/direction,
pre-window excursion, complete return window, contact duration and legal actual
contact must all qualify; two candidates plus ten-second survival define the
diagnostic episode. None qualifies. Reject 120 ms dwell as a sufficient repair;
do not train on these non-contact motions or extend PPO unchanged.

## Evidence, accounting and next decision

Independent review checks all eight cases, exact qpos/qvel continuity with the
archived control **through the first outbound extension before the first changed
dwell**, source/checkpoint hashes, intended channels, 1.20 s capture/phase clock,
zero non-extension/dwell head actions, original filter targets, finite states,
native event counts and all complete candidate criteria. Thirteen focused tests
pass. `full/` and `native/` retain every case; [audit.json](audit.json), progress
and [analysis.json](analysis.json) preserve hashes, full outcomes and accounting.
`native-errors.log` is intentionally empty.

New physical controls **4,000**, cumulative **2,051,072**, exactly its declared
ceiling. PPO remains 2,163,968, main combined transitions 476,041,216, standalone
BC 8,000. No actor, optimizer, normalizer, gate or policy-adoption changes.

The [separately bounded terminal workspace review](../terminal-workspace-v1/README.md)
uses saved states without new physical integration. It finds substantial remaining
endpoint error even in local static searches within the original joint box.
Next distinguish the commanded point from actual legal collision-surface reach
before changing target, authority or motor coordination. Endpoint error alone
does not prove physical fist contact is impossible. No new physical or training
allocation follows from this failed dwell comparison.

## Recorded-state video

[Watch the latest diagnostic](../../../../fights/striking-pilot/terminal-dwell-v3-recorded.mp4):
20 seconds, seed 7 without swap, ten seconds per acting corner in fixed order.
The caption identifies a scripted diagnostic and zero valid impacts. It does
not present the motions as successful boxing. The original closed-fist visual
graft is used; the recorded free-root/leg coordinates are unchanged.

The renderer reads saved qpos/qvel and runs only visual forward calculations;
it never calls physical stepping or the policy. All 500 frames decode at 25 fps,
640x480. [video.json](video.json) retains full source/video SHA-256 values,
selected control indices and validation. Two preview frames are retained beside
it. The closed-fist/physics-coordinate regression test passes.

This adds zero physical controls, optimizer steps or diagnostic geometry queries;
500 rendering-only forward calculations are recorded separately. Main/diagnostic
counters and all physical outcomes remain unchanged. The host emitted a Python
subprocess `os.fork`/multithreaded-JAX warning while starting FFmpeg;
`video-render-stderr.log` preserves it. Encoding and full decoding completed,
with no observed deadlock or physical-policy failure. No simulation was rerun.
