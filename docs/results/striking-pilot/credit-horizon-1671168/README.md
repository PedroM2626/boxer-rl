# Matched discount-horizon comparison: completed review

**Current status: completed, independently reviewed; neither arm passes.**
The final review and one bounded follow-up are recorded below.

See the [completed audit and bounded comparison contract](../../../striking-reward-credit-review.md).
Live state is `/home/pedro/boxing-runs/striking-credit-horizon-1671168/comparison.json`.
Inspect active child/progress, all role/physical counters, hashes and native logs
before another launch. `launch-state.json` is historical, not live accounting.

Both arms retain the latest final neutral actor, critic, normalizer, key, replay
samplers and independently counted Adam histories. Only gamma differs (0.99 /
0.999); reward, lambda, replay, native physics, task and full criteria are unchanged.
`validation.json` records 28 passing focused tests and the exact transition hash.
The version-5 control transition payload/configuration is archived under
`checkpoints/striking-validation/credit-horizon-1671168/transition/`.

The fixed allocation is 32,768 new one-learner credits/world controls per arm,
65,536 total, including identical retained 1,024-credit integration boundaries.
Source fresh plus both complete final original/fresh sets are at most 120
episodes / 60,000 evaluation controls on preregistered 10M-series seeds. Neither
a better shaped return nor two parallel arms supplies a consecutive phase pass.

Pre-job diagnostic PPO is 1,581,312 and formal diagnostic simulation 712,013 world
controls. Source lineage is 1,671,168 / 1,048,576 world controls, Adam 8,584 /
32,256; transition consumes no physics or optimizer steps. Critic calibration
starts from the old gamma identically in both arms, without reset. Main counters,
ceilings and standalone BC remain unchanged. Archive both complete final arms
and actual consumption at the end; no main adoption or further automatic
extension is authorized.

## Verified control integration

The 24-trial source baseline consumed 12,000 world controls: 8/8 survival in all
profiles, no own neutral events, one surviving active event on one seed. It does
not pass complete striking criteria. All raw source reports are archived here.

`control-initial-block-verified.json` records 1,024 retained learner/world credits,
512 newly credited per role, Adam 8,600 / 32,272 and finite checkpoint hash
`402a50b3e5f1444bdcdfc5c1a8b4718bab4b06345a6874ecf1a641f26590778e`.
The payload/configuration is archived under
`checkpoints/striking-validation/credit-horizon-1671168/control-initial-block/`.
At that earlier snapshot native logs were empty and control training was active;
the gamma-0.999 guard was pending. Both arms subsequently completed and their
individual integration proofs are archived. These guards are not boxing passes.

At this boundary cumulative diagnostic PPO is 1,582,336, formal diagnostic
simulation 724,013 world controls, plus 1,024 new collection controls separately.
`integration-state.json` is historical; read current progress before intervention.


The comparison is now **completed and reviewed**. See `analysis.json`,
`completed-comparison.json`, both arm directories (full screen/fresh reports,
progress, state, initial native proof and 32 finite update metrics), and final
models under `checkpoints/striking-validation/credit-horizon-1671168/`.
Gamma 0.999 has six surviving active events across four fresh seeds and one
surviving two-way episode; neither arm passes complete neutral/active criteria.
No actor is adopted. The [review and single bounded follow-up](../../../striking-reward-credit-review.md#completed-horizon-comparison-and-bounded-continuation-2026-10-23)
record the decision, exact hashes, actual compute and limits. The launch and
integration snapshots above describe earlier boundaries, not current counters.
