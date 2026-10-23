# Demonstration-retention PPO continuation

**Completed:** both fixed arms, all 120 physical episodes, final checkpoints,
training metrics and `analysis.json` are archived. Retention restores some legal
striking and one fresh stable two-way episode, but regresses role-1 survival;
neither actor passes. See the [final review](../../../striking-retention-review.md).
The launch and integration notes below retain their original historical scope.

This directory records the [bounded continuation](../../../striking-retention-continuation.md)
from the latest completed PPO-only actor, retaining all optimizer and lineage
state. The hidden persistent job started at **2026-10-23 13:24 UTC**. Its durable
live state is `/home/pedro/boxing-runs/striking-retention-1572864/comparison.json`.

`launch-state.json` is a snapshot, not a live counter. `validation.json` records
21 passing focused tests, the repeated two-test retention suite, all four current
complete foundation histories/hashes and the unchanged main actor. The exact
transition snapshot and configuration are under
`checkpoints/striking-validation/retention-1572864/transition/`. Transition consumes
no physics or optimizer steps and retains Adam counts 7,048 / 30,720.

The fixed matched allocation is 32,768 new learner credits per arm, with weights
0 and 1 on verified demonstration/approach replay. Both first retain a 1,024-credit
integration block. Source fresh evaluation and each final original/fresh complete
physical evaluation remain independent of the replay loss. No actor is adopted
or phase promoted by a launch, integration check or parallel comparison.

Before this job, diagnostic PPO is 1,450,240 learner credits, formal diagnostic
simulation is 547,894 world controls and standalone BC is 8,000 updates. Main
foundation/boxing counters stay 474,992,640 / 1,048,576. Inspect the current child,
native logs, per-arm attempted counters and hashes before intervening. No routine
rollback, duplicate launch or automatic extension is allowed.

## First integration boundary

The control completed and retained 1,024 new learner credits / 512 world controls
at 13:26 UTC, then resumed for its remaining 31,744 credits. Adam counts are
exactly 7,064 / 30,736, the checkpoint is finite and native error logs are empty.
SHA-256: `67f04025d1cb2b0d8037bcf3931de701d73235a97775f9961cc92d19347baa4a`.
The payload/configuration are archived under
`checkpoints/striking-validation/retention-1572864/ppo-only-initial-block/`.
The treatment's real first-block check is still pending; its parent performs the
same guard before allowing ordinary continuation. No final arm result is implied.

Fresh source baseline consumed **11,236 evaluation world controls**. Neutral-role
survival is 7/8 and 8/8, active survival 7/8, with zero surviving own events in any
profile. It fails the unchanged complete striking criteria. All 24 source trials
are archived here. At the integration boundary cumulative diagnostic PPO is
**1,451,264**, formal diagnostic simulation **559,130 world controls**, plus **512**
separately recorded new collection world controls. Live counters continue beyond
this snapshot while the bounded job runs.
