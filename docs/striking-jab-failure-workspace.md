# Exact failed-cycle replay and bounded static workspace audit

**Completed:** the [independent result review](results/striking-pilot/jab-failure-workspace-v1/README.md)
accounts for all 4,048 physical controls and 4,339 separately counted cloned
kinematic evaluations. No learner update or policy adoption. The protocol below
is retained as declared before collection; unused query compute is not a new
continuation allocation.

Declared 2026-10-24 before new collection. The reviewed
[v1 prototype](results/striking-pilot/jab-cycle-v1/README.md) fell in all sixteen
conditions and did not save full root/leg states. Its native arm traces cannot
isolate first-fall causes or whether requested targets fit the real motor-centered
residual box. Do not redesign from a guessed cause or resume unchanged PPO.

## New bounded measurement allocation

Replay all sixteen **unchanged** cases to their recorded first-fall control,
with **at most 4,048 additional physical controls**, the exact sum already
observed. Formal simulation starts at **2,030,927**; new ceiling **2,034,975**.
This is a separate, justified measurement allocation, not automatic reuse of
unused v1 compute. No PPO, imitation, optimizer, actor, normalizer or dataset changes.
Diagnostic PPO stays 2,163,968; all main/BC counters remain unchanged.

Every original case must reproduce its elapsed time, scores, phase counts,
survival and every array in its native punch trace. Require the same individual
control count; stop rather than spend an extra control if replay diverges.
Passive control traces now include before/after qpos/qvel, original actions,
motor actions, pelvis height/upright and terminal contacts/fall predicates.
Capture the original motor result with a transparent observer, then independently
recompute request saturation, filter-limited targets and Cartesian tracking error.
No physical forces, gains, contact pairs, termination criteria or controller
parameters change. Classify observed first-fall flags without inferring causality
from contact timing alone.

## Geometry scope and ceiling

At the first guard control and first extension control at >=half of its clock
fraction in each case, retain the actual state and original motor-centered target
baseline. At most 32 recorded snapshots / 64 guard/strike target queries.
For each, intersect the original +/- .8 rad left-arm target envelope with model
joint limits. Two bounded local least-squares starts use analytic Jacobians and
at most 40 function evaluations each. A shared hard ceiling of **20,000 cloned
kinematic forward evaluations** applies separately from physical consumption.

The solver operates only on disposable MjData copies. Roots, legs and other
joints are held at the recorded pose for geometric queries; this is not assistance
in a live simulation. Only `mj_kinematics`/`mj_comPos` are called, with no mj_step,
integration, reset of an active trial, training or weight update. Static joint
targets do not prove dynamic reachability, collision safety, balance or punch
feasibility. Failure of two local solver starts is not a global impossibility
proof. Report solver errors, bounds and evaluated targets rather than inventing
a pass threshold or treating cloned poses as boxing footage.

Stop on native warning/nonfinite values, unexpected stderr, source/reference or
trace mismatch, or either compute ceiling. Preserve all failed/partial records.
Use the hidden persistent WSL launcher and existing global boxing/diagnostic locks;
inspect `/home/pedro/boxing-runs/striking-jab-failure-workspace-v1/audit.json`
and `progress.json` before recovery. No duplicate or blind retry is authorized.

Interpret the measured fall predicates, tracking/saturation and geometric errors
before a prospectively bounded redesign. Original contact and jab-cycle criteria
remain frozen; no promotion, policy adoption or next training allocation follows
automatically.
