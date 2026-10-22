# Foundation-to-boxing contract audit, 2026-10-22

A read-only CPU audit compiled the existing two-boxer scene at 2.0 m separation
and compared it with the published foundation contract. This prepares the later
integration without changing the active learner or treating locomotion as boxing.
Reproduce it with the curriculum Python runtime:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu "$HOME/venvs/boxing_curriculum/bin/python" \
  tools/audit_curriculum_transfer.py
```

## Measured compatibility and differences

- Both corners have 29 actuators in the same local order as the foundation.
- The raw composed scene has two free roots, no equality constraints, no body
  gravity compensation, and 208 explicit contact pairs. The requested initial
  root positions are (-1, 0, 0.793) and (1, 0, 0.793) m.
- Eight position gains differ: ankle roll and all wrist axes are 20 in the
  existing boxing scene versus 2 in the foundation. A shared joint name does
  not imply the same actuator response.
- The boxing home pose differs in seven leg/waist joints. The largest difference
  is 0.369 rad at each knee; the boxing guard arm targets agree within rounding.
- The foundation records 2 ms physics steps and 20 ms control steps. The boxing
  model uses 4 ms physics steps. Existing `WarpBoxing` defaults to five such steps
  per action, but matching the control interval alone does not match integration.
- The foundation uses a 0.5 rad action scale and 103 actor / 216 privileged
  observation values. Existing `WarpBoxing` defaults to a 0.25 action scale and
  its own observations. Its default 0.75 m separation is also different from the
  explicitly audited wider 2.0 m scene.

The exact gains, poses, differences, and source checkpoint hash are in
`docs/results/curriculum/transfer-audit.json`. The audit scope is the raw scene;
it does not certify runtime wrapper forces, observations, scoring, or policy
performance. Explicit contact count is not a count of legal punch pairs.

## Integration requirements after foundation gates

Preserve or explicitly reconcile the learned action scale, default joint pose,
actuator gains, simulation/control intervals, normalization, and per-corner joint
indexing. Define the required proprioceptive inputs in each robot's local frame.
Opponent observations require a deliberate policy interface or validated network
expansion; loading different input shapes as the same actor is not acceptable.

Validate physical free-body standing, falls, and wider-spawn approach before
sparring. Separate fist-to-legal-target scoring from clinch/leg/forearm collision
contacts. Test side exchange, no stabilizing root forces, first-fall termination,
and real contact impulses. Any contact-buffer changes need overflow validation.
The existing curriculum checkpoint contract must continue rejecting an unsafe
direct transplant. This audit is not a policy-transfer implementation or boxing
training result.

Validation: the tool successfully compiled the actual MuJoCo boxing scene and
read the saved foundation contract on CPU, without constructing another GPU
learner. No scene, gains, pose, checkpoint, or training setting was modified.
