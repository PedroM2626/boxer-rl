# Supervisor log-read recovery, 2026-10-22

## Observed failure and preserved training

Full walking stopped at **5,017,600 stage / 422,891,520 main consumed
transitions**. The supervisor traceback ends at `reader.read()` while monitoring
the child log, with Linux `OSError: [Errno 61] No data available` (`ENODATA`).
The exception handler terminated the learner process group and marked the job
failed, as designed. The recorded PPO metrics were finite and no physics
overflow diagnostic was found. This locates the observed failure in log
supervision; it does not identify the underlying cause of the filesystem error.
The log is on the WSL-mounted Windows workspace.

The last checkpoint hash was independently verified:
`fbc36682e4b3f19de39ed1a2f9a4af76970848c7b47d62ea059a3b597263c15d`.
The policy, metadata, and config are preserved under
`checkpoints/curriculum/archive-log-read-failure-422891520/`.
The original traceback and failed state are under
`docs/results/curriculum/log-read-recovery/`.

Balance and easy-walking certifications remain valid. The new full-walking
stage's initial gate passed only 5/14 individual trials and failed the full
benchmark, so it remains in training. The 5M checkpoint had not completed its
next gate at the interruption; finite PPO losses do not certify its performance.

## Targeted correction

`read_log_chunk` now retries **only ENODATA**, with at most three read attempts
and 0.1/0.2-second delays. Each attempt reopens the file and seeks to the same
previously committed offset. The caller advances its offset only after the
complete read succeeds. A read that internally consumes partial bytes before
raising therefore cannot cause an overflow warning to be skipped on retry.

Persistent ENODATA and every other read error still propagate to the existing
process-group cleanup. The child cannot continue indefinitely without its
physics-output guard. Scanning of new output, the overlap buffer, and termination
on a physics-allocation overflow remain unchanged. Historical log warnings are
still excluded from the new child scan. This is a bounded recovery for an
observed read fault, not a blanket suppression of I/O errors.

## Continuation and verification

Twenty-two focused CPU tests passed, including an injected read that consumes
bytes before ENODATA, recovery from the original offset with the complete
overflow warning retained, a three-attempt bound for persistent ENODATA, and
immediate propagation of an unrelated permission error. Existing real-child
overflow termination, promotion, profile inheritance, and comparison tests also
passed. No separate GPU learner was started for testing.

The normal hidden WSL host was relaunched with `-Publish -WarmStart
-MaxTotalSteps 450000000`. It restores the latest full-walking actor, critic,
and normalizer without rollback or rejected updates, retaining all 422,891,520
consumed main transitions. Adam/RNG restart explicitly, as before. No budget,
reward, command distribution, action/model contract, gate, or certification
criterion changed. The earlier diagnostic accounting remains separate.

The resumed full-walking stage keeps its original 0.25 tracking width and zero
targeted-lateral override. Future full gates decide learning progress. If this
read error repeats persistently, keep the fail-closed behavior and investigate
filesystem/log placement rather than extending retries without a bound.
