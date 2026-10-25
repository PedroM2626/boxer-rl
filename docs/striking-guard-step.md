# Separate lateral movement and straight extension in time

Declared 2026-10-25 after independent v6 review, before collection. Removing arm
requests eliminates all six v5 positive-command failures (8/8 motor-only survivors
versus 2/8 concurrent-control survivors), but no valid punch has been produced.
Test temporal overlap without changing motor/arm authority or physical gates.

Eight cases: seeds 7/23, both acting corners, both swaps. Compare to all archived
v5 positive-command-plus-jab controls; do not rerun or select controls. Request
lateral action +1 only during each 350 ms guard interval. Release it before
extension/dwell, retraction and rest. Retain v3 straight path, learned motor guard,
180/120/250/300 ms remaining clock, original residual box/correction/filter,
2 m initial separation, .75 m activation, .45 m automatic approach stop, same
frozen motor and all original contacts/physics. Compute arm baseline with the
released command. No direct root/leg forces, motor unfreezing or arm enlargement.
Require bitwise qpos/qvel prefixes until first extension command release.

At most 4,000 new physical controls, from 2,070,307 to **2,074,307**. The v6
allocation is exhausted. No additional PPO/BC/geometry/distance/render allocation.
Main 476,041,216, diagnostic PPO 2,163,968, BC 8,000, cloned forwards 6,349,
signed-distance calls 960 and renderer forwards 500 retained. Existing hidden
WSL launcher and global/output locks; persist attempts before stepping, preserve
partial failures, stop each case on first original fall/ten seconds, stop job
on nonfinite/overflow, unexpected native stderr, prefix/hash mismatch or ceiling.

Independent review retains full traces, every matched condition and every original
candidate requirement: acting-left legal event, onset speed >=1 m/s, forward
cosine >=.7, pre-travel >=.12 m, complete post-window and return >=.06 m, duration
<=.10 s. Feasibility still requires two candidates and ten-second pair survival.
Improved survival alone is not a positive punch target or learned boxing.
If no full candidate emerges, reject this handoff as a sufficient punch repair;
review remaining tracking/contact evidence before further bounded work. Do not
repeat unchanged or automatically extend imitation/PPO. Positive scripted cases
would still need broader holdouts and learned validation before any adoption.
