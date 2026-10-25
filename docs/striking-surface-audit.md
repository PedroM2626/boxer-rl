# Signed legal-target and blocking-surface audit

**Completed and independently reviewed:** see [full results](results/striking-pilot/surface-audit-v1/README.md).
All 48 poses and 480 distances are independently reproduced. Cumulative cloned
count is 6,349; separately 960 distance calls. Physical controls stay 2,051,072.
The prospective protocol below is retained unchanged.

Declared 2026-10-25 before new queries. The reviewed terminal dwell preserves
8/8 pair survival but has zero left-fist target contacts. Prior static searches
leave substantial distance to a point near the opposing torso center. That point
error does not establish whether collision surfaces can touch.

Use all 24 independently reviewed terminal-workspace snapshots, each at its
recorded pose and its already-saved best local endpoint pose: **48 poses**.
Do not optimize again or select only favorable cases. Query the acting left-hand
collision capsule against all ten original opposing upper-target geoms: head and
torso are legal score surfaces; shoulders, elbows, wrists and hands are blocking
surfaces. Verify each pair is explicitly declared by the unchanged physical model.
Use MuJoCo's signed geom distance with a 5 m cap, preserve nearest-point segments
and censoring flags, and reject unsupported/capped queries rather than call them
exact gaps. Positive means separation; negative means geometric overlap, not an
impact or a solver force. Compute only kinematics on disposable clones.

Declare **96 new cloned forwards**, 48 producer plus 48 independent review;
prior diagnostic cloned count 6,253, cumulative ceiling **6,349**. Separately
declare **960 signed-distance calls**, 480 producer plus 480 review, a new
measurement counter. This does not reuse the expired earlier geometry budgets.
Physical controls remain **2,051,072**, diagnostic PPO **2,163,968**, combined
main transitions **476,041,216**, BC **8,000**. Rendering-only 500 forwards are
separate and unchanged. No physical integration, training, force, gain, authority,
root/leg constraint in a live episode or policy adoption is authorized here.

Persist counters before each query. Stop on nonfinite/capped distance, unexpected
native stderr, source/hash/pair mismatch or either hard ceiling; preserve partial
accounting and do not retry blindly. Use the hidden persistent WSL launcher and
existing output/global locks. Inspect
`/home/pedro/boxing-runs/striking-surface-audit-v1/audit.json`, progress and process
before recovery. Independently reproduce all 48 poses and 480 surface distances,
verify source pose/joint-box provenance, pair membership and compute totals.

Also retain the physical trace's first positive-force blocking contacts, clearly
separate from static signed distances: an earlier force event does not prove a
block is active at a later snapshot. Negative blocking gaps in a best pose mean
that pose intersects a blocking surface; they do not prove the actual path is
blocked. Static overlap neither certifies dynamic contact nor proves a safe,
balanced punch. Choose a bounded surface-target/path/authority comparison only
after reviewing both legal gaps and blocking intersections; never lower physical
gates or adopt these cloned poses as demonstrations.
