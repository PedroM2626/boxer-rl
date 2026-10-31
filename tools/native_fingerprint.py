"""Fingerprint native stepping to check that a speed-up changes nothing.

Plays a learned-vs-learned 20 s bout, a scripted bout under knockdown rules
and 1500 native fine-tuning steps (``NativeBout``), and records events,
rewards, observations and speed. The first run writes the reference JSON;
later runs compare against it:

    bash tools/run_spar.sh tools.native_fingerprint runs/fingerprint.json
"""
import json
import sys
import time

import jax, numpy as np
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.native_finetune import NativeBout
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import PolicyBrain, RingBrain, bout
f = UnassistedG1("spar")
motor = jax.jit(inference(f, load_policy_params("checkpoints/spar")[0]))
pair = PairedFoundation(f)
tac = load_policy_params("checkpoints/tactics")[0]
out = {}
t = time.perf_counter()
r, _, _ = bout(pair, motor, 3, 20.0, brains={0: PolicyBrain(tac), 1: PolicyBrain(tac, deterministic=False)})
out["bout_seconds"] = time.perf_counter() - t
out["bout"] = {k: r[k] for k in ("clean_punches", "impact_events", "punches_thrown", "fallen", "seconds")}
r, _, _ = bout(pair, motor, 5, 20.0, brains={1: RingBrain(np.random.default_rng([5, 1]), high=True)}, knockdowns=3)
out["bout_scripted"] = {k: r[k] for k in ("clean_punches", "impact_events", "punches_thrown", "fallen", "restarts")}
nb = NativeBout(pair, motor, np.random.default_rng(0), [PolicyBrain(tac, deterministic=False)], .5, 300, 30., .5)
rng = np.random.default_rng(1)
rewards, obs = [], []
t = time.perf_counter()
for k in range(1500):
    reward, term, trunc, info = nb.step(rng.uniform(-1, 1, 7))
    rewards.append(reward)
    if k % 50 == 0:
        obs.append(nb.observe().tolist())
    if term or trunc:
        nb.reset()
out["finetune_steps_per_second"] = 1500 / (time.perf_counter() - t)
out["rewards"], out["obs"] = rewards, obs
golden = sys.argv[1]
try:
    ref = json.load(open(golden))
    same = {k: ref[k] == out[k] for k in ("bout", "bout_scripted", "rewards", "obs")}
    print("identical to reference:", same)
    print(f"speed: fine-tune {ref['finetune_steps_per_second']:.0f} -> {out['finetune_steps_per_second']:.0f} steps/s, "
          f"bout {ref['bout_seconds']:.1f} -> {out['bout_seconds']:.1f} s")
except FileNotFoundError:
    json.dump(out, open(golden, "w"))
    print("reference written", out["finetune_steps_per_second"], out["bout_seconds"])
