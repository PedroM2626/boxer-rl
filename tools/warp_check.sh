#!/usr/bin/env bash
# Does mujoco_warp expose per-contact force?
#
# The question this answers: the published boxing recipe (RoboStriker, same robot)
# scores a punch by a contact-force threshold, and the MJX path we train on cannot
# implement it -- verified locally on mujoco 3.14.0: `mjx.Data` has no force or efc
# field at all, and `mujoco.mj_contactForce` is CPU/mjData-only. MuJoCo Warp is the
# maintained GPU-batched engine the Playground points to, and its documentation does
# not say whether force survives into its batched Data. This script settles it by
# reading the installed package, not by trusting the README.
#
# Installs into its own venv (/home/pedro/venvs/warp_test) so boxer313's pinned
# scipy/pytest stack is untouched. Run on a free GPU: importing warp may initialise
# CUDA, and a second CUDA context next to a trainer has restarted this VM before.
#   wsl bash -lc 'cd /mnt/e/boxer-agent && bash tools/warp_check.sh'
set -uo pipefail

UV=/home/pedro/venvs/boxer/bin/uv
VENV=/home/pedro/venvs/warp_test
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  "$UV" venv "$VENV" --python 3.12
fi
"$UV" pip install --python "$PY" mujoco warp-lang mujoco-warp 2>&1 | tail -4

"$PY" - <<'PY'
import importlib

import mujoco

print("mujoco", mujoco.__version__)

module = None
for name in ("mujoco_warp", "mujoco.warp"):
    try:
        module = importlib.import_module(name)
        print("imported", name, "->", getattr(module, "__file__", "?"))
        break
    except Exception as exc:
        print(name, "not importable:", exc)

if module is not None:
    names = [n for n in dir(module) if not n.startswith("_")]
    print("module force ops:", [n for n in names if "force" in n.lower()])
    data = getattr(module, "Data", None)
    if data is not None:
        fields = [n for n in dir(data) if not n.startswith("_")]
        print("Data field count:", len(fields))
        print("Data force/efc fields:", [n for n in fields if "force" in n.lower() or "efc" in n.lower()])
        print("Data contact fields:", [n for n in fields if "contact" in n.lower()])
PY

echo "=== grep the installed package for force-bearing arrays ==="
grep -rn --include="*.py" -iE "efc_force|contact_force|xfrc|contact_force" "$VENV/lib/" 2>/dev/null |
  grep -i "mujoco_warp\|mujoco/warp\|mujoco\.warp" | head -12
echo "(empty means no contact-force array anywhere in the warp package)"
