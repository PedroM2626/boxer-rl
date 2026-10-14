"""Close-up stills of the grafted fist, because a contact sheet cannot show a hand.

The rollout camera frames both boxers whole, which is right for watching a round
and useless for checking whether the fingers are curled. This puts a free camera
on red's left palm and renders it from three directions at 35 cm, so the graft can
be looked at directly instead of inferred from the mesh transforms.

    MUJOCO_GL=egl python tools/fist_shot.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image

from boxer import scene

# Red faces +x, so the knuckles point at azimuth 0 -- but the other boxer stands
# exactly there, so the low views look through his torso. Elevation clears him.
VIEWS = (
    ("knuckles", 20.0, 55.0),
    ("thumb", -70.0, 15.0),
    ("above", 40.0, 50.0),
)


def main(argv=None) -> int:
    out = Path(argv[0]) if argv else Path("runs/fist_shot")
    out.mkdir(parents=True, exist_ok=True)

    model = scene.build_model(visual=True)
    data = mujoco.MjData(model)
    qpos, ctrl = scene.stance(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    palm = data.site(scene.RED + "left_palm").xpos.copy()
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = palm
    camera.distance = 0.5

    for name, azimuth, elevation in VIEWS:
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        path = out / f"{name}.png"
        Image.fromarray(renderer.render()).save(path)
        print(f"wrote {path}  (palm at {np.round(palm, 3).tolist()})")
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
