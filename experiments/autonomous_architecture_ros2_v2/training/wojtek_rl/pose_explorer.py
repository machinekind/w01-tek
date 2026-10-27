"""Render candidate standing poses so a human can pick one.

For each candidate the script drops the robot from 25 cm with the PD
actuators holding the candidate joint targets, lets it settle for two
seconds, renders a frame, and prints the settled base height and the
four-bar closure error. Frames land in pose_previews/.

Run: ./run.sh pose            # default grid
     ./run.sh pose --second 0.9 --third 1.8   # single candidate
On macOS the default GL backend works; EGL is for the linux box.
"""

import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np

from wojtek_rl import build_model, paths

OUT_DIR = paths.PROJECT_DIR / "pose_previews"


def load_scene(kp: float) -> mujoco.MjModel:
    """Regenerate the model files with this kp, then load the scene.

    The scene carries the floor and the track camera; a robot dropped
    without a floor never settles.
    """
    build_model.write_models(kp=kp)
    return mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))


def actuated_ids(model: mujoco.MjModel) -> list[int]:
    """qpos addresses of the 12 actuated joints, in actuator order."""
    out = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        out.append(model.jnt_qposadr[jid])
    return out


def try_pose(model, second: float, third: float, settle_s: float = 2.0):
    data = mujoco.MjData(model)
    qadr = actuated_ids(model)
    targets = np.zeros(12)
    # actuator order is (first, second, third) x LEGS
    targets[1::3] = second
    targets[2::3] = third
    data.qpos[2] = 0.25  # drop from 25 cm
    for adr, t in zip(qadr, targets):
        data.qpos[adr] = t
    data.ctrl[:] = targets
    mujoco.mj_forward(model, data)
    for _ in range(int(settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    closure = max(
        np.linalg.norm(
            data.body(f"{leg}_foot_link").xpos
            - data.body(f"{leg}_chain_close_a_link").xpos
        )
        for leg in paths.LEGS
    )
    return data, data.qpos[2], closure


def render(model, data, path: Path) -> None:
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera="track")
        import imageio

        imageio.imwrite(path, r.render())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--second", type=float, default=None)
    p.add_argument("--third", type=float, default=None)
    p.add_argument("--kp", type=float, default=build_model.DEFAULT_KP)
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    model = load_scene(kp=args.kp)

    if args.second is not None:
        grid = [(args.second, args.third)]
    else:
        grid = list(itertools.product([0.5, 0.8, 1.1], [1.0, 1.5, 2.0, 2.5]))

    print(f"{'second':>7} {'third':>7} {'height':>7} {'closure':>9}")
    for second, third in grid:
        data, z, closure = try_pose(model, second, third)
        name = OUT_DIR / f"pose_s{second:+.2f}_t{third:+.2f}.png"
        render(model, data, name)
        print(f"{second:7.2f} {third:7.2f} {z:7.3f} {closure:9.5f}  {name.name}")


if __name__ == "__main__":
    main()
