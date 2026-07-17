#!/usr/bin/env python3
"""Sanity check for the wojtek MuJoCo model.

Loads scene.xml, verifies the model structure (actuators, equality
constraints), simulates 2 s with zero control, and verifies the robot
settles on the floor without exploding and without its four-bar closures
drifting apart.

Run with the project venv from any directory:
    .venv/bin/python wojtek_description/mujoco/check_model.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path(__file__).resolve().parent / "scene.xml"

SIM_DURATION_S = 2.0

# Zero-torque rollout: the robot collapses and lies flat. The band only
# guards against falling through the floor or being launched by an
# unstable solve; a few mm below 0 is normal contact softness.
TRUNK_HEIGHT_MIN = -0.05
TRUNK_HEIGHT_MAX = 0.5
CLOSURE_ERROR_MAX = 2e-3

LEGS = ("rear_left", "rear_right", "front_right", "front_left")
JOINT_CTRLRANGES = {
    "first_joint": 2.0,
    "second_joint": 1.0,
    "third_joint": 1.0,
}


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check_structure(model):
    if model.nu != 12:
        fail(f"expected nu == 12, got {model.nu}")
    if model.neq != 4:
        fail(f"expected neq == 4, got {model.neq}")

    # 25 visual + 25 collision mesh geoms; a geom that references a mesh but
    # omits type="mesh" is silently fitted with a sphere primitive instead.
    n_mesh_geoms = int((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH).sum())
    if n_mesh_geoms != 50:
        fail(f"expected 50 mesh geoms, got {n_mesh_geoms} (mesh-referencing "
             "geoms without type='mesh' compile as fitted primitives)")

    for i in range(model.nu):
        name = model.actuator(i).name
        for suffix, expected_range in JOINT_CTRLRANGES.items():
            if name.endswith(f"_{suffix}"):
                break
        else:
            fail(
                f"actuator '{name}' does not end in "
                "_first_joint/_second_joint/_third_joint"
            )
        ctrlrange = model.actuator_ctrlrange[i]
        expected = np.array([-expected_range, expected_range])
        if not np.allclose(ctrlrange, expected):
            fail(
                f"actuator '{name}' ctrlrange {ctrlrange} != "
                f"expected {expected}"
            )


def simulate(model, data):
    data.ctrl[:] = 0.0
    n_steps = round(SIM_DURATION_S / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def check_physics(model, data):
    for name, arr in (("qpos", data.qpos), ("qvel", data.qvel), ("qacc", data.qacc)):
        if not np.all(np.isfinite(arr)):
            fail(f"{name} contains non-finite values after roll-out")

    trunk_z = data.body("root").xpos[2]
    if not (TRUNK_HEIGHT_MIN <= trunk_z <= TRUNK_HEIGHT_MAX):
        fail(
            f"trunk height {trunk_z:.4f} m outside expected band "
            f"[{TRUNK_HEIGHT_MIN}, {TRUNK_HEIGHT_MAX}] m"
        )

    max_closure_error = 0.0
    for leg in LEGS:
        foot_pos = data.body(f"{leg}_foot_link").xpos
        chain_pos = data.body(f"{leg}_chain_close_a_link").xpos
        error = np.linalg.norm(foot_pos - chain_pos)
        max_closure_error = max(max_closure_error, error)
        if error >= CLOSURE_ERROR_MAX:
            fail(
                f"leg '{leg}' four-bar closure error {error:.6f} m >= "
                f"{CLOSURE_ERROR_MAX} m"
            )

    return trunk_z, max_closure_error


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    check_structure(model)
    simulate(model, data)
    trunk_z, max_closure_error = check_physics(model, data)

    print("PASS")
    print(f"  nu: {model.nu}")
    print(f"  nq: {model.nq}")
    print(f"  final trunk height: {trunk_z:.4f} m")
    print(f"  max closure error: {max_closure_error:.6f} m")


if __name__ == "__main__":
    main()
