"""Load the exported NumPy policy runtime (piesek_ws fbb_policy) without ROS.

The deploy workspace ships a pure-NumPy WojtekPolicy (obs assembly + MLP + gait
clock, no ROS imports) plus the exported fbb_loco_v8 weights. Reuse it here
for plain-MuJoCo apps instead of duplicating the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wojtek_rl import paths


def _bootstrap() -> None:
    pkg = str(paths.WOJTEK_POLICY_PKG)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)


def load_fbb_policy(npz: Path = paths.POLICY_NPZ, meta: Path | None = None):
    """WojtekPolicy from the exported .npz (+ its policy_meta.json)."""
    _bootstrap()
    from wojtek_policy.policy import WojtekPolicy

    return WojtekPolicy(npz, meta_path=meta)


def gravity_from_quat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    _bootstrap()
    from wojtek_policy.policy import gravity_from_quat as impl

    return impl(qw, qx, qy, qz)


def actuator_addresses(model) -> tuple[np.ndarray, np.ndarray]:
    """(qpos, qvel) addresses of the actuated joints, in actuator order."""
    joint_ids = model.actuator_trnid[: model.nu, 0]
    qadr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    vadr = np.array([model.jnt_dofadr[j] for j in joint_ids])
    return qadr, vadr
