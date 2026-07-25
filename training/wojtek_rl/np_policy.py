"""Load the exported NumPy policy runtime (ros/ wojtek_policy) without ROS.

The deploy workspace ships a pure-NumPy WojtekPolicy (obs assembly + MLP,
no ROS imports) and a resolver that turns a policy reference -- a local
directory, a path to a policy.npz, or a Hugging Face repo id like
<HF_ORGANIZATION>/wojtek-springy-locomotion[@rev] -- into artifact paths.
Reuse both here for plain-MuJoCo apps instead of duplicating the pipeline.
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


def load_policy_runtime(ref: str | Path | None = None, meta: Path | None = None):
    """WojtekPolicy from a policy reference (default: paths.DEFAULT_POLICY).

    `ref` may be a Hugging Face repo id, a directory holding policy.npz +
    policy_meta.json, or a direct path to a policy.npz (exporter output;
    the meta is found next to it unless `meta` is given).
    """
    _bootstrap()
    from wojtek_policy.policy import WojtekPolicy
    from wojtek_policy.policy_source import resolve_policy

    if ref is None:
        ref = paths.DEFAULT_POLICY
    as_path = Path(ref).expanduser() if not isinstance(ref, Path) else ref
    if as_path.suffix == ".npz":
        return WojtekPolicy(as_path, meta_path=meta)
    resolved = resolve_policy(str(ref))
    return WojtekPolicy(resolved.npz, meta_path=meta or resolved.meta)


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
