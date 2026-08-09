"""Load the exported NumPy policy runtime (ros/ wojtek_policy) without ROS.

The deploy workspace ships a pure-NumPy WojtekPolicy (obs assembly + MLP,
no ROS imports) and a resolver that turns a policy reference -- a local
directory, a path to a policy.npz, or a Hugging Face repo id like
<HF_ORGANIZATION>/wojtek-springy-locomotion[@rev] -- into artifact paths.
Reuse both here for plain-MuJoCo apps instead of duplicating the pipeline.
"""

from __future__ import annotations

import os
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
        # WOJTEK_POLICY overrides the default keeper for sim/demo processes
        # without touching paths.DEFAULT_POLICY (which the deploy tooling and
        # the export contract read). Temporary escape hatch while the
        # published keeper does not walk -- see wojtek_rl.legacy_policy.
        ref = os.environ.get("WOJTEK_POLICY") or paths.DEFAULT_POLICY
    if ref is None:
        raise ValueError(
            "no policy reference: pass one explicitly, or set WOJTEK_POLICY "
            "or HF_ORGANIZATION (see .env.example)"
        )
    as_path = Path(ref).expanduser() if not isinstance(ref, Path) else ref
    if as_path.suffix == ".npz":
        npz, meta_file = as_path, meta
    else:
        resolved = resolve_policy(str(ref))
        npz, meta_file = resolved.npz, (meta or resolved.meta)
    if _is_legacy(npz, meta_file):
        # TEMPORARY: schema-1 exports carrying the gait clock (fbb_loco_v8).
        # See wojtek_rl.legacy_policy -- sim only, delete with that module.
        from wojtek_rl.legacy_policy import load as load_legacy

        return load_legacy(npz, meta_file)
    return WojtekPolicy(npz, meta_path=meta_file)


def _is_legacy(npz: Path, meta_file: Path | None) -> bool:
    import json

    from wojtek_rl.legacy_policy import is_legacy_meta

    path = Path(meta_file) if meta_file else Path(npz).with_name("policy_meta.json")
    if not path.exists():
        return False
    return is_legacy_meta(json.loads(path.read_text()))


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
