"""Resolve a policy reference to local policy.npz/policy_meta.json paths.

A reference is either
  - a local directory containing policy.npz + policy_meta.json, or
  - a Hugging Face model repo id, e.g.
        <HF_ORGANIZATION>/wojtek-springy-locomotion
        <HF_ORGANIZATION>/wojtek-stiff-kp80-locomotion@<commit-or-branch>
    downloaded with huggingface_hub (only the two policy files, never the
    checkpoint) into its standard cache, so after one online resolve the
    robot keeps working with no network. Private repos need a token
    (HF_TOKEN or `hf auth login`).

Pin real-robot launches to a commit (`@<sha>`): the resolved revision is
part of what ran on the robot, and a moving branch is not a record.

Prefetch on a machine that has network and a token:
    python3 -m wojtek_policy.policy_source <ref>

No ROS imports here; huggingface_hub is imported only when an HF ref is
actually used, so local-directory workflows need nothing extra installed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

FILES = ("policy.npz", "policy_meta.json")


@dataclass
class ResolvedPolicy:
    npz: Path
    meta: Path
    # Human-readable provenance for logs: "local:<dir>" or "hf:<repo>@<commit>".
    source: str


def resolve_policy(ref: str) -> ResolvedPolicy:
    ref = str(ref).strip()
    if not ref:
        raise ValueError(
            "empty policy reference -- set the `policy` parameter to a local "
            "directory or a Hugging Face repo id (org/name[@revision])"
        )
    as_path = Path(ref).expanduser()
    if as_path.is_dir():
        missing = [f for f in FILES if not (as_path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"policy directory {as_path} is missing {missing}"
            )
        return ResolvedPolicy(
            as_path / FILES[0], as_path / FILES[1], f"local:{as_path}"
        )
    if "/" not in ref or as_path.suffix:
        raise ValueError(
            f"policy reference {ref!r} is neither an existing directory nor "
            "a Hugging Face repo id (org/name[@revision])"
        )
    return _resolve_hf(ref)


def _resolve_hf(ref: str) -> ResolvedPolicy:
    from huggingface_hub import hf_hub_download

    try:  # moved between modules across huggingface_hub versions
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        from huggingface_hub.utils import LocalEntryNotFoundError

    repo_id, _, revision = ref.partition("@")
    revision = revision or None
    paths = {}
    for fname in FILES:
        try:
            paths[fname] = hf_hub_download(repo_id, fname, revision=revision)
        except LocalEntryNotFoundError as e:
            # Offline (or HF unreachable) and nothing cached: say how to fix
            # rather than surfacing a bare network error.
            raise RuntimeError(
                f"cannot reach Hugging Face and {ref!r} is not in the local "
                "cache -- prefetch on a networked machine with a token: "
                f"python3 -m wojtek_policy.policy_source {ref}"
            ) from e
    npz = Path(paths[FILES[0]])
    # Cache layout: .../snapshots/<commit>/<file>; the dir name is the
    # resolved commit, which makes branch refs auditable in the logs.
    commit = npz.parent.name
    return ResolvedPolicy(npz, Path(paths[FILES[1]]), f"hf:{repo_id}@{commit}")


def load_meta(ref: str) -> tuple[dict, str]:
    """(policy_meta dict, provenance) for a reference, without the weights.

    For launch files that need contract fields (e.g. the pd block) before
    any node starts; the npz is downloaded alongside but not parsed.
    """
    resolved = resolve_policy(ref)
    return json.loads(resolved.meta.read_text()), resolved.source


def pd_settings(meta: dict) -> dict:
    """Driver-side servo settings for a policy contract, taken verbatim.

    The MD80 impedance kp/kd and torque cap are the ones the policy trained
    against; they come straight from the contract's pd block.
    """
    pd = meta["pd"]
    return {
        "kp": float(pd["kp"]),
        "kd": float(pd["kd"]),
        "max_torque": float(pd["max_torque"]),
    }


@dataclass
class LoadedPolicy:
    """A resolved policy plus its parsed contract, loaded once for a launch.

    `directory` is what a node should get as its `policy` parameter: the
    resolved snapshot dir, where policy.npz and policy_meta.json sit together
    (in the HF cache too), so the node reads the same files without resolving
    the reference a second time. `source` is the provenance to pass through
    for logs so it stays readable ("hf:org/name@commit", not the cache path).
    """
    npz: Path
    meta_path: Path
    source: str
    meta: dict
    pd: dict
    run_name: str
    directory: Path


def load_policy(ref: str, overrides=None) -> LoadedPolicy:
    """Resolve a reference once and read everything a launch needs from it.

    pd is the contract's servo block; overrides ({kp,kd,max_torque}, float or
    string, empty string = keep the contract value) replace individual entries
    verbatim -- e.g. a low max_torque for cautious first tests.
    """
    resolved = resolve_policy(ref)
    meta = json.loads(resolved.meta.read_text())
    pd = pd_settings(meta)
    for key in ("kp", "kd", "max_torque"):
        override = (overrides or {}).get(key)
        if override is not None and str(override) != "":
            pd[key] = float(override)
    return LoadedPolicy(
        npz=resolved.npz,
        meta_path=resolved.meta,
        source=resolved.source,
        meta=meta,
        pd=pd,
        run_name=meta["run_name"],
        directory=resolved.npz.parent,
    )


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    resolved = resolve_policy(args[0])
    print(f"resolved {args[0]} -> {resolved.source}")
    print(f"  {resolved.npz}")
    print(f"  {resolved.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
