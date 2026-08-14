"""Resolve a policy reference to local policy.npz/policy_meta.json paths.

A reference is either
  - a local directory containing policy.npz + policy_meta.json, or
  - a Hugging Face model repo id, e.g.
        <HF_ORGANIZATION>/wojtek-springy-locomotion
        <HF_ORGANIZATION>/wojtek-stiff-kp80-locomotion@<commit-or-branch>
    resolved against the POLICY STORE: a plain directory of downloaded
    snapshots, one per commit, laid out as

        <store>/<org>/<name>/<commit>/policy.npz + policy_meta.json
        <store>/<org>/<name>/refs/<branch-or-tag>   # the commit it pointed
                                                    # at when last fetched

    The store is WOJTEK_POLICY_STORE, or `policies/` next to the workspace's
    `src/` (ros/policies in a checkout), or ~/.wojtek/policies.

The robot never talks to Hugging Face -- it has no internet, and
huggingface_hub is not even installed on it. Downloading happens on the
operator PC, which has the network and the token for the private repos:

    python3 -m wojtek_policy.policy_source --default  # the pin below
    python3 -m wojtek_policy.policy_source <ref>      # any other reference
    ./deploy.sh                                       # syncs it to the robot

so on the robot every reference is answered from the store, offline. A ref
that is not there yet fails with that same instruction instead of a network
error. Only the two policy files are ever fetched, never the checkpoint.

Nobody has to run this by hand for the normal flow: deploy.sh runs
`--default` itself before syncing the store, so the policy the launch files
come up with is in place. The bare-<ref> form is for a one-off policy that
is not the default.

Pin real-robot launches to a commit (`@<sha>`): the resolved revision is
part of what ran on the robot, and a moving branch is not a record. A commit
is also the one reference that needs no network at all once stored, since a
commit never moves; a branch is re-fetched when Hugging Face is reachable.

No ROS imports here; huggingface_hub is imported only when something has to
be downloaded, so store and local-directory workflows need nothing extra
installed.
"""

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

FILES = ("policy.npz", "policy_meta.json")


class _HFUnavailable(Exception):
    """Nothing can be downloaded from here: no huggingface_hub, or no network.

    Not an error by itself -- it is the signal to answer from the store.
    A refused or unauthorized repo is a different thing and propagates.
    """


@dataclass
class ResolvedPolicy:
    npz: Path
    meta: Path
    # Human-readable provenance for logs: "local:<dir>" or "hf:<repo>@<commit>".
    source: str


def policy_store() -> Path:
    """Where downloaded policy snapshots live on this machine.

    WOJTEK_POLICY_STORE wins when set (the container sets it). Otherwise it
    sits beside the workspace's `src/`, found by walking up from this file:
    ros/policies in a checkout, ~/wojtek_ws/policies on the robot. That works
    because every build here is --symlink-install, so this file resolves back
    into the source tree rather than into install/. The home directory is the
    last resort, for a copy installed some other way.
    """
    env = os.environ.get("WOJTEK_POLICY_STORE", "").strip()
    if env:
        return Path(env).expanduser()
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent / "policies"
    return Path.home() / ".wojtek" / "policies"


# Which policy every bringup -- robot and simulation -- comes up with when no
# policy:= is given. Pinned to a commit on purpose: an unpinned repo id follows
# whatever main is, so what walks today would silently be a different policy
# tomorrow; a commit is also the one reference the robot can answer from the
# store with no network. It lives here rather than in the launch files because
# host-side tooling (deploy.sh) has to resolve it too, and this module is
# stdlib-only.
_DEFAULT_REPO = ("wojtek-quiet-locomotion"
                 "@553795b13001cc1f519a4abc0235f275095129f8")


def default_policy() -> str:
    """The pinned default reference, or "" when there is no organization.

    The org comes from HF_ORGANIZATION in the environment (see .env.example),
    because the keeper repos are private and this repository is public.
    Without it there is no default and every launch needs an explicit
    policy:=.
    """
    org = os.environ.get("HF_ORGANIZATION", "").strip()
    return f"{org}/{_DEFAULT_REPO}" if org else ""


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


def _is_commit(revision: str) -> bool:
    """A full commit sha, as opposed to a branch or tag name."""
    return re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None


def _from_store(store: Path, repo_id: str, commit: str) -> ResolvedPolicy | None:
    """The stored snapshot for a commit, or None if it isn't there (whole)."""
    snapshot = store / repo_id / commit
    if not all((snapshot / f).is_file() for f in FILES):
        return None
    return ResolvedPolicy(
        snapshot / FILES[0], snapshot / FILES[1], f"hf:{repo_id}@{commit}"
    )


def _fetch_into_store(store: Path, repo_id: str, revision: str) -> ResolvedPolicy:
    """Download the two policy files and materialize them in the store.

    Raises _HFUnavailable when downloading is not possible here at all.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        # The robot deliberately doesn't have it -- see the module docstring.
        raise _HFUnavailable("huggingface_hub is not installed") from e

    try:  # moved between modules across huggingface_hub versions
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        from huggingface_hub.utils import LocalEntryNotFoundError

    paths = {}
    for fname in FILES:
        try:
            paths[fname] = Path(
                hf_hub_download(repo_id, fname, revision=revision)
            )
        except LocalEntryNotFoundError as e:
            # Offline (or HF unreachable) and nothing in the download cache.
            # Everything else -- a missing repo, a bad or absent token -- is a
            # real answer from HF and must not be mistaken for being offline.
            raise _HFUnavailable(
                f"cannot reach Hugging Face for {repo_id}@{revision}"
            ) from e

    # Download cache layout: .../snapshots/<commit>/<file>; the dir name is the
    # resolved commit, which makes branch refs auditable in the logs.
    commit = paths[FILES[0]].parent.name
    snapshot = store / repo_id / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        # copyfile, not a link: in the download cache these are symlinks into
        # a blob directory, and the store must hold real files -- it gets
        # rsynced to the robot and read by hand.
        shutil.copyfile(paths[fname], snapshot / fname)
    if revision != commit:
        # Remember where the branch/tag pointed, so an offline machine can
        # still answer that name. A branch with a "/" just nests.
        recorded = store / repo_id / "refs" / revision
        recorded.parent.mkdir(parents=True, exist_ok=True)
        recorded.write_text(commit + "\n")
    return ResolvedPolicy(
        snapshot / FILES[0], snapshot / FILES[1], f"hf:{repo_id}@{commit}"
    )


def _not_in_store(ref: str, store: Path) -> RuntimeError:
    return RuntimeError(
        f"{ref!r} is not in the policy store ({store}) and Hugging Face is "
        "not reachable from here -- prefetch on the operator PC: "
        f"python3 -m wojtek_policy.policy_source {ref}; then ./deploy.sh "
        "syncs the store to the robot"
    )


def _resolve_hf(ref: str) -> ResolvedPolicy:
    repo_id, _, revision = ref.partition("@")
    store = policy_store()

    if _is_commit(revision):
        # A commit never moves, so a stored snapshot is the right answer for
        # good and there is nothing to check over the network.
        stored = _from_store(store, repo_id, revision)
        if stored is not None:
            return stored
        try:
            return _fetch_into_store(store, repo_id, revision)
        except _HFUnavailable as e:
            raise _not_in_store(ref, store) from e

    # A branch or tag moves, so ask Hugging Face first when we can -- that is
    # what keeps a desk workflow following the branch. No revision means the
    # default branch.
    revision = revision or "main"
    try:
        return _fetch_into_store(store, repo_id, revision)
    except _HFUnavailable as e:
        recorded = store / repo_id / "refs" / revision
        if recorded.is_file():
            stored = _from_store(store, repo_id, recorded.read_text().strip())
            if stored is not None:
                return stored
        raise _not_in_store(ref, store) from e


def load_meta(ref: str) -> tuple[dict, str]:
    """(policy_meta dict, provenance) for a reference, without the weights.

    For launch files that need contract fields (e.g. the pd block) before
    any node starts; the npz is resolved alongside but not parsed.
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
    (in the policy store too), so the node reads the same files without
    resolving the reference a second time. `source` is the provenance to pass
    through for logs so it stays readable ("hf:org/name@commit", not the
    store path).
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
    # One reference, or --default; nothing else is a usage this understands.
    if len(args) != 1 or (args[0].startswith("-") and args[0] != "--default"):
        print(__doc__)
        return 2
    ref = args[0]
    if ref == "--default":
        ref = default_policy()
        if not ref:
            print(
                "no default policy: HF_ORGANIZATION is not set -- put it in "
                ".env (see .env.example), or name a reference instead"
            )
            return 2
    resolved = resolve_policy(ref)
    print(f"resolved {ref} -> {resolved.source}")
    print(f"  store: {policy_store()}")
    print(f"  {resolved.npz}")
    print(f"  {resolved.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
