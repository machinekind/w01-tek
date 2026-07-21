"""Regenerate a published keeper's policy_meta.json as a schema-2 contract.

Keepers exported before the schema-2 contract carry incomplete metas (no
command box, no clamps, no resolved anchor). This tool rebuilds the meta
from the keeper's own run.json -- already on HF next to the checkpoint --
with the same deploy_contract code the exporter now uses, so old keepers
become loadable by name exactly like fresh exports.

Dry-run (default) prints the old->new diff per repo; --apply uploads the
regenerated policy_meta.json back to the repo.

  ./run.sh python -m wojtek_rl.migrate_keeper_meta            # all known keepers
  ./run.sh python -m wojtek_rl.migrate_keeper_meta --repo <HF_ORGANIZATION>/x --apply
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

KNOWN_KEEPERS = (
    "<HF_ORGANIZATION>/wojtek-springy-locomotion",
    "<HF_ORGANIZATION>/wojtek-springy-locomotion-v2",
    "<HF_ORGANIZATION>/wojtek-stiff-locomotion",
    "<HF_ORGANIZATION>/wojtek-stiff-kp80-locomotion",
    "<HF_ORGANIZATION>/wojtek-stiff-kp90-locomotion",
)


def migrate(repo_id: str, apply: bool) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    from wojtek_rl.deploy_contract import build_contract
    from wojtek_rl.export_policy import build_env
    from wojtek_rl.np_policy import load_policy_runtime

    run = json.loads(Path(hf_hub_download(repo_id, "run.json")).read_text())
    old_meta = json.loads(
        Path(hf_hub_download(repo_id, "policy_meta.json")).read_text()
    )
    npz = Path(hf_hub_download(repo_id, "policy.npz"))

    env = build_env(run)
    meta = build_contract(env, run, checkpoint=old_meta.get("checkpoint", ""))

    # The regenerated contract must describe the same artifact the repo
    # already ships: same network input width, same actuator order, same
    # home pose (a mismatch means the local model XML no longer matches the
    # keeper's generation -- do NOT upload such a meta).
    norm_mean = np.load(npz)["norm_mean"]
    assert meta["obs_size"] == norm_mean.shape[0], (
        f"{repo_id}: contract obs_size {meta['obs_size']} != policy.npz "
        f"normalizer width {norm_mean.shape[0]}"
    )
    assert meta["run_name"] == old_meta["run_name"], (
        f"{repo_id}: run_name mismatch {meta['run_name']} vs {old_meta['run_name']}"
    )
    assert meta["actuator_names"] == old_meta["actuator_names"]
    assert np.allclose(meta["home_ctrl"], old_meta["home_ctrl"], atol=1e-6)

    out_dir = Path(tempfile.mkdtemp(prefix=f"migrate_{repo_id.split('/')[-1]}_"))
    out_meta = out_dir / "policy_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")

    # Smoke: the deploy runtime must load and step the migrated pair.
    policy = load_policy_runtime(npz, meta=out_meta)
    targets = policy.step(
        np.zeros(3), [0.0, 0.0, -1.0], policy.home_ctrl, np.zeros(12),
        [0.3, 0.0, 0.0],
    )
    assert np.all(np.isfinite(targets))

    print(f"== {repo_id}")
    for key in sorted(set(old_meta) | set(meta)):
        o, n = old_meta.get(key), meta.get(key)
        if o != n:
            print(f"  {key}: {o!r} -> {n!r}")

    if apply:
        HfApi().upload_file(
            path_or_fileobj=out_meta,
            path_in_repo="policy_meta.json",
            repo_id=repo_id,
            commit_message="policy_meta.json: schema-2 deploy contract "
            "(regenerated from run.json by wojtek_rl.migrate_keeper_meta)",
        )
        print(f"  uploaded to {repo_id}")
    else:
        print(f"  dry-run only; regenerated meta at {out_meta}")
    return out_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", action="append", default=None,
        help="HF repo id; repeatable (default: all known keepers)",
    )
    ap.add_argument("--apply", action="store_true", help="upload the new meta")
    args = ap.parse_args()
    for repo_id in args.repo or KNOWN_KEEPERS:
        migrate(repo_id, apply=args.apply)


if __name__ == "__main__":
    main()
