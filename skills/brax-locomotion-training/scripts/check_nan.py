#!/usr/bin/env python3
"""Check Brax/Orbax PPO checkpoints for non-finite array values.

Usage: check_nan.py RUN_DIR [--all]

Exit 0 means the selected checkpoint data is finite. Exit 1 means at least one
selected checkpoint contains NaN or Inf. Argument and filesystem errors exit 2.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--all", action="store_true", help="check every checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_root = args.run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        print(f"checkpoint directory not found: {checkpoint_root}", file=sys.stderr)
        return 2

    checkpoints = sorted(
        (path for path in checkpoint_root.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not checkpoints:
        print(
            f"no numeric checkpoint directories under {checkpoint_root}",
            file=sys.stderr,
        )
        return 2

    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    selected = checkpoints if args.all else checkpoints[-1:]
    corrupted = 0
    for checkpoint in selected:
        params = ppo_checkpoint.load(str(checkpoint.resolve()))
        leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(params)
            if hasattr(leaf, "dtype")
        ]
        bad = sum(bool((~jnp.isfinite(leaf)).any()) for leaf in leaves)
        corrupted += bad > 0
        print(f"{checkpoint.name}: {len(leaves)} arrays, {bad} with NaN/Inf")

    return 1 if corrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
