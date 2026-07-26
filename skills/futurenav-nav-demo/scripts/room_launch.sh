#!/usr/bin/env bash
# Wrapper so the room app gets the private-org HF token: .claude/launch.json has
# no `env` field, and the machine-level `hf auth login` identity may not be the
# account with access to <HF_ORGANIZATION>.
#
# Use as the launch.json runtimeExecutable (absolute path), with the room args
# in runtimeArgs:
#
#   {"name": "room-demo",
#    "runtimeExecutable": "/abs/path/to/room_launch.sh",
#    "runtimeArgs": ["--scene-name", "apartment",
#                    "--vlm-backend", "futurenav",
#                    "--vlm-url", "http://127.0.0.1:8100"],
#    "port": 8010}
#
# REPO must be the main checkout: worktrees lack .venv, the built scene_*.xml,
# and the scan assets.
set -euo pipefail

REPO=${REPO:?"set REPO to the main checkout path"}
export HF_TOKEN=$(grep '^HUGGINGFACE_API_KEY=' "$REPO/.env" | cut -d= -f2)
exec "$REPO/training/run.sh" room "$@"
