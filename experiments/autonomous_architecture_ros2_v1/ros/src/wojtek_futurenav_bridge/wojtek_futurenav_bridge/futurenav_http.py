"""Synchronous HTTP client for the FutureNav action server.

The server contract is the one `training/wojtek_rl/futurenav_server/` exposes:
``POST /reset {"instruction": ...}`` starts an episode (the server is stateful
-- frame history and VGGT KV cache live server-side) and ``POST /act
{"frame_b64": ...}`` returns one discrete VLN-CE action per call.

This is a deliberate, minimal re-statement of the client in
`wojtek_rl.futurenav_nav` rather than an import: that module pulls in the
demo-app stack (`wojtek_rl.vlm_nav`, loguru) which the ROS container venv
does not carry.  The action grid constants below must stay equal to the
originals -- `test_futurenav_http.py` cross-checks them against
`wojtek_rl.futurenav_nav` whenever the training project is importable.
"""

from __future__ import annotations

import json
import urllib.request

DEFAULT_FUTURENAV_URL = "http://127.0.0.1:8100"
# 4B generation is ~1-3 s per /act; margin covers CUDA warmup after a server
# restart and a remote link's hiccups.
REQUEST_TIMEOUT_S = 60.0

# FutureNav's training-time discrete grid (VLN-CE): metres per MOVE_FORWARD,
# degrees per TURN_*.  STOP means "instruction complete", not abort.
FORWARD_STEP_M = 0.25
TURN_STEP_DEG = 15.0

KNOWN_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


class FutureNavHttpError(RuntimeError):
    """Any transport or protocol failure talking to the action server."""


class FutureNavHttpClient:
    """Blocking client; the node calls it from a worker thread."""

    def __init__(self, url: str = DEFAULT_FUTURENAV_URL,
                 timeout_s: float = REQUEST_TIMEOUT_S):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except Exception as exc:  # URLError, timeout, bad JSON -- all fatal here
            raise FutureNavHttpError(f"POST {path}: {exc}") from exc

    def reset(self, instruction: str) -> None:
        self._post("/reset", {"instruction": instruction})

    def act(self, frame_b64: str) -> str:
        """One decision; returns the discrete action name."""
        out = self._post("/act", {"frame_b64": frame_b64})
        action = str(out.get("action", ""))
        if action not in KNOWN_ACTIONS:
            raise FutureNavHttpError(
                f"unknown action {action!r} (raw: {out.get('raw')!r})"
            )
        return action
