"""FutureNav client that remembers what the server actually said.

`wojtek_rl.futurenav_nav.FutureNavVlmClient` returns a mid-level decision and
drops the rest of the response.  The demo UI wants the raw VLA action text and
the per-stage timings so it can show what the model emitted, not just the
command it was translated into.

This subclass captures that into `last_meta`, which `TracedVlmNavigator` copies
into the trace (it reads the attribute defensively, so a plain client still
works).  Subclassing keeps `wojtek_rl` free of demo-only instrumentation.
"""

from __future__ import annotations

from wojtek_rl.futurenav_nav import FutureNavVlmClient


class TracedFutureNavVlmClient(FutureNavVlmClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_meta: dict | None = None

    def _post(self, path: str, payload: dict) -> dict:
        out = super()._post(path, payload)
        if path == "/act":
            self.last_meta = {
                "action": out.get("action"),
                "raw": out.get("raw"),
                "server_step": out.get("step"),
                "timings": out.get("timings"),
            }
        return out
