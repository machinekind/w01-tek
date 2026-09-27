"""How far the robot's clock is from the DGX's, read off the traffic. Pure.

The pixel resolver on the DGX looks the robot's pose up at the stamp of the
camera's picture, so the robot Pi, the camera Pi and the DGX must agree on
the time: 0.1 s of disagreement at 0.3 m/s puts the target 3 cm off, a
second puts it a step off, and a clock set to boot time puts it nowhere.

Every IN message carries the robot's stamp; the relay notes when it
arrived on the DGX. receive - stamp = clock offset + transport delay, and
the transport delay is never negative, so the smallest value over a
window is the offset plus the fastest delivery -- within a few ms of the
offset on a quiet link. The relay reports it; it does not correct anything.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple


class OffsetEstimator:
    def __init__(self, window_s: float = 10.0) -> None:
        self.window_s = float(window_s)
        self._samples: Deque[Tuple[float, float]] = deque()   # (local receive s, receive - stamp)

    def add(self, local_receive_s: float, remote_stamp_s: float) -> None:
        if remote_stamp_s <= 0.0:          # an unstamped message says nothing about the clock
            return
        self._samples.append((local_receive_s, local_receive_s - remote_stamp_s))
        cutoff = local_receive_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def offset_s(self) -> Optional[float]:
        """DGX clock minus robot clock (s), plus the fastest delivery; None
        before the first stamped message."""
        if not self._samples:
            return None
        return min(d for _t, d in self._samples)
