"""Command ownership and receipt-time leases, without ROS dependencies.

Only the current owner's command is retained. Rejected commands are discarded,
so releasing a lease never replays a waiting source. A manual acquisition
inhibits autonomy until the operator explicitly enables it again.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Source:
    priority: int
    manual: bool


SOURCES = {
    "pad": Source(100, True),
    "deck": Source(90, True),
    "console": Source(80, True),
    "external": Source(70, True),
    "brain": Source(30, False),
    "nav": Source(20, False),
    "text": Source(10, False),
}


@dataclass(frozen=True)
class Decision:
    # Twist order: linear x/y/z, angular x/y/z. linear.z is stance height.
    command: tuple[float, ...] | None
    cancel_autonomy: bool = False


class CommandMux:
    """Fixed-priority ownership with an exclusive, refreshable lease.

    A higher-priority source may preempt. A lower-priority source must send
    again after the owner expires; nothing queues behind the owner. Zero
    commands refresh ownership too, so sources must go silent when finished.
    """

    def __init__(self, timeout_s=0.5):
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.owner = None
        self.autonomy_enabled = False
        self.reason = "startup"
        self.rejected = 0
        self._command = None
        self._received_at = None
        self._last_now = None
        self._height = 0.0
        self._pending_stop = False
        self._pending_cancel = False

    def _reject(self, reason):
        self.reason = reason
        self.rejected += 1
        return False

    def _release(self, reason):
        if self.owner is not None:
            self._pending_stop = True
        self.owner = None
        self._command = None
        self._received_at = None
        self.reason = reason

    def _advance(self, now):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self._last_now is not None and now < self._last_now:
            self._release("clock_regressed")
            self.autonomy_enabled = False
            self._pending_cancel = True
            self._last_now = now
            return False
        self._last_now = now
        if self.owner is not None and now - self._received_at >= self.timeout_s:
            self._release("owner_expired")
        return True

    def receive(self, source, command, now):
        """Accept a fresh six-component Twist, or discard it without a lease."""
        if not self._advance(now):
            return self._reject("clock_regressed")
        if source not in SOURCES:
            return self._reject("unknown_source")
        try:
            command = tuple(float(value) for value in command)
        except (TypeError, ValueError, OverflowError):
            return self._reject("invalid_command")
        if len(command) != 6 or not all(math.isfinite(value) for value in command):
            return self._reject("invalid_command")
        candidate = SOURCES[source]
        if not candidate.manual and not self.autonomy_enabled:
            return self._reject("autonomy_inhibited")
        if self.owner is not None and self.owner != source:
            if candidate.priority <= SOURCES[self.owner].priority:
                return self._reject(f"owned_by_{self.owner}")
        acquired = source != self.owner
        if acquired and candidate.manual:
            self.autonomy_enabled = False
            self._pending_cancel = True
        self.owner = source
        self._command = command
        self._received_at = float(now)
        self.reason = "manual_takeover" if acquired and candidate.manual else "command"
        return True

    def set_autonomy_enabled(self, enabled, now):
        """Enable only without a manual owner; disabling also cancels the task."""
        if not self._advance(now):
            return False
        if not enabled:
            self.cancel_autonomy(now)
            return True
        if self.owner is not None and SOURCES[self.owner].manual:
            return self._reject("manual_owner_active")
        if self._pending_cancel:
            return self._reject("cancel_pending")
        self.autonomy_enabled = True
        self.reason = "autonomy_enabled"
        return True

    def cancel_autonomy(self, now, *, notify=True):
        """Inhibit autonomy. notify=False handles a cancel topic without echo."""
        self._advance(now)
        self.autonomy_enabled = False
        if self.owner is not None and not SOURCES[self.owner].manual:
            self._release("autonomy_cancelled")
        else:
            self.reason = "autonomy_cancelled"
        self._pending_cancel |= notify

    def step(self, now):
        """Current command, one stop after expiry/cancel, then silence.

        A stop keeps the last emitted stance-height channel. If a new source
        arrived after expiry but before this tick, the stop is emitted first;
        the new command can follow on the next tick if it is still fresh.
        """
        self._advance(now)
        cancel = self._pending_cancel
        self._pending_cancel = False
        if self._pending_stop:
            self._pending_stop = False
            return Decision((0.0, 0.0, self._height, 0.0, 0.0, 0.0), cancel)
        if self._command is not None:
            self._height = self._command[2]
        return Decision(self._command, cancel)

    def status(self):
        return {
            "schema": 1,
            "owner": self.owner,
            "autonomy_enabled": self.autonomy_enabled,
            "reason": self.reason,
            "timeout_s": self.timeout_s,
            "rejected": self.rejected,
        }
