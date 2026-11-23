"""The drive gate: pad frames in, /cmd_vel values out, dead-man in between.

Pure Python on purpose (no ROS, no asyncio) so the safety logic has a
model-free unit test. The gateway node feeds it wall-clock seconds and
publishes whatever `tick` returns.

The rules are the ones gamepad_teleop already lives by, because policy_node
latches the last /cmd_vel it received and never times it out:

  * Nothing is published until the pad has spoken once. A resident gateway
    with no client must not spam zeros over another /cmd_vel source.
  * A frame older than `timeout_s` means the link is gone (tab closed, wifi
    drop, handheld asleep). The gate then publishes ZEROS -- the burst that
    overwrites the latched command -- for `silence_after_s`, and only then
    goes silent so another source can take over without being fought.
  * An explicit stop (the page said so, or the last client disconnected)
    is the same burst-then-silence, started right away.
  * The standing height is a held set-point: it survives stops, so the
    stance does not jump when the sticks are released.

Two kinds of sender share the gate. A person's pad (the handheld's page) and
the bridge, a program on another computer that drives for an AI system. The
person wins:

  * A frame from the bridge is ignored for `human_wins_s` after any frame
    from a pad. So a person who takes the sticks has the robot at once, and
    keeps it until they let go for that long.
  * The bridge moves the sticks and nothing else. Its frames cannot change
    the standing height, and BRIDGE_MAY lists the only messages the gateway
    takes from it: it cannot arm, stand the robot up or restart the stack.
  * `owner` says who the robot obeys now. The bridge reads it from the
    status frames and must not drive while it cannot.
"""

IDLE = "idle"        # nothing to publish
LIVE = "live"        # fresh pad frames, sticks drive
DEADMAN = "deadman"  # frames stopped: zeroing burst

# Who the robot obeys, as the status frame reports it.
OPERATOR = "operator"  # a person's pad
BRIDGE = "bridge"      # the bridge, driving for an AI system
NOBODY = "none"

# The only message types the gateway takes from the bridge.
BRIDGE_MAY = frozenset({"cmd", "stop"})


class DriveGate:
    def __init__(self, cmd_low, cmd_high, height_range, height_default,
                 timeout_s=0.5, silence_after_s=2.0, human_wins_s=1.0):
        self.cmd_low = [float(v) for v in cmd_low]
        self.cmd_high = [float(v) for v in cmd_high]
        self.height_range = (float(height_range[0]), float(height_range[1]))
        self.height = float(height_default)
        self.timeout_s = float(timeout_s)
        self.silence_after_s = float(silence_after_s)
        self.human_wins_s = float(human_wins_s)
        self._cmd = (0.0, 0.0, 0.0)   # normalized [-1, 1] (vx, vy, yaw)
        self._stamp = None            # seconds, of the last pad frame
        self._human_stamp = None      # seconds, of the last frame from a person
        self._from_bridge = False     # who sent the frame being obeyed
        self.state = IDLE

    # -- input ---------------------------------------------------------------
    def command(self, now, vx, vy, yaw, height=None, bridge=False):
        """A pad frame: normalized sticks, optional absolute height (m).

        `bridge` marks a frame from the bridge. It is dropped while a
        person is driving, and it never changes the height. Returns whether
        the frame was taken.
        """
        now = float(now)
        if bridge and self.human_driving(now):
            return False
        clip = lambda v: max(-1.0, min(1.0, float(v)))  # noqa: E731
        self._cmd = (clip(vx), clip(vy), clip(yaw))
        if height is not None and not bridge:
            lo, hi = self.height_range
            self.height = max(lo, min(hi, float(height)))
        if not bridge:
            self._human_stamp = now
        self._from_bridge = bool(bridge)
        self._stamp = now
        return True

    def human_driving(self, now):
        """A person's pad sent a frame within the last `human_wins_s`."""
        return (self._human_stamp is not None
                and float(now) - self._human_stamp <= self.human_wins_s)

    def owner(self, now):
        """Who the robot obeys now: OPERATOR, BRIDGE or NOBODY."""
        if self.human_driving(now):
            return OPERATOR
        fresh = (self._stamp is not None
                 and float(now) - self._stamp < self.timeout_s)
        return BRIDGE if fresh and self._from_bridge else NOBODY

    def client_gone(self, now, bridge, pads_left):
        """A client disconnected: stop what it alone was commanding.

        The bridge leaving stops the robot if the bridge was driving it. A
        pad leaving stops it if no other pad is connected and a pad was
        driving. One sender going away never stops the other's motion.
        """
        if bridge:
            if self._from_bridge:
                self.stop(now)
        elif pads_left == 0 and not self._from_bridge:
            self.stop(now)

    def stop(self, now):
        """Explicit stop: start the zeroing burst now, keep the height."""
        if self._stamp is None:
            return  # never drove: stay silent, there is nothing to undo
        self._cmd = (0.0, 0.0, 0.0)
        # Backdate the stamp so the very next tick sees a stale frame.
        self._stamp = min(self._stamp, float(now) - self.timeout_s)

    # -- output --------------------------------------------------------------
    def tick(self, now):
        """(vx, vy, yaw, height) to publish this tick, or None for silence.

        Updates `state` as a side effect; the node reports its changes.
        """
        if self._stamp is None:
            self.state = IDLE
            return None
        age = float(now) - self._stamp
        if age > self.timeout_s + self.silence_after_s:
            self.state = IDLE
            return None
        if age >= self.timeout_s:
            self.state = DEADMAN
            return (0.0, 0.0, 0.0, self.height)
        self.state = LIVE
        # Normalized [-1, 1] -> the trained (asymmetric) command box:
        # positive stick scales by high, negative by low.
        vx, vy, yaw = (v * self.cmd_high[i] if v >= 0 else v * -self.cmd_low[i]
                       for i, v in enumerate(self._cmd))
        return (vx, vy, yaw, self.height)

    def step_height(self, delta_m):
        lo, hi = self.height_range
        self.height = max(lo, min(hi, self.height + float(delta_m)))
        return self.height
