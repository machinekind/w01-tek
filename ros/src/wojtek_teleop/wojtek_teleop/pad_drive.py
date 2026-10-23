"""The pad's drive gate: joy frames in, /cmd_vel values out, quiet when idle.

Pure Python on purpose (no ROS) so the behaviour has a model-free unit
test. gamepad_teleop feeds it clock seconds and publishes what `tick`
returns.

The joy driver repeats the pad's state at a steady rate even when nobody
touches it, so "there is a joy message" does not mean "somebody is
driving". This gate makes the distinction. It follows the rule the other
/cmd_vel sources already live by (the deck gateway's DriveGate, the text
commander): publish while there is input, zero the motion for a while when
it stops, then go silent so another source can take /cmd_vel without being
shouted over.

  * Nothing is published until the sticks leave the deadzone once. A pad
    that sits idle next to the robot is not a driver.
  * While the sticks are deflected, or were within `timeout_s`, the gate is
    LIVE and returns the scaled command.
  * Once `timeout_s` passes with the sticks centred or the joy stream gone,
    the gate ZEROES the motion for `silence_after_s` -- the burst that
    overwrites the command policy_node latched -- and then goes IDLE and
    returns None.
  * A height step counts as input: the new set-point has to reach the
    policy even with the sticks centred, so it opens a zeroing burst that
    carries it.
  * `pad_lost` tells the two ZEROING causes apart for the log: the joy
    stream stopped (pad off, out of range, driver gone) versus the sticks
    just went back to centre.
"""

IDLE = "idle"        # nothing to publish
LIVE = "live"        # sticks drive
ZEROING = "zeroing"  # input stopped: zeroing burst


class PadDrive:
    def __init__(self, cmd_low, cmd_high, height_range, height_default,
                 timeout_s=0.5, silence_after_s=2.0):
        self.cmd_low = [float(v) for v in cmd_low]
        self.cmd_high = [float(v) for v in cmd_high]
        self.height_range = (float(height_range[0]), float(height_range[1]))
        self.height = float(height_default)
        self.timeout_s = float(timeout_s)
        self.silence_after_s = float(silence_after_s)
        self._cmd = (0.0, 0.0, 0.0)   # normalized [-1, 1] (vx, vy, yaw)
        self._joy_stamp = None        # last joy frame of any kind
        self._input_stamp = None      # last frame with input worth driving on
        self.state = IDLE

    # -- input ---------------------------------------------------------------
    def joy(self, now, vx, vy, yaw):
        """A joy frame, sticks already shaped through the deadzone."""
        self._cmd = (float(vx), float(vy), float(yaw))
        self._joy_stamp = float(now)
        if any(v != 0.0 for v in self._cmd):
            self._input_stamp = float(now)

    def step_height(self, now, delta_m):
        lo, hi = self.height_range
        self.height = max(lo, min(hi, self.height + float(delta_m)))
        self._input_stamp = float(now)
        return self.height

    # -- output --------------------------------------------------------------
    def pad_lost(self, now):
        """True when the joy stream itself has stopped, not just the sticks."""
        return (self._joy_stamp is not None
                and float(now) - self._joy_stamp > self.timeout_s)

    def tick(self, now):
        """(vx, vy, yaw, height) to publish this tick, or None for silence.

        Updates `state` as a side effect; the node reports its changes.
        """
        if self._input_stamp is None:
            self.state = IDLE
            return None
        age = float(now) - self._input_stamp
        if age > self.timeout_s + self.silence_after_s:
            self.state = IDLE
            return None
        if age > self.timeout_s or self.pad_lost(now):
            self.state = ZEROING
            return (0.0, 0.0, 0.0, self.height)
        self.state = LIVE
        # Normalized [-1, 1] -> the trained (asymmetric) command box:
        # positive stick scales by high, negative by low.
        vx, vy, yaw = (v * self.cmd_high[i] if v >= 0 else v * -self.cmd_low[i]
                       for i, v in enumerate(self._cmd))
        return (vx, vy, yaw, self.height)
