"""The /cmd_vel dead-man: the latest command, zeroed when its source goes quiet.

policy_node used to hold the last velocity command forever. Every drive
source (pad, consoles, text_commander, goto, Nav2) publishes at 20 Hz or
more and sends an explicit zero when it stops, so a *living* source never
notices this gate. What it catches is a source that dies mid-walk: measured
in the simulation on 2026-09-25, a killed navigation stack left the robot
walking on with no one to stop it. With the gate the velocities read zero
`timeout_s` after the last message.

Pure logic, no ROS: the node feeds it commands and a clock, the tests feed
it numbers.
"""

import numpy as np


class CommandGate:
    """Latest command in, the command to track out.

    `fill` is the contract's fill for the extra command channels (the
    standing height of a 4-D policy); before the first message the gate
    answers zero velocities plus that fill, exactly what the node started
    with before the gate existed.
    """

    def __init__(self, timeout_s, fill=None):
        self.timeout_s = float(timeout_s)
        fill = np.zeros(0) if fill is None else np.asarray(fill, dtype=float)
        self._cmd = np.concatenate([np.zeros(3), fill])
        self._stamp = None  # seconds; None until the first command

    def update(self, cmd, now):
        self._cmd = np.array(cmd, dtype=float, copy=True)
        self._stamp = float(now)

    def current(self, now):
        """(command, expired). `expired` is True only while a real command
        has aged past the timeout; a gate that never received one is idle,
        not faulted. A timeout of 0 (or less) disables the gate."""
        if self._stamp is None or self.timeout_s <= 0.0:
            return self._cmd.copy(), False
        if float(now) - self._stamp <= self.timeout_s:
            return self._cmd.copy(), False
        cmd = self._cmd.copy()
        cmd[:3] = 0.0  # stop; a commanded height is kept (no drop on timeout)
        return cmd, True
