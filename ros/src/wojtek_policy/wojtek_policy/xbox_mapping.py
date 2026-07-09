"""Gamepad-axes -> policy-command mapping, ROS-free so it is unit-testable.

Axes follow the ROS joy convention: +1.0 = stick pushed up (Y axes) or
left (X axes), matching REP 103 (x forward, y left, z yaw CCW), so no
sign flips are needed for a standard driver.

Velocity axes go through a deadzone (rescaled so output is continuous at
the deadzone edge) and then onto the trained command box, which may be
asymmetric (fbb_loco_v8: vx in [-0.8, 1.2]) -- full forward gives
cmd_high, full back gives cmd_low.

The height axis is a *rate*: holding the stick fully up raises the
commanded height by height_rate m/s, integrated by step_height().
"""

import numpy as np


class JoyCommandMapper:
    def __init__(
        self,
        cmd_low,
        cmd_high,
        height_range=(0.09, 0.17),
        height_rate=0.02,
        deadzone=0.15,
        initial_height=0.13,
    ):
        self.cmd_low = np.asarray(cmd_low, np.float64)
        self.cmd_high = np.asarray(cmd_high, np.float64)
        self.height_range = (float(height_range[0]), float(height_range[1]))
        self.height_rate = float(height_rate)
        self.deadzone = float(deadzone)
        self.height = float(
            np.clip(initial_height, self.height_range[0], self.height_range[1])
        )

    def _shaped(self, a):
        """Deadzone with rescale: continuous, hits +-1 at full deflection."""
        a = float(np.clip(a, -1.0, 1.0))
        if abs(a) < self.deadzone:
            return 0.0
        return np.sign(a) * (abs(a) - self.deadzone) / (1.0 - self.deadzone)

    def velocity(self, ax_vx, ax_vy, ax_wz):
        """(vx, vy, wz) command from three stick deflections in [-1, 1]."""
        cmd = np.zeros(3)
        for i, a in enumerate((ax_vx, ax_vy, ax_wz)):
            a = self._shaped(a)
            cmd[i] = a * (self.cmd_high[i] if a >= 0.0 else -self.cmd_low[i])
        return cmd

    def step_height(self, ax_h, dt):
        """Integrate the height-rate axis over dt seconds; returns height (m)."""
        self.height = float(
            np.clip(
                self.height + self._shaped(ax_h) * self.height_rate * dt,
                self.height_range[0],
                self.height_range[1],
            )
        )
        return self.height
