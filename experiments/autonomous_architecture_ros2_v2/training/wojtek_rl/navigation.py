"""Go-to-point navigation on top of the joystick (velocity-tracking) policy.

Ported verbatim from an earlier prototype's navigation module -- robot
agnostic, so it works unchanged for Wojtek's [vx, vy, wz] joystick
command.

The trained policy tracks a *body-frame* velocity command
``[vx, vy, vyaw]``. To make the robot walk to a clicked world point we close a
simple control loop: every step we look at where the target is relative to the
robot and emit the command that drives it there, then let the policy figure out
the legs. Pure NumPy / math so it's trivial to unit-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about world +z) from a wxyz quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class NavConfig:
    vx_max: float = 1.2      # forward speed cap (policy trained up to ~1.5)
    vy_max: float = 0.6      # strafe speed cap (policy trained up to ~0.8)
    yaw_max: float = 1.2     # turn-rate cap (rad/s)
    k_v: float = 1.5         # speed gain vs. distance
    k_yaw: float = 1.5       # turn gain vs. heading error
    stop_radius: float = 0.18  # within this distance, command "stand still"


def command_to_target(
    x: float, y: float, yaw: float, tx: float, ty: float, cfg: NavConfig
) -> tuple[float, float, float, float, bool]:
    """Return ``(vx, vy, vyaw, distance, reached)`` to walk toward (tx, ty).

    Holonomic: we steer in the body frame toward the target while also yawing to
    face it, so the robot mostly walks forward (and arrives facing the goal)
    rather than crab-walking the whole way.
    """
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist < cfg.stop_radius:
        return 0.0, 0.0, 0.0, dist, True

    # Rotate the world-frame target direction into the robot's body frame.
    c, s = math.cos(yaw), math.sin(yaw)
    bx = c * dx + s * dy       # forward component
    by = -s * dx + c * dy      # left component

    speed = min(cfg.k_v * dist, cfg.vx_max)
    vx = max(-cfg.vx_max, min(cfg.vx_max, speed * bx / dist))
    vy = max(-cfg.vy_max, min(cfg.vy_max, speed * by / dist))

    heading_err = math.atan2(by, bx)  # angle to target in body frame
    vyaw = max(-cfg.yaw_max, min(cfg.yaw_max, cfg.k_yaw * heading_err))
    return vx, vy, vyaw, dist, False
