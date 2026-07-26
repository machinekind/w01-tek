"""Polyline building blocks for course waypoints.

Pure numpy, unit-testable, and deliberately public: new courses are composed
from these. Everything returns (K, 2) waypoint arrays in the course's local
frame (origin at the robot's start, +x along its initial heading).
"""

import math

import numpy as np

LEAD_IN_M = 1.0  # straight run-up prepended by lead_in(), so a curve never
# starts on the first step of walking


def line(length, start=(0.0, 0.0), heading=0.0):
    """Two-point segment of `length` from `start` along `heading` (rad)."""
    x0, y0 = start
    return np.array(
        [[x0, y0], [x0 + length * math.cos(heading), y0 + length * math.sin(heading)]]
    )


def arc(radius, sweep, start=(0.0, 0.0), heading=0.0, n=64):
    """Constant-radius arc leaving `start` tangent to `heading`.

    `sweep` is signed in radians: positive turns left (CCW), negative right.
    The first returned point IS `start`, so callers concatenate with
    `join` and drop duplicates.
    """
    phi = np.linspace(0.0, abs(sweep), n)
    s = math.copysign(1.0, sweep)
    # Local frame: heading along +x, turn centre at (0, s*radius).
    lx = radius * np.sin(phi)
    ly = s * radius * (1.0 - np.cos(phi))
    c, sn = math.cos(heading), math.sin(heading)
    return np.stack([start[0] + c * lx - sn * ly, start[1] + sn * lx + c * ly], -1)


def join(*parts):
    """Concatenate polylines, dropping the duplicated shared endpoints."""
    out = [np.asarray(parts[0], dtype=float)]
    for p in parts[1:]:
        p = np.asarray(p, dtype=float)
        if np.allclose(out[-1][-1], p[0], atol=1e-9):
            p = p[1:]
        out.append(p)
    return np.concatenate(out, axis=0)


def sine_slalom(length, amplitude, wavelength, n=128):
    """y = amplitude * sin(2 pi x / wavelength) sampled over [0, length]."""
    x = np.linspace(0.0, length, n)
    return np.stack([x, amplitude * np.sin(2 * np.pi * x / wavelength)], -1)


def lead_in(shape):
    """Prepend LEAD_IN_M of straight run-up along +x to a shape at origin."""
    return join(line(LEAD_IN_M), np.asarray(shape, dtype=float) + np.array([LEAD_IN_M, 0.0]))
