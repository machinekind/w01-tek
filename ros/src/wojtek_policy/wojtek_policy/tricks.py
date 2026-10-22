"""Scripted trick clips: PD-target timelines for show moves, per set of legs.

A trick is a short clip in the policy/MuJoCo actuator convention
(poses.ACTUATOR_NAMES order). Every clip starts AND ends at the robot's
home standing pose, so clips chain safely and the player can require a
near-home start. real_io_node plays them through the same machinery as the
stand_up/lie_down ramps -- DISARMED only, so a trick and the policy can
never fight over the command topic.

`for_robot(training_robot)` returns the clip set for one pair of legs:

  "wojtek"     keyframe clips authored by hand (TRICKS below): a list of
               (time, 12-target) keyframes interpolated with cosine easing,
               plus optional sinusoidal overlay layers (paw shake, body
               shake). Designed and validated in MuJoCo (2026-08-24): peak
               |actuator_force| <= 8.4 of the 9 N*m envelope, knees within
               [0.45, 3.15] (singularity guard 3.2), no falls, every clip
               settles back to the home stand.
  "legs_v627"  sampled clips (data/tricks_legs_v627.json): the stock clips
               carried over to the v6.27 four-bar by the mechanical team
               (foot-path IK, 2026-10-19). bow, sit and shake are variant
               M3, the abduction scaled so the foot moves sideways as far
               as on the stock robot (these legs are twice as long).
               paw_wave is variant M2, the stock 0.35 rad lean with slower
               weight-shift transitions: the M3 lean tips the robot over at
               a 0.035 rad zeroing error, M2 keeps the roll under 10
               degrees for every error tried, at a higher load on the hip
               motor's rotor flange (54/16 N*m against 16/13 for M3 -- the
               stock robot stripped those screws at 26/11). Validated in
               MuJoCo on legs_v627 (kp 60, kd 2, 22 N*m): no fall, no
               self-collision, peak knee torque 19 N*m.

Both kinds expose the same three calls: names, duration(name), sample(name, t).
"""

import json
from importlib import resources

import numpy as np

from wojtek_policy import poses
from wojtek_policy.poses import HOME_CTRL

KNEE_LO, KNEE_HI = 0.45, 3.15  # stock playback clamp, matches the sim validator


def _pose(rl=None, rr=None, fr=None, fl=None):
    """12-vector from per-leg (abd, hip, knee) tuples; None = home leg."""
    c = HOME_CTRL.copy()
    for i, leg in enumerate([rl, rr, fr, fl]):
        if leg is not None:
            c[3 * i : 3 * i + 3] = leg
    return c


_SIT = _pose(rl=(0, -1.1, 1.8), rr=(0, -1.1, 1.8))
# Weight shift right: unloads the front-left paw to ~0 N (measured in sim)
# so it can lift instead of being leaned on.
_SIT_LEAN = _pose(
    rl=(0.35, -1.1, 1.8), rr=(0.35, -1.1, 1.8), fr=(0.35, -0.2, 3.1)
)
# Paw offered forward: 0.35 m ahead of the base, 0.22 m up.
_PAW_UP = _pose(
    rl=(0.35, -1.1, 1.8), rr=(0.35, -1.1, 1.8),
    fr=(0.35, -0.2, 3.1), fl=(0, -1.8, 2.0),
)
_BOW = _pose(fr=(0, -1.0, 1.2), fl=(0, -1.0, 1.2))

# Stock legs: name -> (duration_s, keyframes [(t, pose)], osc layers
#          [(channel_idxs, amp_rad, hz, t0, t1, phases)])
TRICKS = {
    "bow": (
        4.5,
        [(0.5, HOME_CTRL), (1.5, _BOW), (3.2, _BOW), (4.2, HOME_CTRL)],
        [],
    ),
    "sit": (
        5.0,
        [(0.5, HOME_CTRL), (1.7, _SIT), (3.6, _SIT), (4.8, HOME_CTRL)],
        [],
    ),
    "paw_wave": (
        10.0,
        [
            (0.5, HOME_CTRL), (1.7, _SIT), (2.2, _SIT), (3.6, _SIT_LEAN),
            (4.4, _PAW_UP), (7.0, _PAW_UP), (7.8, _SIT_LEAN),
            (8.6, _SIT), (9.8, HOME_CTRL),
        ],
        [([10], 0.25, 2.0, 4.7, 6.8, [0.0])],  # front-left hip shake
    ),
    "shake": (
        4.5,
        [(0.5, HOME_CTRL), (4.0, HOME_CTRL)],
        [
            ([1, 4, 7, 10], 0.12, 5.0, 0.8, 3.2, [0.0, 0.0, 0.0, 0.0]),
            ([0, 3, 6, 9], 0.15, 5.0, 0.8, 3.2, [0.0, np.pi, 0.0, np.pi]),
        ],
    ),
}


class KeyframeTricks:
    """The stock legs' hand-authored clips."""

    def __init__(self, tricks=TRICKS):
        self._tricks = tricks

    @property
    def names(self):
        return tuple(self._tricks)

    def duration(self, name):
        return self._tricks[name][0]

    def sample(self, name, t):
        """Clip targets at time t (clamped to the clip), policy convention."""
        T, keys, layers = self._tricks[name]
        t = min(max(t, 0.0), T)
        c = keys[-1][1].copy()
        if t <= keys[0][0]:
            c = keys[0][1].copy()
        else:
            for (t0, c0), (t1, c1) in zip(keys, keys[1:]):
                if t <= t1:
                    s = 0.5 - 0.5 * np.cos(np.pi * (t - t0) / (t1 - t0))
                    c = c0 + s * (c1 - c0)
                    break
        for idxs, amp, hz, t0, t1, phases in layers:
            if t0 <= t <= t1:
                env = min(1.0, (t - t0) / 0.3, (t1 - t) / 0.3)
                for i, ph in zip(idxs, phases):
                    c[i] += env * amp * np.sin(2 * np.pi * hz * (t - t0) + ph)
        c[2::3] = np.clip(c[2::3], KNEE_LO, KNEE_HI)
        return c


class SampledTricks:
    """Clips given as dense target rows at a fixed tick, from a data file.

    Each clip is shifted by (home - its first row) so it starts and ends
    exactly at the robot's home stance (the authored clips start at the
    CAD stance, a few hundredths of a radian away), and every target is
    clipped to the legs' ctrl envelope. Playback interpolates linearly
    between rows.
    """

    def __init__(self, data: dict, home_ctrl, ctrl_low, ctrl_high):
        self._dt = float(data["dt"])
        self._clips = {}
        home = np.asarray(home_ctrl, float)
        for name, clip in data["clips"].items():
            rows = np.asarray(clip["targets"], float)
            if rows.ndim != 2 or rows.shape[1] != home.size:
                raise ValueError(
                    f"trick {name!r}: targets must be (n, {home.size}), "
                    f"got {rows.shape}"
                )
            rows = rows + (home - rows[0])
            self._clips[name] = np.clip(rows, ctrl_low, ctrl_high)

    @property
    def names(self):
        return tuple(self._clips)

    def duration(self, name):
        return (len(self._clips[name]) - 1) * self._dt

    def sample(self, name, t):
        rows = self._clips[name]
        x = min(max(t, 0.0), self.duration(name)) / self._dt
        i = min(int(x), len(rows) - 2)
        f = x - i
        return (1.0 - f) * rows[i] + f * rows[i + 1]


def _load_data(name):
    with resources.files("wojtek_policy").joinpath("data", name).open() as fh:
        return json.load(fh)


def for_robot(training_robot: str):
    """The trick clips of one set of legs, by training variant name."""
    if training_robot == "wojtek":
        return KeyframeTricks()
    if training_robot == "legs_v627":
        p = poses.for_robot(training_robot)
        return SampledTricks(
            _load_data("tricks_legs_v627.json"), p.home_ctrl, p.ctrl_low,
            p.ctrl_high,
        )
    raise ValueError(
        f"no tricks for training robot {training_robot!r} "
        "(known: 'wojtek', 'legs_v627')"
    )


# Module-level calls kept for the stock legs' callers and tests.
_STOCK = KeyframeTricks()


def duration(name):
    return _STOCK.duration(name)


def sample(name, t):
    """Stock clip targets at time t (clamped to the clip), policy convention."""
    return _STOCK.sample(name, t)
