"""Scripted trick clips: keyframe PD-target timelines for show moves.

Each trick is a short clip in the policy/MuJoCo actuator convention
(poses.ACTUATOR_NAMES order, same as poses.HOME_CTRL): a list of
(time, 12-target) keyframes interpolated with cosine easing, plus
optional sinusoidal overlay layers (paw shake, body shake). Every clip
starts AND ends at the home standing pose, so clips chain safely and
the player can require a near-home start.

Designed and validated in MuJoCo (2026-08-24): peak |actuator_force|
<= 8.4 of the 9 N*m envelope, knees within [0.45, 3.15] (singularity
guard 3.2), no falls, every clip settles back to the home stand.
real_io_node plays them through the same machinery as the
stand_up/lie_down ramps -- DISARMED only, so a trick and the policy can
never fight over the command topic.
"""

import numpy as np

from wojtek_policy.poses import HOME_CTRL

KNEE_LO, KNEE_HI = 0.45, 3.15  # playback clamp, matches the sim validator


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

# Weight onto the other three legs, then the rear-left swings out and up
# like a dog at a tree ("siku pod drzewem", user request 2026-08-26).
# Designed by LOOKING at rendered hold frames (v1 passed every numeric
# gate and still sagged onto the lifted corner on camera). Hip +0.7 with
# the knee folded to 0.5 raises the foot ~12 cm at a 3.6 N*m hold torque;
# hip targets past ~0.85 sit unreachably far and pin the actuator at the
# 9 N*m clamp for the whole hold. The other three legs lean 0.35 so the
# body stays level instead of sagging onto the lifted corner.
# Three feet planted, one leg HIGH (user spec, v6). The lift rides the
# ABDUCTION axis with the leg folded flat (knee on the 0.45 stop): rotating
# a folded leg sideways-up costs 3.9 N*m where holding an extended leg out
# via the hip pinned the actuator at the 9 N*m clamp. Foot 16 cm clear --
# above the body line, unambiguous on camera; the body stands tall and
# leans naturally over the stance legs.
_PEE_LEAN = _pose(rl=(0.25, -0.2, 3.1), rr=(0.25, -0.2, 3.1),
                  fr=(0.25, -0.2, 3.1), fl=(0.25, -0.2, 3.1))
_PEE_UP = _pose(rl=(-1.1, 0.6, 0.45), rr=(0.25, -0.2, 3.1),
                fr=(0.25, -0.2, 3.1), fl=(0.25, -0.2, 3.1))
# Two-stage descent: swinging the folded leg straight back down brakes the
# abduction motor at 8.6 N*m; un-swinging first, then unfolding, keeps the
# whole return under the envelope.
_PEE_DOWN = _pose(rl=(-0.4, 0.2, 1.6), rr=(0.25, -0.2, 3.1),
                  fr=(0.25, -0.2, 3.1), fl=(0.25, -0.2, 3.1))

# name -> (duration_s, keyframes [(t, pose)], osc layers
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
    "pee": (
        12.0,
        [
            (0.5, HOME_CTRL), (2.0, _PEE_LEAN), (4.6, _PEE_UP),
            (7.4, _PEE_UP), (9.2, _PEE_DOWN), (10.6, _PEE_LEAN),
            (11.8, HOME_CTRL),
        ],
        [],
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


def duration(name):
    return TRICKS[name][0]


def sample(name, t):
    """Clip targets at time t (clamped to the clip), policy convention."""
    T, keys, layers = TRICKS[name]
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
