"""A pixel in the camera picture becomes a setpoint for goto: the maths.

The VLM answers a picture with WHERE (a pixel on the thing to walk to) and
nothing else. Everything metric happens here, on the robot, from the depth
image that was taken with that picture:

    pixel (colour image) --viewing ray--> pixel (depth image) --median--> z
    --pinhole--> point in the depth optical frame --tf2 @ picture stamp-->
    point in odom --standoff--> the setpoint goto drives to

Pure logic, no ROS: the node (pixel_goal_node.py) feeds it images,
intrinsics and stamps; the tests feed it numbers. Measured in the
simulator (2026-09-25): 2 mm between this chain and the truth, in the
robot's own frame -- the picture's depth is the whole answer, as long as
it is the depth of THAT picture.
"""

import math

import numpy as np

# Below this share of usable pixels in the patch the depth is not trusted:
# a hole-ridden edge, a black or shiny surface, the sky.
MIN_VALID_FRACTION = 0.3


def depth_at(depth, u, v, radius=4, z_min=0.2, z_max=3.0):
    """Median depth in metres of the (2*radius+1)^2 patch around pixel (u, v)
    of a depth image (uint16 millimetres as RealSense publishes it, or float
    metres), and the share of the patch that was usable. Zero is a hole
    (no return); anything outside (z_min, z_max) counts as one too. Returns
    (None, share) when fewer than MIN_VALID_FRACTION of the patch is usable.
    The patch is clipped at the image edge, not rejected."""
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) * 1e-3
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(0, vi - radius):min(h, vi + radius + 1),
                  max(0, ui - radius):min(w, ui + radius + 1)]
    if patch.size == 0:
        return None, 0.0
    valid = patch[(patch > z_min) & (patch < z_max)]
    share = float(valid.size) / float(patch.size)
    if share < MIN_VALID_FRACTION:
        return None, share
    return float(np.median(valid)), share


def colour_to_depth_pixel(u_c, v_c, k_colour, k_depth):
    """The depth-image pixel on the same viewing ray as colour pixel (u_c, v_c).

    Colour and depth have different intrinsics (the sim renders them from
    one camera at 2:1; the D435 has two imagers). Going through the ray
    (colour K) and back (depth K) is exact when the two optical centres
    coincide, which holds for the sim and for depth aligned to colour on the
    real camera. Raw D435 depth sits ~15 mm beside the colour imager: a
    parallax of fx_d * 0.015 / z pixels, under 2 px beyond 1.5 m. k_* are the
    9-element row-major CameraInfo.k."""
    tx = (u_c - k_colour[2]) / k_colour[0]
    ty = (v_c - k_colour[5]) / k_colour[4]
    return k_depth[2] + k_depth[0] * tx, k_depth[5] + k_depth[4] * ty


def deproject(u, v, z, k):
    """Pixel plus depth to a point in the camera's optical frame (x right,
    y down, z forward), pinhole, no distortion (the D435 depth stream and
    the sim both publish zero coefficients)."""
    x = (u - k[2]) / k[0] * z
    y = (v - k[5]) / k[4] * z
    return x, y, z


def standoff(target, robot, distance):
    """The setpoint in front of an object: `distance` back from `target`
    along the line from `robot`, facing the object. Closer than that
    already: stay and just face it. Planar (x, y); returns (x, y, yaw)."""
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    dist = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    if dist <= distance or dist < 1e-6:
        return robot[0], robot[1], yaw
    k = (dist - distance) / dist
    return robot[0] + dx * k, robot[1] + dy * k, yaw


def nearest_frame(frames, stamp, max_skew):
    """From [(stamp_s, frame), ...] the frame closest to `stamp`, or None
    when none is within max_skew seconds. The picture the VLM saw and the
    depth image must be the same moment: in the sim the two streams run on
    separate timers (up to ~67 ms apart), on the real camera they are
    synchronised."""
    best, best_dt = None, None
    for t, frame in frames:
        dt = abs(t - stamp)
        if best_dt is None or dt < best_dt:
            best, best_dt = frame, dt
    if best is None or best_dt > max_skew:
        return None
    return best


class GoalTracker:
    """Keeps a setpoint alive at goto and decides when the job is over.

    goto expires a setpoint 3 s after it arrived, so the resolved goal is
    re-sent every `repeat_s` (in odom, resolved once: the picture's stamp
    would fall out of goto's 10 s TF buffer on a longer approach). goto's
    `blocked` is transient by design (it may still turn towards the goal
    and go on), so it is final only after `blocked_hold_s` of it. `reached`
    counts only after goto has visibly worked on THIS goal: its latched
    status may still carry the previous goal's word when the new one goes
    out. Nothing at all within `max_s` is a timeout.
    """

    ACTIVE = ("driving", "turning", "blocked")

    def __init__(self, repeat_s=1.0, blocked_hold_s=5.0, max_s=60.0):
        self.repeat_s = float(repeat_s)
        self.blocked_hold_s = float(blocked_hold_s)
        self.max_s = float(max_s)
        self.done = False
        self._t0 = None
        self._last_send = None
        self._blocked_since = None
        self._seen_active = False

    def start(self, now):
        self.done = False
        self._t0 = float(now)
        self._last_send = None
        self._blocked_since = None
        self._seen_active = False

    def step(self, status, now):
        """-> "send" (re-publish the setpoint now), "reached", "blocked",
        "timeout", or None (nothing to do this tick)."""
        if self.done or self._t0 is None:
            return None
        now = float(now)
        if now - self._t0 > self.max_s:
            self.done = True
            return "timeout"
        if status in self.ACTIVE:
            self._seen_active = True
        if status == "reached" and self._seen_active:
            self.done = True
            return "reached"
        if status == "blocked":
            if self._blocked_since is None:
                self._blocked_since = now
            elif now - self._blocked_since > self.blocked_hold_s:
                self.done = True
                return "blocked"
        else:
            self._blocked_since = None
        if self._last_send is None or now - self._last_send >= self.repeat_s:
            self._last_send = now
            return "send"
        return None
