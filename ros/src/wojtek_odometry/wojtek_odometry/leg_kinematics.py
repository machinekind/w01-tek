"""Foot forward kinematics for one leg, straight from the robot's URDF.

The chain constants are parsed out of the same robot_description that
robot_state_publisher renders, so they cannot drift from what RViz shows.
The four-bar closure is not re-derived either: the passive fourth/fifth
joints come from wojtek_policy.poses.PASSIVE_FROM_KNEE, the same
polynomials real_io_node publishes for RViz. This module adds only what
neither of those provides: the foot point and the Jacobian.

All angles are ABSOLUTE URDF convention -- exactly what
/wojtek/joint_states_abs carries. One leg has three measured DOFs
(first=hip roll, second=hip pitch, third=knee); the passive joints are a
function of the knee, so the Jacobian is 3x3 per leg and the knee column
picks up the four-bar coupling automatically by re-evaluating the
polynomials under perturbation.

The URDF has no foot link (it ends at sixth_link); the foot sphere centre
sits at FOOT_IN_SIXTH in sixth_link's frame, the constant the MJX model
uses for its foot_link body.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from wojtek_policy import poses

# foot_link body position inside sixth_link, from the MJX model
# (wojtek_mjx.xml: <body name="*_foot_link" pos="0.21 0 0.0115">).
FOOT_IN_SIXTH = np.array([0.21, 0.0, 0.0115])

# Foot sphere radius (wojtek_mjx.xml: <geom name="*_foot_sphere" size="0.046">).
FOOT_RADIUS = 0.046

LEGS = ("front_left", "front_right", "rear_left", "rear_right")


def _rpy_matrix(rpy):
    """URDF fixed-axis rpy -> rotation matrix (R = Rz @ Ry @ Rx)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_rotation(axis, angle):
    """Rodrigues rotation about a unit axis.

    The expanded closed form, one array construction. The textbook
    I + sK + (1-c)K^2 costs seven numpy dispatches on 3x3 matrices, and at
    ~600 calls/s on the Pi 4 that dispatch overhead (not the flops) was a
    visible slice of the odometry profile. Verified equal to 1e-15.
    """
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
        [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
        [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
    ])


def _horner(coeffs, x):
    """np.polyval semantics in plain Python.

    The passive polynomials are 3rd/4th degree; np.polyval spends ~10 us of
    numpy machinery per call to do four multiply-adds. This is the same
    Horner scheme without the dispatch.
    """
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


class _Joint:
    def __init__(self, name, jtype, xyz, rpy, axis):
        self.name = name
        self.type = jtype
        self.xyz = xyz
        self.rot = _rpy_matrix(rpy)
        self.axis = axis


def _parse_chain(urdf_xml, tip_link):
    """Joints from base_link down to tip_link, in base-to-tip order."""
    root = ET.fromstring(urdf_xml)
    by_child = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = np.fromstring(
            (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"), sep=" ")
        rpy = np.fromstring(
            (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"), sep=" ")
        axis_el = joint.find("axis")
        axis = np.fromstring(
            axis_el.get("xyz") if axis_el is not None else "1 0 0", sep=" ")
        by_child[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            _Joint(joint.get("name"), joint.get("type"), xyz, rpy, axis),
        )

    chain = []
    link = tip_link
    while link != "base_link":
        if link not in by_child:
            raise ValueError(f"no path from base_link to {tip_link} (stuck at {link})")
        link, joint = by_child[link]
        chain.append(joint)
    chain.reverse()
    return chain


class LegKinematics:
    """FK and Jacobian of one leg's foot point in the base_link frame."""

    def __init__(self, urdf_xml, leg):
        if leg not in LEGS:
            raise ValueError(f"unknown leg {leg!r}")
        self.leg = leg
        self.chain = _parse_chain(urdf_xml, f"{leg}_sixth_link")
        self.actuated = [f"{leg}_{j}_joint" for j in ("first", "second", "third")]
        self._passive = {
            name: coeffs
            for name, coeffs in poses.PASSIVE_FROM_KNEE.items()
            if name.startswith(leg)
        }
        if len(self._passive) != 2:
            raise ValueError(f"expected 2 passive joints for {leg}")
        # For the single-pass foot_state(): actuated joint -> Jacobian
        # column, and the passive polynomials' derivatives (np.polyder, so
        # still nothing hand-differentiated).
        self._actuated_col = {name: i for i, name in enumerate(self.actuated)}
        self._dpassive = {
            name: tuple(float(c) for c in np.polyder(coeffs))
            for name, coeffs in self._passive.items()
        }

    def _angles(self, q):
        """Map (q1, q2, q3) to every revolute joint in the chain."""
        angles = dict(zip(self.actuated, q))
        knee = float(q[2])
        for name, coeffs in self._passive.items():
            angles[name] = _horner(coeffs, knee)
        return angles

    def foot_position(self, q):
        """Foot sphere centre in base_link, for URDF-absolute (q1, q2, q3)."""
        angles = self._angles(q)
        rot = np.eye(3)
        pos = np.zeros(3)
        for joint in self.chain:
            pos = pos + rot @ joint.xyz
            rot = rot @ joint.rot
            if joint.type == "revolute":
                rot = rot @ _axis_rotation(joint.axis, angles[joint.name])
        return pos + rot @ FOOT_IN_SIXTH

    def shank_angular_velocity(self, q, qd):
        """Angular velocity of sixth_link (the shank) in base_link, from the
        joint rates alone -- the base's own rotation is not included.

        Needed for the rolling correction: the foot is a 46 mm sphere, so
        in stance the CONTACT POINT is stationary while the sphere centre
        translates at omega_shank x (R * up). Treating the centre as pinned
        under-reads the base speed by exactly that term (~15 % of the walk
        speed at gait pitch rates).
        """
        angles = self._angles(q)
        # d(passive)/dt via the chain rule through the knee polynomial.
        rates = dict(zip(self.actuated, qd))
        for name, coeffs in self._passive.items():
            rates[name] = float(np.polyval(np.polyder(coeffs), q[2])) * qd[2]
        rot = np.eye(3)
        omega = np.zeros(3)
        for joint in self.chain:
            rot = rot @ joint.rot
            if joint.type == "revolute":
                omega = omega + (rot @ joint.axis) * rates[joint.name]
                rot = rot @ _axis_rotation(joint.axis, angles[joint.name])
        return omega

    def jacobian(self, q, eps=1e-6):
        """d(foot)/d(q1,q2,q3), central differences; 3x3.

        Numeric on purpose: the knee column must include the passive
        four-bar coupling, and differentiating the polynomial chain by
        hand is exactly the kind of duplication that drifts.

        Kept as the reference implementation: foot_state() below computes
        the same matrix geometrically in one chain walk and is what the
        odometry node runs; a unit test holds the two equal, so the fast
        path cannot drift from this definition.
        """
        jac = np.empty((3, 3))
        q = np.asarray(q, dtype=float)
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = eps
            jac[:, i] = (
                self.foot_position(q + dq) - self.foot_position(q - dq)
            ) / (2 * eps)
        return jac

    def foot_state(self, q, qd):
        """(foot position, 3x3 Jacobian, shank angular velocity) in ONE
        chain walk.

        Numerically identical to foot_position() + jacobian() +
        shank_angular_velocity() (a unit test enforces it), ~8x cheaper:
        the numeric Jacobian alone walks the chain six times, which at the
        Pi 4's per-numpy-call overhead dominated the whole odometry node
        (~25 ms/sample, 63% of a core).

        Geometry: a revolute joint's contribution to the foot velocity is
        z x (p_foot - p_joint) per radian, with z and p_joint taken from
        the same forward pass that produces the foot position. Actuated
        joints map straight to their column; the passive four-bar joints
        fold into the knee column through d(poly)/dq3 -- np.polyder again,
        so the "no hand-derived polynomial" rule from jacobian() holds
        here too.
        """
        angles = self._angles(q)
        knee = float(q[2])
        dknee = {
            name: _horner(dcoeffs, knee)
            for name, dcoeffs in self._dpassive.items()
        }
        rot = np.eye(3)
        pos = np.zeros(3)
        revolute = []  # (world axis, world joint origin, name)
        for joint in self.chain:
            pos = pos + rot @ joint.xyz
            rot = rot @ joint.rot
            if joint.type == "revolute":
                revolute.append((rot @ joint.axis, pos, joint.name))
                rot = rot @ _axis_rotation(joint.axis, angles[joint.name])
        p_foot = pos + rot @ FOOT_IN_SIXTH

        jac = np.zeros((3, 3))
        omega = np.zeros(3)
        for z, p_j, name in revolute:
            col = np.cross(z, p_foot - p_j)
            if name in self._actuated_col:
                jac[:, self._actuated_col[name]] += col
                rate = qd[self._actuated_col[name]]
            else:
                jac[:, 2] += col * dknee[name]
                rate = dknee[name] * qd[2]
            omega = omega + z * rate
        return p_foot, jac, omega
