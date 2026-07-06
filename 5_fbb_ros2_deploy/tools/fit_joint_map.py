"""Fit the affine map between MuJoCo joint angles and URDF joint angles.

The MJCF (four_bar_bot_mjx.xml, what the policy was trained on) and the URDF
(four_bar_bot.urdf.xacro, what RViz/ros2_control use) describe the same robot
but with different joint-zero conventions (gear-correction offsets and
mirrored axes are baked into the URDF joint origins). Because every joint is
the same physical hinge, the relation is exactly affine per joint:

    q_urdf = sign * q_mujoco + offset

This script solves (sign, offset) for all 20 revolute joints by comparing
relative body orientations, validates the fit over random configurations
(including world-position agreement of every link), and writes
ros2_ws/src/fbb_policy/config/joint_map.yaml consumed by the sim and real
nodes.

Run: 4_four_bar_bot_rl/.venv/bin/python tools/fit_joint_map.py
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import xacro

REPO = Path(__file__).resolve().parents[2]
DESC = REPO / "quadruped_ros2_original/four_bar_bot_description"
MJCF = DESC / "mujoco/four_bar_bot_mjx.xml"
import os

# The real xacro uses $(find four_bar_bot_description); outside ROS pass a
# preprocessed copy via FBB_URDF_XACRO (see run.sh fit-map).
URDF_XACRO = Path(os.environ.get("FBB_URDF_XACRO", DESC / "urdf/four_bar_bot.urdf.xacro"))
OUT = Path(__file__).resolve().parents[1] / "ros2_ws/src/fbb_policy/config/joint_map.yaml"

LEGS = ("rear_left", "rear_right", "front_right", "front_left")
JOINTS = ("first", "second", "third", "fourth", "fifth")


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def axis_angle_mat(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def angle_about_axis(M, axis):
    """q such that Rot(axis, q) best equals M."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    s = 0.5 * a @ np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    c = 0.5 * (np.trace(M) - 1.0)
    return math.atan2(s, c)


class Urdf:
    """Minimal URDF kinematics: joints keyed by child link."""

    def __init__(self, xml_str):
        root = ET.fromstring(xml_str)
        self.by_child = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()] if origin is not None else [0, 0, 0]
            axis_el = j.find("axis")
            axis = [float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [1, 0, 0]
            self.by_child[j.find("child").get("link")] = {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "pos": np.array(xyz),
                "rot": rpy_to_mat(*rpy),
                "axis": np.array(axis),
            }

    def chain(self, ancestor, descendant):
        """Joints from ancestor link down to descendant link, top first."""
        out = []
        link = descendant
        while link != ancestor:
            j = self.by_child[link]
            out.append(j)
            link = j["parent"]
        return list(reversed(out))

    def fk(self, angles, base_pos, base_rot):
        """World pose of every link given {joint_name: angle}; base_link at base_pos/rot."""
        poses = {"base_link": (base_pos, base_rot)}
        # iterate until fixed point (tree order unknown)
        pending = dict(self.by_child)
        while pending:
            progressed = False
            for child, j in list(pending.items()):
                if j["parent"] not in poses:
                    continue
                pp, pr = poses[j["parent"]]
                rot = pr @ j["rot"]
                pos = pp + pr @ j["pos"]
                if j["type"] == "revolute":
                    rot = rot @ axis_angle_mat(j["axis"], angles[j["name"]])
                poses[child] = (pos, rot)
                del pending[child]
                progressed = True
            if not progressed:
                break
        return poses


def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    urdf_xml = xacro.process_file(
        str(URDF_XACRO), mappings={"use_hardware": "false", "use_sim": "false"}
    ).toxml()
    urdf = Urdf(urdf_xml)

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    home_qpos = model.key("home").qpos.copy()

    hinge_names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    qadr = {n: model.joint(n).qposadr[0] for n in hinge_names}

    def mj_fk(qpos):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        return {
            model.body(i).name: (data.xpos[i].copy(), quat_to_mat(data.xquat[i]))
            for i in range(model.nbody)
        }

    rng = np.random.default_rng(0)

    def urdf_angle(joint_urdf_name, leg, jname, poses):
        """URDF angle of a joint given MuJoCo world body poses."""
        child = f"{leg}_{'second third fourth fifth sixth'.split()[JOINTS.index(jname)]}_link"
        # MJCF parent body of the child
        parent_map = {
            "first": "base_link",  # root <-> base_link
            "second": f"{leg}_second_link",
            "third": f"{leg}_third_link",
            "fourth": f"{leg}_fourth_link",
            "fifth": f"{leg}_third_link",
        }
        parent = parent_map[jname]
        mj_parent = "root" if parent == "base_link" else parent
        pp, pr = poses[mj_parent]
        cp, cr = poses[child]
        # URDF fixed transforms + joint origin between parent link and child
        joints = urdf.chain(parent, child)
        R_pre = np.eye(3)
        for j in joints[:-1]:  # fixed joints
            assert j["type"] == "fixed", j["name"]
            R_pre = R_pre @ j["rot"]
        j = joints[-1]
        assert j["name"] == joint_urdf_name and j["type"] == "revolute"
        R_pre = R_pre @ j["rot"]
        M = R_pre.T @ pr.T @ cr
        return angle_about_axis(M, j["axis"])

    # -- fit sign/offset per joint ------------------------------------------
    result = {}
    for leg in LEGS:
        for jname in JOINTS:
            mj_joint = f"{leg}_{jname}_joint"
            adr = qadr[mj_joint]
            samples = []
            for dq in (-0.3, 0.0, 0.3, 0.55):
                qpos = home_qpos.copy()
                qpos[adr] += dq
                poses = mj_fk(qpos)
                q_mj = qpos[adr]
                q_u = urdf_angle(mj_joint, leg, jname, poses)
                samples.append((q_mj, q_u))
            (a, ua), (b, ub) = samples[0], samples[-1]
            sign = round((ub - ua) / (b - a))
            assert sign in (-1, 1), f"{mj_joint}: non-unit slope {(ub-ua)/(b-a):.4f}"
            # offset via wrapped mean over all samples
            offs = [math.remainder(u - sign * m, 2 * math.pi) for m, u in samples]
            spread = max(offs) - min(offs)
            assert spread < 1e-6, f"{mj_joint}: non-affine, offset spread {spread:.2e}"
            result[mj_joint] = {"sign": int(sign), "offset": float(np.mean(offs))}

    # -- validate: full-pose position agreement over random configs ---------
    worst = 0.0
    for _ in range(20):
        qpos = home_qpos.copy()
        for n in hinge_names:
            qpos[qadr[n]] += rng.uniform(-0.4, 0.4)
        poses = mj_fk(qpos)
        angles = {
            j: result[j]["sign"] * qpos[qadr[j]] + result[j]["offset"]
            for j in result
        }
        root_pos, root_rot = poses["root"]
        upose = urdf.fk(angles, root_pos, root_rot)
        for leg in LEGS:
            for link in ("second", "third", "fourth", "fifth", "sixth"):
                name = f"{leg}_{link}_link"
                err = float(np.linalg.norm(upose[name][0] - poses[name][0]))
                worst = max(worst, err)
    print(f"validated URDF vs MuJoCo link positions: worst error {worst*1000:.3f} mm")
    assert worst < 1e-4, "URDF/MJCF kinematics disagree; map is unsafe"

    lines = [
        "# Generated by tools/fit_joint_map.py -- do not edit by hand.",
        "# q_urdf = sign * q_mujoco + offset  (radians). MuJoCo == policy convention.",
        "joint_map:",
    ]
    for name, so in result.items():
        lines.append(f"  {name}: {{sign: {so['sign']}, offset: {so['offset']:.10f}}}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
