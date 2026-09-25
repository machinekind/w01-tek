"""Read wojtek ROS2 bags into numpy arrays for system identification.

Uses the pure-python `rosbags` package, so the training venv reads
`ros2 bag` recordings (sqlite3 or mcap) without a ROS install. All joint
signals are converted from URDF to MuJoCo convention with the same affine
joint map the ROS nodes use: q_mjc = sign * (q_urdf - offset).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from wojtek_rl import paths

TOPIC_TARGETS = "/wojtek/joint_targets"
TOPIC_STATES = "/joint_states"
TOPIC_IMU = "/imu/data"
TOPIC_ODOM = "/odom_vel"

_EXTRA_HINT = "sysid needs the 'sysid' extra: cd training && uv sync --extra sysid"


@dataclass
class BagSignals:
    """Time series from one bag, MuJoCo convention, actuator order.

    passive holds the non-actuated mapped joints (four-bar fourth/fifth)
    when the bag has them (the ROS MuJoCo sim publishes them, the real
    robot does not); each series is aligned with t_meas. The IMU and base
    velocity blocks are None when their topic is absent.
    """

    t_cmd: np.ndarray  # (Nc,) seconds from bag start
    cmd: np.ndarray  # (Nc, nu) commanded joint positions
    t_meas: np.ndarray  # (Nm,)
    qpos: np.ndarray  # (Nm, nu) measured positions, actuated joints
    qvel: np.ndarray  # (Nm, nu) measured velocities (zeros if not recorded)
    passive: dict  # joint name -> (Nm,) measured position
    t_imu: np.ndarray | None
    quat: np.ndarray | None  # (Ni, 4) wxyz, world<-base
    gyro: np.ndarray | None  # (Ni, 3) body frame
    t_base: np.ndarray | None
    linvel: np.ndarray | None  # (Nb, 3) world frame (sim ground truth only)


def load_joint_map(yaml_path=None):
    """joint name -> (sign, offset) with q_urdf = sign * q_mjc + offset."""
    yaml_path = yaml_path or paths.JOINT_MAP_YAML
    raw = yaml.safe_load(open(yaml_path))["joint_map"]
    return {n: (float(v["sign"]), float(v["offset"])) for n, v in raw.items()}


def _joint_series(msgs, names, with_velocity):
    """(t, pos, vel) arrays for `names` from JointState messages.

    Messages missing any requested joint are skipped (counted by the
    caller via the returned length). Velocity falls back to zeros when a
    message carries no velocity field.
    """
    ts, pos, vel = [], [], []
    index_cache = {}
    for t, msg in msgs:
        key = tuple(msg.name)
        idx = index_cache.get(key)
        if idx is None:
            lookup = {n: i for i, n in enumerate(msg.name)}
            idx = [lookup.get(n, -1) for n in names]
            index_cache[key] = idx
        if any(i < 0 for i in idx):
            continue
        ts.append(t)
        pos.append([msg.position[i] for i in idx])
        if with_velocity and len(msg.velocity) == len(msg.name):
            vel.append([msg.velocity[i] for i in idx])
        else:
            vel.append([0.0] * len(names))
    return np.array(ts), np.array(pos), np.array(vel)


def read_bag(
    bag_path,
    actuator_names,
    joint_map_yaml=None,
    targets_topic=TOPIC_TARGETS,
    states_topic=TOPIC_STATES,
    imu_topic=TOPIC_IMU,
    odom_topic=TOPIC_ODOM,
):
    """Read one ROS2 bag directory (or file) into BagSignals."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError(_EXTRA_HINT) from e

    jmap = load_joint_map(joint_map_yaml)
    missing = [n for n in actuator_names if n not in jmap]
    if missing:
        raise ValueError(f"joints missing from joint_map.yaml: {missing}")
    sign = np.array([jmap[n][0] for n in actuator_names])
    offset = np.array([jmap[n][1] for n in actuator_names])

    rows = {targets_topic: [], states_topic: [], imu_topic: [], odom_topic: []}
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in rows]
        for conn, t_ns, raw in reader.messages(connections=conns):
            rows[conn.topic].append((t_ns, reader.deserialize(raw, conn.msgtype)))
        topics_in_bag = sorted({c.topic for c in reader.connections})

    if not rows[targets_topic] or not rows[states_topic]:
        raise ValueError(
            f"bag {bag_path} must contain {targets_topic} and {states_topic}; "
            f"found topics: {topics_in_bag}"
        )
    for msgs in rows.values():
        msgs.sort(key=lambda tm: tm[0])

    t0 = min(msgs[0][0] for msgs in rows.values() if msgs)

    t_cmd, cmd_urdf, _ = _joint_series(
        rows[targets_topic], actuator_names, with_velocity=False
    )
    t_meas, qpos_urdf, qvel_urdf = _joint_series(
        rows[states_topic], actuator_names, with_velocity=True
    )
    if not len(t_cmd) or not len(t_meas):
        raise ValueError(
            f"bag {bag_path}: no message carries all actuated joints "
            f"{list(actuator_names)}"
        )

    # Passive mapped joints that every kept state message carries (the sim
    # bags have the four-bar fourth/fifth joints, real bags do not).
    first_names = set(rows[states_topic][0][1].name)
    passive_names = [
        n for n in jmap if n not in set(actuator_names) and n in first_names
    ]
    passive = {}
    if passive_names:
        t_p, p_urdf, _ = _joint_series(
            rows[states_topic], passive_names, with_velocity=False
        )
        if len(t_p) == len(t_meas):
            for k, n in enumerate(passive_names):
                s, o = jmap[n]
                passive[n] = s * (p_urdf[:, k] - o)

    t_imu = quat = gyro = None
    if rows[imu_topic]:
        t_imu = np.array([t for t, _ in rows[imu_topic]])
        quat = np.array(
            [
                (m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z)
                for _, m in rows[imu_topic]
            ]
        )
        gyro = np.array(
            [
                (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
                for _, m in rows[imu_topic]
            ]
        )
        t_imu = (t_imu - t0) * 1e-9

    t_base = linvel = None
    if rows[odom_topic]:
        t_base = np.array([t for t, _ in rows[odom_topic]])
        linvel = np.array(
            [(m.linear.x, m.linear.y, m.linear.z) for _, m in rows[odom_topic]]
        )
        t_base = (t_base - t0) * 1e-9

    return BagSignals(
        t_cmd=(t_cmd - t0) * 1e-9,
        cmd=sign * (cmd_urdf - offset),
        t_meas=(t_meas - t0) * 1e-9,
        qpos=sign * (qpos_urdf - offset),
        qvel=sign * qvel_urdf,
        passive=passive,
        t_imu=t_imu,
        quat=quat,
        gyro=gyro,
        t_base=t_base,
        linvel=linvel,
    )
