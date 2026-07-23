#!/usr/bin/env python3
"""Magnetometer ellipsoid calibration ("compass dance") for the Adafruit
5543 IMU, drone-style: rotate the robot through all orientations while the
stack is running, fit an ellipsoid to the raw field cloud, and write the
hard-iron offset + soft-iron matrix that imu_i2c_hardware_interface's
orientation ESKF applies before its yaw update.

    # live, robot stack running (magnetometer_broadcaster publishing):
    python3 mag_calib.py --duration 60 --out mag_calib.yaml

    # offline, from a recorded bag:
    python3 mag_calib.py --bag ~/wojtek_bags/run_XXX --out mag_calib.yaml

Rotate the robot SLOWLY through as many orientations as possible (all six
faces plus diagonals) during the capture -- coverage is reported per
octant, and a fit from a poorly covered cloud is rejected.  Calibrates the
STATIC, frame-fixed distortion only; current-dependent disturbance from
the drives is handled at runtime by the ESKF's adaptive R/gating.

Copy the resulting yaml over
ros/src/wojtek_bringup/config/mag_calib.yaml (identity placeholder) and
rebuild/deploy so the driver picks it up.
"""

import argparse
import sys

import numpy as np

TOPIC = "/magnetometer_broadcaster/magnetic_field"
T_TO_UT = 1e6  # sensor_msgs/MagneticField is tesla; the driver works in uT


def collect_live(duration_s: float) -> np.ndarray:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import MagneticField

    samples = []
    rclpy.init()
    node = Node("mag_calib_collector")
    node.create_subscription(
        MagneticField,
        TOPIC,
        lambda m: samples.append(
            (m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z)
        ),
        50,
    )
    node.get_logger().info(
        f"collecting {TOPIC} for {duration_s:.0f}s -- rotate the robot "
        "slowly through ALL orientations now"
    )
    end = node.get_clock().now().nanoseconds + int(duration_s * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    return np.asarray(samples) * T_TO_UT


def collect_bag(bag_path: str) -> np.ndarray:
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import MagneticField

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path), ConverterOptions("", ""))
    samples = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == TOPIC:
            m = deserialize_message(data, MagneticField)
            samples.append(
                (m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z)
            )
    return np.asarray(samples) * T_TO_UT


def octant_coverage(centered: np.ndarray) -> int:
    """Number of sign-octants (of 8) that contain at least one sample."""
    signs = centered > 0
    return len({tuple(row) for row in signs})


def fit_ellipsoid(pts: np.ndarray):
    """Least-squares ellipsoid fit.

    Returns (hard_iron (3,), soft_iron (3,3), field_norm) such that
    soft_iron @ (raw - hard_iron) lies on a sphere of radius field_norm.
    """
    x, y, z = pts.T
    # Quadric: a x^2 + b y^2 + c z^2 + 2f yz + 2g xz + 2h xy
    #          + 2p x + 2q y + 2r z + d = 0, solved as D v = 1 with the
    # constant folded into the normalization.
    design = np.column_stack(
        [x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z]
    )
    v, *_ = np.linalg.lstsq(design, np.ones(len(pts)), rcond=None)
    a, b, c, f, g, h, p, q, r = v
    m = np.array([[a, h, g], [h, b, f], [g, f, c]])
    center = -np.linalg.solve(m, np.array([p, q, r]))
    # Translate to the center: (x-c)^T Q (x-c) = 1.
    scale = center @ m @ center + 1.0
    quadric = m / scale

    eigval, eigvec = np.linalg.eigh(quadric)
    if np.any(eigval <= 0):
        raise ValueError(
            "fit is not an ellipsoid (bad coverage or disturbed data)"
        )
    radii = 1.0 / np.sqrt(eigval)
    field_norm = float(np.prod(radii) ** (1.0 / 3.0))  # geometric mean
    # sqrtm(Q) maps the ellipsoid onto the unit sphere; scale it back up to
    # the mean field magnitude so calibrated values stay in uT.
    soft_iron = field_norm * (eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T)
    return center, soft_iron, field_norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--duration", type=float, default=60.0,
                     help="live capture length [s]")
    src.add_argument("--bag", help="read a rosbag2 directory instead")
    ap.add_argument("--out", default="mag_calib.yaml")
    ap.add_argument("--min-octants", type=int, default=7,
                    help="reject the fit below this orientation coverage")
    args = ap.parse_args()

    pts = collect_bag(args.bag) if args.bag else collect_live(args.duration)
    if len(pts) < 200:
        print(f"FAIL: only {len(pts)} samples on {TOPIC}", file=sys.stderr)
        return 1

    hard_iron, soft_iron, field_norm = fit_ellipsoid(pts)

    centered = pts - hard_iron
    octants = octant_coverage(centered)
    calibrated = centered @ soft_iron.T
    residual = np.linalg.norm(calibrated, axis=1) - field_norm
    rms = float(np.sqrt(np.mean(residual**2)))

    print(f"samples          : {len(pts)}")
    print(f"octant coverage  : {octants}/8")
    print(f"hard iron [uT]   : {np.round(hard_iron, 2).tolist()}")
    print(f"field norm [uT]  : {field_norm:.2f}")
    print(f"fit residual RMS : {rms:.2f} uT ({100 * rms / field_norm:.1f}%)")
    if not 15.0 < field_norm < 100.0:
        print(f"FAIL: |B|={field_norm:.1f} uT is not a plausible Earth field",
              file=sys.stderr)
        return 1
    if octants < args.min_octants:
        print(f"FAIL: coverage {octants}/8 octants < {args.min_octants} -- "
              "rotate through more orientations", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(
            "# Magnetometer ellipsoid calibration consumed by\n"
            "# imu_i2c_hardware_interface (m = soft_iron * (raw - hard_iron),"
            " units uT).\n"
            f"# Generated by mag_calib.py: {len(pts)} samples, "
            f"{octants}/8 octants, residual RMS {rms:.2f} uT.\n"
        )
        fh.write(f"hard_iron_ut: {np.round(hard_iron, 4).tolist()}\n")
        fh.write("soft_iron: [")
        rows = [", ".join(f"{v:.6f}" for v in row) for row in soft_iron]
        fh.write((",\n" + " " * 12).join(rows))
        fh.write("]\n")
        fh.write(f"field_norm_ut: {field_norm:.4f}\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
