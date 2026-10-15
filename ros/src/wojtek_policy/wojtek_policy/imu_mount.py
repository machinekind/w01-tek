"""How Wojtek's IMU sits on the body, and the maths that takes it back out.

The IMU chip is bolted on upright but turned a quarter turn, so its +y
axis points forward and its +x axis points to the right. The driver
publishes what the chip measures in the chip's own axes, unchanged. So a
reading straight off /imu_sensor_broadcaster/imu is the chip's attitude and
the chip's turn rates, not the robot's: tilt the robot nose down and the
raw numbers call it a roll. Whatever shows or uses the robot's roll, pitch
and heading has to take the mount out first.

Only `math`, no numpy and no ROS: the launch files import this module, and
its tests run on machines that have neither. web/attitude.js in wojtek_deck
does the same sums for the deck page; the two must stay the same maths.
"""

import math

# The IMU's rotation in base_link, a copy of the rpy on body.urdf.xacro's
# imu_joint: upright, turned -90 deg about the vertical (chip +y forward,
# +x right). The driver publishes the chip's axes as they are, and the
# simulator fakes the same mount (wojtek_sim.urdf.xacro), so this number
# is used in the sim too, not only on the robot. Every node that reads the
# sensor's axes takes it from here. A copy that had drifted away from the
# URDF is what once fed the v41 policy an upside-down gravity.
# Floats on purpose: ROS reads a parameter default of floats as a list of
# doubles, and a list of ints would refuse a launch file's float override.
IMU_MOUNT_RPY = [0.0, 0.0, -1.5707963]


def quat_from_rpy(roll, pitch, yaw):
    """The quaternion (x, y, z, w) for a URDF roll, pitch and yaw.

    URDF turns about fixed axes: roll about x first, then pitch about y,
    then yaw about z, which is Rz(yaw) @ Ry(pitch) @ Rx(roll). That is the
    order policy_node.rpy_to_mat and leg_odometry's _rot_rpy use too.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_mul(a, b):
    """a times b, both (x, y, z, w): turn by b first, then by a.

    The usual (Hamilton) product, the same one MuJoCo's mju_mulQuat does
    with w first.
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q):
    """The opposite turn of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def rotate(q, v):
    """The 3-vector v turned by the unit quaternion q: q * (v, 0) * conj(q)."""
    x, y, z, _ = quat_mul(quat_mul(q, (v[0], v[1], v[2], 0.0)), quat_conj(q))
    return (x, y, z)


def rpy_from_quat(q):
    """(roll, pitch, yaw) in radians from a quaternion (x, y, z, w).

    The same sums the operator displays always used. The pitch is clamped
    before the arcsine so rounding cannot push it out of range, and the
    all-zero quaternion reads as level, (0, 0, 0).
    """
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


def into_base(orientation, gyro, mount_rpy):
    """An IMU reading turned into the robot's own frame, base_link.

    The sensor reports how its own frame is turned in the world, and how
    fast it turns about its own axes. The mount says how the sensor sits in
    base_link. Take the mount back out and the two numbers are the robot's.

    orientation  (x, y, z, w) as the IMU topic carries it. The driver's
                 all-zero "no fusion yet" quaternion stays all zero.
    gyro         (x, y, z) in rad/s, about the sensor's axes.
    mount_rpy    the sensor's roll, pitch, yaw in base_link (IMU_MOUNT_RPY).

    Returns (orientation, gyro) for base_link, in the same shapes.
    """
    mount = quat_from_rpy(*mount_rpy)
    # The sensor's attitude is the robot's attitude with the mount added on
    # top, so the robot's is the sensor's with the mount taken off again.
    body = quat_mul(orientation, quat_conj(mount))
    # A turn rate about the sensor's axes, said in the robot's axes. Same
    # direction as policy_node's `imu_mount @ gyro`.
    return body, rotate(mount, gyro)


def attitude_in_base(orientation, gyro, mount_rpy):
    """The robot's ((roll, pitch, yaw), (gx, gy, gz)) from a raw IMU reading."""
    body, gyro_base = into_base(orientation, gyro, mount_rpy)
    return rpy_from_quat(body), gyro_base
