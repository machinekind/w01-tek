"""This module provides timestamp arithmetic for io_timing without rclpy.

A timestamp is an integer count of nanoseconds or an object with sec and nanosec.
"""


def stamp_ns(stamp):
    """The result is the timestamp in integer nanoseconds."""
    if isinstance(stamp, int):
        return stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def age_ms(now, stamp):
    """The result is now minus stamp in milliseconds.

    A later stamp produces a negative result.
    """
    return (stamp_ns(now) - stamp_ns(stamp)) / 1e6
