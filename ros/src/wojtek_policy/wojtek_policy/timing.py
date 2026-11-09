"""This module computes sensor sample ages for /wojtek/policy_timing.

All timestamps must use the node clock.
Inputs are integer nanoseconds.
"""

NAN = float("nan")


def stamp_to_ns(stamp):
    """The result is the timestamp in nanoseconds.

    stamp must provide sec and nanosec fields.
    """
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def age_ms(now_ns, then_ns):
    """The result is now_ns minus then_ns in milliseconds.

    None and zero timestamps produce NaN.
    """
    if not then_ns:
        return NAN
    return (now_ns - then_ns) * 1e-6


def observation_ages(tick_ns, joints_stamp_ns, joints_arrival_ns,
                     imu_stamp_ns, uses_imu):
    """The result contains joint age, joint wait and IMU age in milliseconds.

    Each value measures elapsed time up to tick start.
    Joint age starts at the joint header stamp.
    Joint wait starts at callback entry.
    IMU age starts at the IMU header stamp.
    IMU age is NaN when the policy ignores the IMU.
    """
    imu_age = age_ms(tick_ns, imu_stamp_ns) if uses_imu else NAN
    return (
        age_ms(tick_ns, joints_stamp_ns),
        age_ms(tick_ns, joints_arrival_ns),
        imu_age,
    )


class InputSamples:
    """This class tracks timestamps of the latest accepted sensor samples.

    arrival_ns must be the time at joint callback entry.
    Rejected messages must leave the stored timestamps unchanged.
    """

    def __init__(self):
        self.joints_arrival_ns = None
        self.joints_stamp_ns = None
        self.imu_stamp_ns = None

    def note_joints(self, arrival_ns, header_stamp_ns):
        self.joints_arrival_ns = arrival_ns
        self.joints_stamp_ns = header_stamp_ns

    def note_imu(self, header_stamp_ns):
        self.imu_stamp_ns = header_stamp_ns

    def ages(self, tick_ns, uses_imu):
        """The result contains the stored samples' ages at tick_ns."""
        return observation_ages(
            tick_ns,
            self.joints_stamp_ns,
            self.joints_arrival_ns,
            self.imu_stamp_ns,
            uses_imu,
        )
