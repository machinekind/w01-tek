# This message reports policy tick timing and sensor ages.
#
# inference_ms covers the policy step alone. period_ms is the gap since the
# previous tick. Compare it to the control period to see whether the timer
# kept up.
#
# The ages use the node clock at tick start.
# Header stamps must use the same clock.
# joints_age_ms - joints_wait_ms measures delivery time from the joint header
# stamp to callback entry.
# An age is NaN before the first sample or when its timestamp is zero.
# imu_age_ms is also NaN when the policy ignores the IMU.

std_msgs/Header header

float32 inference_ms
float32 period_ms
# joints_age_ms measures the time from the joint header stamp to tick start.
float32 joints_age_ms
# joints_wait_ms measures the time from joint callback entry to tick start.
float32 joints_wait_ms
# imu_age_ms measures the time from the IMU header stamp to tick start.
float32 imu_age_ms
