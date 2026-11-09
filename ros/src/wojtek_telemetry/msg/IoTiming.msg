# The adapter publishes one timing sample per received joint_targets message.
# Samples include disarmed and rejected targets.
# All timestamps use the node clock.

# The stamp records target handling after any command publication.
std_msgs/Header header

# The stamp is joint_targets.header.stamp. It matches policy_timing.header.stamp.
builtin_interfaces/Time target_stamp

# The age is header.stamp minus target_stamp in milliseconds.
float32 target_age_ms

# The delay is the latest joint_states age in milliseconds just after the relay publishes joint_states_abs. It is NaN before the first sample.
float32 state_relay_ms

# The flag records the adapter's arm state when the target arrived.
bool armed

# The flag records whether the adapter published a position command for this target.
bool command_sent
