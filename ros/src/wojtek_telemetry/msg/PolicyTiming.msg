# How long one control tick of policy_node took.
#
# inference_ms covers the policy step alone. period_ms is the gap since the
# previous tick, which is the rate the timer actually achieved rather than
# the rate it was asked for.

std_msgs/Header header

float32 inference_ms
float32 period_ms
