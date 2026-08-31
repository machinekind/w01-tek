# How long one control tick of policy_node took.
#
# inference_ms covers the policy step alone. period_ms is the gap since the
# previous tick. Compare it to the control period to see whether the timer
# kept up.

std_msgs/Header header

float32 inference_ms
float32 period_ms
