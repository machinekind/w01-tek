# Command request as a TOPIC, for split deployments: zenoh-bridge-ros2dds
# carries topics reliably across boxes but service replies did not survive
# the double-router hop (measured 2026-08-27). Correlate with WorldAck via
# req_seq. One-box stacks may still use the WorldCommand service.
std_msgs/Header header
uint64 req_seq   # caller-chosen, echoed in the ack
string kind      # "midlevel" | "goto" | "trick" | "reset"
string text
float64[] args
