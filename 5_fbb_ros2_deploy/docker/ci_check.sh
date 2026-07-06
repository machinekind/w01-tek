#!/usr/bin/env bash
# End-to-end smoke test, runs inside the fbb-deploy container:
# launch sim + policy headless, command +0.4 m/s, verify the base moves.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

ros2 launch fbb_policy sim.launch.py rviz:=false &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null || true' EXIT
sleep 6

echo "--- nodes ---"
ros2 node list
echo "--- checking topics publish ---"
for topic in /joint_states /imu/data /fbb/joint_targets; do
  if ! timeout 10 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    echo "FAIL: no messages on $topic"
    exit 1
  fi
  echo "$topic OK"
done

echo "--- walking test: cmd_vel x=0.4 for 8 s ---"
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.4}}' &
PUB_PID=$!
sleep 8
kill $PUB_PID

python3 - <<'EOF'
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import time

rclpy.init()
node = Node("ci_probe")
buf = Buffer()
TransformListener(buf, node)
deadline = time.time() + 5
x = None
while time.time() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
    try:
        t = buf.lookup_transform("odom", "base_link", rclpy.time.Time())
        x, z = t.transform.translation.x, t.transform.translation.z
    except Exception:
        continue
assert x is not None, "no odom->base_link TF"
print(f"base at x={x:+.2f} m, z={z:.3f} m after 8 s of cmd_vel 0.4")
assert x > 1.5, f"robot did not walk (x={x:.2f})"
assert z > 0.05, f"robot fell (z={z:.3f})"
print("E2E CHECK PASSED")
EOF
