#!/bin/bash
# Partition the wojtek stack across the isolated cores (Pi 3).
#
# isolcpus disables the scheduler's load balancing on the isolated cores, so
# every process the service tasksets to 2,3 lands on core 2 and stays there
# while core 3 idles. On a Pi 5 the A76 hid it; on the Pi 3's A53 it showed
# up as a 122/200 Hz controller loop, a 20 Hz policy tick (contract: 50) and
# constant "sensor data stale" holds. Measured layout that fixes it:
#   core 3: ros2_control_node alone (RT loop + CANdle SPI)
#   core 2: policy_node + real_io_node -- a coupled pair: real_io relays
#           wojtek/joint_states_abs that the policy's freshness check rides
#           on, so splitting them across cores re-creates the stale-holds
#   cores 0,1: pad/joy/state-publisher/launch, sharing with the OS
# Reapplied forever (nodes respawn; a one-shot ExecStartPost proved flaky).
while true; do
  cm=$(pgrep -f ros2_control_node | head -1)
  [ -n "$cm" ] && taskset -apc 3 "$cm" >/dev/null 2>&1
  for p in $(pgrep -f "policy_node|real_io_node"); do
    taskset -apc 2 "$p" >/dev/null 2>&1
  done
  for p in $(pgrep -f "gamepad_teleop|joy_node|robot_state_publisher|ros2 launch"); do
    taskset -apc 0,1 "$p" >/dev/null 2>&1
  done
  sleep 15
done
