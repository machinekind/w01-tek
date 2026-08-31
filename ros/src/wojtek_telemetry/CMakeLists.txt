cmake_minimum_required(VERSION 3.8)
project(wojtek_telemetry)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/SysInfo.msg
  msg/PolicyTiming.msg
  DEPENDENCIES std_msgs)

# The node is a single file installed as a program, not as an importable
# Python package. A package directory called wojtek_telemetry would land in
# the same place as the one the message generator writes, and the two would
# fight over __init__.py.
install(PROGRAMS scripts/sysinfo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sysinfo_node)

ament_package()
