cmake_minimum_required(VERSION 3.8)
project(wojtek_agent_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioChunk.msg"
  "msg/Transcript.msg"
  "msg/RoutedIntent.msg"
  "msg/Sentence.msg"
  "msg/ExecStatus.msg"
  "msg/WorldMap.msg"
  "msg/WorldCmd.msg"
  "msg/WorldAck.msg"
  "srv/WorldCommand.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
