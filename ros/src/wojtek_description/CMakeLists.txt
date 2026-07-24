cmake_minimum_required(VERSION 3.8)
project(wojtek_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes urdf rviz DESTINATION share/${PROJECT_NAME})

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/setup_envs.sh.in")
ament_package()
