cmake_minimum_required(VERSION 3.8)
project(four_bar_bot_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch meshes urdf rviz DESTINATION share/${PROJECT_NAME})

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/setup_envs.sh.in")
ament_package()
