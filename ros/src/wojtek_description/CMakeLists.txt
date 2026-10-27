cmake_minimum_required(VERSION 3.8)
project(wojtek_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes urdf rviz DESTINATION share/${PROJECT_NAME})
# The generated MuJoCo model of the v6.27 legs. A simulation on those legs
# (robot:=wojtek_v2) loads its scene from here. The stock scene ships with
# wojtek_pc instead.
install(DIRECTORY mujoco/legs_v627 DESTINATION share/${PROJECT_NAME}/mujoco)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/setup_envs.sh.in")
ament_package()
