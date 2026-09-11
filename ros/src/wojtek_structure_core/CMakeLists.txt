cmake_minimum_required(VERSION 3.16)
project(wojtek_structure_core)

# The Structure SDK is closed source (XRPro LLC) and is NOT vendored here:
# this repository is public and the SDK is not ours to redistribute. Its
# location enters through an environment variable declared right here, the
# same pattern UBUNTU_PRO_TOKEN and WOJTEK_AP_PSK use in ros/deploy.sh:
#
#   export STRUCTURE_SDK_DIR=/opt/StructureSDK-CrossPlatform   # or -D on the
#                                                             # cmake line
#
# WITHOUT IT THIS PACKAGE STILL BUILDS. The conversion layer and its tests
# are SDK-free and always built; only the driver node is skipped. That is
# deliberate: ros/deploy.sh builds the robot workspace with
# `--packages-up-to wojtek_bringup`, and a package that hard-failed on a
# missing SDK would take the robot's build down with it.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# ---------------------------------------------------------------- convert --
# No SDK, no ROS, no threads. Always built, always tested.
add_library(frame_convert src/frame_convert.cpp)
target_include_directories(frame_convert PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS frame_convert
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# -------------------------------------------------------------------- SDK --
set(STRUCTURE_SDK_DIR "$ENV{STRUCTURE_SDK_DIR}" CACHE PATH
    "Root of the Structure SDK (contains Libraries/Structure/...)")

set(SDK_FOUND FALSE)
if(STRUCTURE_SDK_DIR)
  # Both the archive layout and a flattened install are accepted; the SDK has
  # moved these between revisions.
  find_path(ST_INCLUDE_DIR ST/CaptureSession.h
    HINTS "${STRUCTURE_SDK_DIR}"
    PATHS "${STRUCTURE_SDK_DIR}/Libraries/Structure/Headers"
          "${STRUCTURE_SDK_DIR}/include"
          "${STRUCTURE_SDK_DIR}/Headers"
    NO_DEFAULT_PATH)

  # NOTE: the official Linux distribution of the SDK ships x86_64 objects.
  # If this find_library fails on the Pi with the headers present, that is
  # the arm64 question, not a CMake problem -- see README "Will this run on
  # the Pi at all".
  find_library(ST_LIBRARY NAMES Structure
    HINTS "${STRUCTURE_SDK_DIR}"
    PATHS "${STRUCTURE_SDK_DIR}/Libraries/Structure/Linux/${CMAKE_SYSTEM_PROCESSOR}"
          "${STRUCTURE_SDK_DIR}/Libraries/Structure/Linux/x86_64"
          "${STRUCTURE_SDK_DIR}/lib"
    NO_DEFAULT_PATH)

  if(ST_INCLUDE_DIR AND ST_LIBRARY)
    set(SDK_FOUND TRUE)
  endif()
endif()

if(SDK_FOUND)
  message(STATUS "Structure SDK: ${ST_LIBRARY}")
  add_executable(structure_core_node src/structure_core_node.cpp)
  target_include_directories(structure_core_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${ST_INCLUDE_DIR})
  target_link_libraries(structure_core_node frame_convert ${ST_LIBRARY})
  ament_target_dependencies(structure_core_node
    rclcpp sensor_msgs std_msgs geometry_msgs tf2_ros)
  install(TARGETS structure_core_node DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING
    "Structure SDK not found -- building the conversion layer and its tests "
    "only, skipping structure_core_node. Set STRUCTURE_SDK_DIR to build the "
    "driver. This is not an error: the robot and CI build this package "
    "without the SDK on purpose.")
endif()

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# ------------------------------------------------------------------ tests --
if(BUILD_TESTING)
  # The same binary a developer runs by hand with g++. No SDK, no ROS, no
  # camera, no gtest.
  add_executable(test_frame_convert test/host/test_frame_convert.cpp)
  target_link_libraries(test_frame_convert frame_convert)
  add_test(NAME frame_convert COMMAND test_frame_convert)
endif()

ament_package()
