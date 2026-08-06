cmake_minimum_required(VERSION 3.8)
project(wojtek_mujoco_hardware_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

# MuJoCo ships as a Python wheel here (Dockerfile installs it with pip), and
# that wheel carries both the C library and its headers -- so ask Python where
# it is instead of asking for a second, separately-installed copy that could
# drift from the one training/ uses. -DMUJOCO_DIR=... overrides.
if(NOT MUJOCO_DIR)
  execute_process(
    COMMAND python3 -c
            "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)"
    OUTPUT_VARIABLE MUJOCO_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
# The wheel ships libmujoco.so.<version> with no unversioned symlink, so
# find_library(NAMES mujoco) does not see it -- take the versioned file.
file(GLOB MUJOCO_CANDIDATES "${MUJOCO_DIR}/libmujoco.so*")
if(MUJOCO_CANDIDATES)
  list(GET MUJOCO_CANDIDATES 0 MUJOCO_LIBRARY)
endif()
if(NOT MUJOCO_DIR OR NOT MUJOCO_LIBRARY OR
   NOT EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR
          "MuJoCo not found. This package needs libmujoco and its headers; the "
          "usual source is the pip wheel (pip install mujoco), which ships "
          "both. Looked in MUJOCO_DIR='${MUJOCO_DIR}'. Point -DMUJOCO_DIR at a "
          "directory containing libmujoco.so* and include/mujoco/.")
endif()
message(STATUS "MuJoCo: ${MUJOCO_LIBRARY}")

add_library(${PROJECT_NAME} SHARED
            src/mujoco_hardware_interface.cpp
            src/mujoco_plant.cpp
            src/joint_map.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include
                                                  "${MUJOCO_DIR}/include")
target_link_libraries(${PROJECT_NAME} "${MUJOCO_LIBRARY}" yaml-cpp::yaml-cpp)
# The wheel's library lives outside the linker's search path, so record where
# it was found; without this the plugin loads only with LD_LIBRARY_PATH set.
set_target_properties(${PROJECT_NAME} PROPERTIES
                      INSTALL_RPATH "${MUJOCO_DIR}"
                      BUILD_WITH_INSTALL_RPATH TRUE)
ament_target_dependencies(${PROJECT_NAME} geometry_msgs hardware_interface rclcpp
                          rclcpp_lifecycle std_msgs tf2_ros)
# prevent pluginlib from using boost
target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(hardware_interface
                                         wojtek_mujoco_hardware_interface.xml)

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_joint_map test/test_joint_map.cpp src/joint_map.cpp)
  target_include_directories(test_joint_map PRIVATE include)
  target_link_libraries(test_joint_map yaml-cpp::yaml-cpp)
  # The map under test is wojtek_policy's own file, not a fixture copy.
  find_package(ament_index_cpp REQUIRED)
  ament_target_dependencies(test_joint_map ament_index_cpp)

  ament_add_gtest(test_mujoco_plant test/test_mujoco_plant.cpp src/mujoco_plant.cpp)
  target_include_directories(test_mujoco_plant PRIVATE include
                                                      "${MUJOCO_DIR}/include")
  target_link_libraries(test_mujoco_plant "${MUJOCO_LIBRARY}")
  set_target_properties(test_mujoco_plant PROPERTIES
                        BUILD_RPATH "${MUJOCO_DIR}")
endif()

ament_package()
