cmake_minimum_required(VERSION 3.8)
project(imu_i2c_hardware_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(imu_i2c_hardware_interface SHARED
            src/imu_i2c_hardware_interface.cpp src/imu_i2c.cpp
            src/orientation_eskf.cpp)
target_include_directories(imu_i2c_hardware_interface PUBLIC include)
target_link_libraries(imu_i2c_hardware_interface Eigen3::Eigen yaml-cpp::yaml-cpp)
ament_target_dependencies(
  imu_i2c_hardware_interface hardware_interface rclcpp rclcpp_lifecycle)
# prevent pluginlib from using boost
target_compile_definitions(imu_i2c_hardware_interface
                           PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(hardware_interface
                                         imu_i2c_hardware_interface.xml)

install(
  TARGETS imu_i2c_hardware_interface
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(imu_i2c_hardware_interface)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_orientation_eskf test/test_orientation_eskf.cpp
                  src/orientation_eskf.cpp)
  target_include_directories(test_orientation_eskf PRIVATE include)
  target_link_libraries(test_orientation_eskf Eigen3::Eigen)
endif()

ament_package()
