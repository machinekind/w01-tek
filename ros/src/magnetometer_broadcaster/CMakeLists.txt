cmake_minimum_required(VERSION 3.8)
project(magnetometer_broadcaster)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(magnetometer_broadcaster SHARED src/magnetometer_broadcaster.cpp)
target_include_directories(magnetometer_broadcaster PUBLIC include)
ament_target_dependencies(
  magnetometer_broadcaster controller_interface hardware_interface rclcpp
  rclcpp_lifecycle realtime_tools sensor_msgs)
# prevent pluginlib from using boost
target_compile_definitions(magnetometer_broadcaster
                           PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(controller_interface
                                         magnetometer_broadcaster.xml)

install(
  TARGETS magnetometer_broadcaster
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(magnetometer_broadcaster)
ament_export_dependencies(controller_interface hardware_interface pluginlib
                          rclcpp rclcpp_lifecycle realtime_tools sensor_msgs)

ament_package()
