cmake_minimum_required(VERSION 3.8)
project(md80_hardware_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)

set(CMAKE_INSTALL_RPATH $ORIGIN)

set(CANDLE_BUILD_STATIC ON)
set(CANDLE_OMIT_EXAMPLES ON)
add_subdirectory(3rd_party/candle)

add_library(md80_hardware_interface SHARED src/md80_hardware_interface.cpp)

target_link_libraries(md80_hardware_interface candle)

target_include_directories(md80_hardware_interface PUBLIC include)
ament_target_dependencies(md80_hardware_interface hardware_interface rclcpp
                          rclcpp_lifecycle std_msgs)

target_compile_definitions(md80_hardware_interface
                           PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(hardware_interface
                                         md80_hardware_interface.xml)

install(
  TARGETS md80_hardware_interface
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(md80_hardware_interface)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle
                          std_msgs)

ament_package()
