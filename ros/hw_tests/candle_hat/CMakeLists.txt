cmake_minimum_required(VERSION 3.16)
project(candle_hw_tests CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reuse the candle checkout the ws already has (deploy.sh clones it there on
# the RPi). Override for an out-of-ws build:
#   cmake -B build -DCANDLE_DIR=/path/to/candle
set(CANDLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/md80_hardware_interface/3rd_party/candle"
    CACHE PATH "Path to the mabrobotics/candle checkout")
if(NOT EXISTS "${CANDLE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "candle not found at ${CANDLE_DIR} -- pass -DCANDLE_DIR=... (deploy.sh clones it into the ws on the RPi)")
endif()
add_subdirectory("${CANDLE_DIR}" candle_build)

add_executable(candle_bus_test candle_bus_test.cpp)
target_link_libraries(candle_bus_test candle)
