// Copyright (c) 2024, Jakub Delicat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bmx160_serial_hardware_interface/bmx160_serial.hpp"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

namespace
{
// termios speaks in speed_t constants (B115200 == 4098), not in bit rates:
// cfsetospeed(&tty, 115200) returns -1/EINVAL and leaves the port's speed
// untouched. Ignoring that return left the port at its 9600 default while the
// firmware streamed at 115200 -- the reader saw only noise and every field
// stayed NaN. Map the URDF's plain integer onto the constant; 0 = unsupported.
speed_t to_speed_constant(int baud)
{
  switch (baud) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    case 460800:
      return B460800;
    case 921600:
      return B921600;
    default:
      return 0;
  }
}
}  // namespace

// Constructor: Opens the serial port
BMX160Serial::BMX160Serial(const std::string & port_name, int baud_rate)
: port_name_{port_name}, baud_rate_{baud_rate}
{
}

bool BMX160Serial::initialize()
{
  serial_fd = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (serial_fd < 0) {
    std::cerr << "Error opening serial port!" << std::endl;
    return false;
  }

  struct termios tty;
  if (tcgetattr(serial_fd, &tty) != 0) {
    std::cerr << "Error from tcgetattr" << std::endl;
    return false;
  }

  const speed_t speed = to_speed_constant(baud_rate_);
  if (speed == 0) {
    std::cerr << "Unsupported baud rate " << baud_rate_ << " for " << port_name_ << std::endl;
    return false;
  }
  if (cfsetospeed(&tty, speed) != 0 || cfsetispeed(&tty, speed) != 0) {
    std::cerr << "Error setting baud rate " << baud_rate_ << " on " << port_name_ << ": "
              << strerror(errno) << std::endl;
    return false;
  }

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_iflag &= ~IGNBRK;
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  tty.c_cc[VMIN] = 1;
  tty.c_cc[VTIME] = 1;

  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | PARODD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;

  if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
    std::cerr << "Error from tcsetattr" << std::endl;
    return false;
  }
  return true;
}

// Destructor: Closes the serial port
BMX160Serial::~BMX160Serial()
{
  if (serial_fd >= 0) {
    close(serial_fd);
  }
}

// Reads a line of input from the serial port
std::string BMX160Serial::read_line()
{
  std::string line;
  char ch;
  while (read(serial_fd, &ch, 1) > 0 && ch != '\n') {
    if (ch != '\r') {
      line += ch;
    }
  }
  return line;
}

// Parses a single line of sensor data and updates the SensorData struct
void BMX160Serial::parse_line(const std::string & line, SensorData & data)
{
  std::regex pattern(
    R"(([MAGR]) X:\s*(-?\d+\.\d+)\s+Y:\s*(-?\d+\.\d+)\s+Z:\s*(-?\d+\.\d+)(?:\s+W:\s*(-?\d+\.\d+))?)");

  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    char type = match[1].str()[0];
    float x = std::stof(match[2].str());
    float y = std::stof(match[3].str());
    float z = std::stof(match[4].str());
    switch (type) {
      case 'R': {
        float w = std::stof(match[5].str());
        data.quat_x = x;
        data.quat_y = y;
        data.quat_z = z;
        data.quat_w = w;
        break;
      }
      case 'M':
        data.mag_x = x;
        data.mag_y = y;
        data.mag_z = z;
        break;
      case 'G':
        data.gyro_x = x;
        data.gyro_y = y;
        data.gyro_z = z;
        break;
      case 'A':
        data.accel_x = x;
        data.accel_y = y;
        data.accel_z = z;
        break;
    }
  }
}

// Reads and returns the latest sensor data
BMX160Serial::SensorData BMX160Serial::read_sensor_data()
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  SensorData data = {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};

  while (std::isnan(data.quat_x) || std::isnan(data.gyro_x) || std::isnan(data.accel_x) ||
         std::isnan(data.mag_x)) {
    std::string line = read_line();
    if (line.empty()) {
      continue;
    }

    parse_line(line, data);
  }

  return data;
}
