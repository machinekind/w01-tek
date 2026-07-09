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

#include "bmi160_serial_hardware_interface/bmi160_serial.hpp"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

BMI160Serial::BMI160Serial(const std::string & port_name, int baud_rate)
: port_name_{port_name}, baud_rate_{baud_rate}
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  latest_data_ = {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

bool BMI160Serial::initialize()
{
  serial_fd = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (serial_fd < 0) {
    std::cerr << "Error opening serial port " << port_name_ << ": " << strerror(errno)
              << std::endl;
    return false;
  }

  struct termios tty;
  if (tcgetattr(serial_fd, &tty) != 0) {
    std::cerr << "Error from tcgetattr" << std::endl;
    return false;
  }

  cfsetospeed(&tty, baud_rate_);
  cfsetispeed(&tty, baud_rate_);

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_iflag &= ~IGNBRK;
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  // VMIN=0 + VTIME=1: each read() returns after at most 0.1 s even with no
  // data, so the reader thread stays responsive to shutdown and a silent
  // sensor never blocks anything indefinitely.
  tty.c_cc[VMIN] = 0;
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

  // Hard-reset the ESP32 through the board's auto-reset circuit (RTS drives
  // EN, DTR drives IO0), same sequence esptool uses to reboot into the app.
  // Previous openers of the port can leave the chip held in reset, stuck in
  // the ROM bootloader, or with the firmware wedged mid-print streaming
  // garbage forever -- a fresh boot on every activation makes the sensor
  // state deterministic instead of depending on port-open history.
  int dtr_bit = TIOCM_DTR;
  int rts_bit = TIOCM_RTS;
  if (
    ioctl(serial_fd, TIOCMBIC, &dtr_bit) != 0 ||   // IO0 high: boot to app
    ioctl(serial_fd, TIOCMBIS, &rts_bit) != 0) {   // EN low: chip in reset
    std::cerr << "Warning: could not toggle DTR/RTS for ESP32 reset" << std::endl;
  } else {
    usleep(100 * 1000);
    ioctl(serial_fd, TIOCMBIC, &rts_bit);  // EN high: chip boots
  }

  tcflush(serial_fd, TCIFLUSH);

  running_ = true;
  reader_thread_ = std::thread(&BMI160Serial::reader_loop, this);
  return true;
}

BMI160Serial::~BMI160Serial()
{
  running_ = false;
  if (reader_thread_.joinable()) {
    reader_thread_.join();
  }
  if (serial_fd >= 0) {
    close(serial_fd);
  }
}

// Reads a line of input from the serial port. Returns an empty string when
// the port times out (VMIN=0/VTIME=1) before a full line arrives -- a
// partial line is discarded rather than returned, so a later read never
// parses a truncated tail as a fresh line.
std::string BMI160Serial::read_line()
{
  std::string line;
  char ch;
  while (running_) {
    ssize_t n = read(serial_fd, &ch, 1);
    if (n <= 0) {
      return "";  // timeout or error: drop any partial line
    }
    if (ch == '\n') {
      return line;
    }
    if (ch != '\r') {
      line += ch;
    }
  }
  return "";
}

// Background thread: keeps latest_data_ current with whatever the firmware
// streams. Blocking here is harmless -- read_sensor_data() never waits.
void BMI160Serial::reader_loop()
{
  SensorData working;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    working = latest_data_;
  }

  while (running_) {
    std::string line = read_line();
    if (line.empty()) {
      continue;
    }
    parse_line(line, working);
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_data_ = working;
    }
  }
}

// Parses a single line of sensor data and updates the SensorData struct
void BMI160Serial::parse_line(const std::string & line, SensorData & data)
{
  std::regex pattern(
    R"(([GAR]) X:\s*(-?\d+\.\d+)\s+Y:\s*(-?\d+\.\d+)\s+Z:\s*(-?\d+\.\d+)(?:\s+W:\s*(-?\d+\.\d+))?)");

  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    char type = match[1].str()[0];
    float x = std::stof(match[2].str());
    float y = std::stof(match[3].str());
    float z = std::stof(match[4].str());
    switch (type) {
      case 'R': {
        if (!match[5].matched) {
          // Truncated R line (e.g. opened the port mid-stream) -- W is
          // missing and stof("") would throw. Skip the line instead.
          break;
        }
        float w = std::stof(match[5].str());
        data.quat_x = x;
        data.quat_y = y;
        data.quat_z = z;
        data.quat_w = w;
        break;
      }
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

// Returns the most recent sample without blocking. Fields the firmware has
// not delivered yet (or ever, if the sensor is silent) stay NaN; the caller
// latches previous values for those.
BMI160Serial::SensorData BMI160Serial::read_sensor_data()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  return latest_data_;
}
