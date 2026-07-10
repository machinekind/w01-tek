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

#ifndef BMI160_SERIAL_H
#define BMI160_SERIAL_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class BMI160Serial
{
public:
  using SharedPtr = std::shared_ptr<BMI160Serial>;
  struct SensorData
  {
    float gyro_x, gyro_y, gyro_z;          // Gyroscope data in g
    float accel_x, accel_y, accel_z;       // Accelerometer data in m/s^2
    float quat_w, quat_x, quat_y, quat_z;  // Quaternion data (gyro+accel fusion, no mag)
  };

  BMI160Serial(const std::string & port_name, int baud_rate = 115200);
  ~BMI160Serial();

  bool initialize();  // Opens the port and starts the background reader

  // Returns the most recent complete sample immediately (never blocks).
  // Fields are NaN until the first line of that type has arrived.
  SensorData read_sensor_data();

private:
  int serial_fd = -1;
  std::string port_name_;
  int baud_rate_;

  // The firmware streams at its own 100 Hz pace while the ros2_control
  // update loop may run much faster (e.g. 400 Hz) -- and, worse, a silent
  // sensor must never block that loop (a blocked read() freezes every
  // controller_manager service, which synchronizes with the loop). So all
  // serial I/O happens on this background thread; the control loop only
  // copies latest_data_.
  std::thread reader_thread_;
  std::atomic<bool> running_{false};
  std::mutex data_mutex_;
  SensorData latest_data_;

  void reader_loop();
  std::string read_line();
  void parse_line(const std::string & line, SensorData & data);
};

#endif  // BMI160_SERIAL_H
