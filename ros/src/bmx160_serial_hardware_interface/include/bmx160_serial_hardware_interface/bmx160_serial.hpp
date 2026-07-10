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

#ifndef BMX160_SERIAL_H
#define BMX160_SERIAL_H

#include <memory>
#include <string>

class BMX160Serial
{
public:
  using SharedPtr = std::shared_ptr<BMX160Serial>;
  struct SensorData
  {
    float mag_x, mag_y, mag_z;             // Magnetometer data in uT
    float gyro_x, gyro_y, gyro_z;          // Gyroscope data in g
    float accel_x, accel_y, accel_z;       // Accelerometer data in m/s^2
    float quat_w, quat_x, quat_y, quat_z;  // Quaternion data
  };

  BMX160Serial(const std::string & port_name, int baud_rate = 9600);
  ~BMX160Serial();

  bool initialize();              // Opens and configures the serial port
  SensorData read_sensor_data();  // Reads and parses the data from the sensor

private:
  int serial_fd;
  std::string port_name_;
  int baud_rate_;
  std::string read_line();
  void parse_line(const std::string & line, SensorData & data);
};

#endif  // BMX160_SERIAL_H
