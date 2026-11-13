// Copyright (c) 2026, Machinekind
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

#ifndef MD80_HARDWARE_INTERFACE__DRIVE_RESPONSE_AGE_HPP_
#define MD80_HARDWARE_INTERFACE__DRIVE_RESPONSE_AGE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace md80_hardware_interface
{

// Each drive's age measures seconds since its raw position, velocity or
// torque last changed. A change resets that drive's age to 0.0.
// Only resize() allocates storage.
class DriveResponseAge
{
public:
  // resize() sets the drive count to n. It clears the baseline.
  void resize(std::size_t n)
  {
    prev_.assign(3 * n, std::numeric_limits<double>::quiet_NaN());
    last_change_ns_.assign(n, 0);
    baseline_set_ = false;
  }

  // The next timed update() sets a new baseline after reset().
  void reset() { baseline_set_ = false; }

  std::size_t size() const { return last_change_ns_.size(); }
  bool baseline_set() const { return baseline_set_; }

  // readback must contain position, velocity and torque for each of size() drives.
  // ages must contain size() entries. Each entry receives its drive's age.
  // timed must be false for reads without a usable time.
  // Untimed reads return ages of 0.0. They clear the baseline.
  // The next timed update sets a new baseline.
  void update(
    int64_t now_ns, bool timed, const std::vector<double> & readback, std::vector<double> & ages)
  {
    const bool baseline = timed && !baseline_set_;
    const std::size_t n = size();
    for (std::size_t i = 0; i < n; ++i) {
      const double * cur = &readback[3 * i];
      double * prev = &prev_[3 * i];
      const bool changed = cur[0] != prev[0] || cur[1] != prev[1] || cur[2] != prev[2];
      prev[0] = cur[0];
      prev[1] = cur[1];
      prev[2] = cur[2];
      if (!timed) {
        ages[i] = 0.0;
      } else if (baseline || changed) {
        last_change_ns_[i] = now_ns;
        ages[i] = 0.0;
      } else if (now_ns <= last_change_ns_[i]) {
        // The age stays at 0.0 while the clock is at or before the last change.
        ages[i] = 0.0;
      } else {
        // Integer subtraction preserves nanosecond precision at epoch times.
        ages[i] = 1e-9 * static_cast<double>(now_ns - last_change_ns_[i]);
      }
    }
    baseline_set_ = timed;
  }

private:
  std::vector<double> prev_;
  std::vector<int64_t> last_change_ns_;
  bool baseline_set_ = false;
};

}  // namespace md80_hardware_interface

#endif  // MD80_HARDWARE_INTERFACE__DRIVE_RESPONSE_AGE_HPP_
