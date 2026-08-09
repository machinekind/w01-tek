# Firmware licensing exception

This firmware directory is NOT covered by the repository's root
Apache-2.0 license.

- `BMX160/Mahony_BMX160.{h,cpp}` are Madgwick's implementation of
  Mahony's AHRS algorithm from x-io Technologies
  (http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/),
  published under the GNU General Public License.
- `BMX160/DPEng_BMX160.{h,cpp}` are DP Engineering's BMX160 Arduino
  driver, under the MIT license (see the file headers, which must be
  retained in any redistribution).
- The sketch in `main/` links the Mahony filter, so the compiled
  firmware as a whole is distributed under the GNU General Public
  License. Upstream does not state a GPL version; per the GPL's own
  terms you may follow any published version.
