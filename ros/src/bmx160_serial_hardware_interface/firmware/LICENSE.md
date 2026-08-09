# Firmware licensing exception

This firmware directory is NOT covered by the repository's root
Apache-2.0 license.

- The calibration sketch `main.ino` uses only DP Engineering's
  MIT-licensed BMX160 driver and the Adafruit/Arduino libraries; it does
  not link any GPL code.
- The sketch in `main/` links the x-io Technologies Madgwick/Mahony AHRS
  libraries (http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/),
  published under the GNU General Public License, so that compiled
  firmware as a whole is distributed under the GPL. Upstream does not
  state a GPL version; per the GPL's own terms you may follow any
  published version.
