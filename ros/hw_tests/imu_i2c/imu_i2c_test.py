#!/usr/bin/env python3
"""Bench test for the Adafruit 5543 9-DoF IMU (LSM6DS3TR-C + LIS3MDL) on I2C.

    python3 imu_i2c_test.py [--bus 1] [--addr-ag 0x6A] [--addr-mag 0x1C]
                            [--seconds 10]

No ROS, no MCU firmware, no serial -- talks straight to /dev/i2c-<bus>.
The board carries two independent ST chips on the same bus:
  * LSM6DS3TR-C  accel + gyro   (default address 0x6A, 0x6B if SDO high)
  * LIS3MDL      magnetometer   (default address 0x1C, 0x1E if SDO high)
Checks, in order:
  1. WHO_AM_I of both chips (LSM6DS3TR-C: 0x6A, LIS3MDL: 0x3D),
  2. power-up/config (accel 104 Hz +/-4 g, gyro 104 Hz +/-500 dps, BDU;
     mag ultra-high-performance 80 Hz continuous +/-4 gauss),
  3. a streaming run: achieved sample rate, |accel| mean/std at rest
     (expect ~9.81 m/s^2), gyro bias (expect < ~3 dps), mag norm plausible
     (~25-65 uT indoors).
Prints PASS/FAIL per criterion and exits non-zero on any FAIL.
"""

import argparse
import math
import statistics
import sys
import time

try:
    from smbus2 import SMBus
except ImportError:  # classic python3-smbus has the same API surface we use
    from smbus import SMBus

# --- LSM6DS3TR-C (accel + gyro) register map ---
LSM6_WHO_AM_I = 0x0F     # reads 0x6A
LSM6_CTRL1_XL = 0x10     # accel ODR/FS
LSM6_CTRL2_G = 0x11      # gyro ODR/FS
LSM6_CTRL3_C = 0x12      # BDU, IF_INC, SW_RESET
LSM6_STATUS = 0x1E       # bit0 XLDA, bit1 GDA
LSM6_OUTX_L_G = 0x22     # gyro xyz (6) then accel xyz (6) at 0x28

LSM6_ID = 0x6A
LSM6_SW_RESET = 0x01
LSM6_BDU_IFINC = 0x44          # block data update + register auto-increment
LSM6_XL_104HZ_4G = 0x48        # ODR=104 Hz (0100....), FS=+/-4 g (..10..)
LSM6_G_104HZ_500DPS = 0x44     # ODR=104 Hz, FS=+/-500 dps

ACC_MS2_PER_LSB = 0.122e-3 * 9.80665   # +/-4 g -> 0.122 mg/LSB
GYR_DPS_PER_LSB = 17.5e-3              # +/-500 dps -> 17.5 mdps/LSB

# --- LIS3MDL (magnetometer) register map ---
LIS3_WHO_AM_I = 0x0F     # reads 0x3D
LIS3_CTRL_REG1 = 0x20
LIS3_CTRL_REG2 = 0x21
LIS3_CTRL_REG3 = 0x22
LIS3_CTRL_REG4 = 0x23
LIS3_OUT_X_L = 0x28
LIS3_AUTOINC = 0x80      # ST mag: set sub-address MSb for multi-byte reads

LIS3_ID = 0x3D
LIS3_UHP_80HZ = 0x7C     # OM=ultra-high (xy), DO=80 Hz
LIS3_FS_4GAUSS = 0x00
LIS3_CONTINUOUS = 0x00
LIS3_UHP_Z = 0x0C        # OMZ=ultra-high (z)

MAG_UT_PER_LSB = 100.0 / 6842.0   # +/-4 gauss -> 6842 LSB/gauss; 1 G = 100 uT


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v


class Imu:
    def __init__(self, bus, addr_ag, addr_mag):
        self.bus = bus
        self.ag = addr_ag
        self.mag = addr_mag

    def w(self, addr, reg, val, delay_s=0.001):
        self.bus.write_byte_data(addr, reg, val)
        time.sleep(delay_s)

    def r(self, addr, reg):
        return self.bus.read_byte_data(addr, reg)

    def init(self):
        self.w(self.ag, LSM6_CTRL3_C, LSM6_SW_RESET, 0.02)
        self.w(self.ag, LSM6_CTRL3_C, LSM6_BDU_IFINC)
        self.w(self.ag, LSM6_CTRL1_XL, LSM6_XL_104HZ_4G)
        self.w(self.ag, LSM6_CTRL2_G, LSM6_G_104HZ_500DPS)
        self.w(self.mag, LIS3_CTRL_REG1, LIS3_UHP_80HZ)
        self.w(self.mag, LIS3_CTRL_REG2, LIS3_FS_4GAUSS)
        self.w(self.mag, LIS3_CTRL_REG4, LIS3_UHP_Z)
        self.w(self.mag, LIS3_CTRL_REG3, LIS3_CONTINUOUS, 0.02)

    def read_sample(self):
        # One 12-byte burst: gyro xyz then accel xyz (IF_INC crosses 0x27->0x28)
        d = self.bus.read_i2c_block_data(self.ag, LSM6_OUTX_L_G, 12)
        gyr = [s16(d[0], d[1]), s16(d[2], d[3]), s16(d[4], d[5])]
        acc = [s16(d[6], d[7]), s16(d[8], d[9]), s16(d[10], d[11])]
        m = self.bus.read_i2c_block_data(self.mag, LIS3_OUT_X_L | LIS3_AUTOINC, 6)
        mag = [s16(m[0], m[1]), s16(m[2], m[3]), s16(m[4], m[5])]
        return (
            [v * ACC_MS2_PER_LSB for v in acc],
            [v * GYR_DPS_PER_LSB for v in gyr],
            [v * MAG_UT_PER_LSB for v in mag],
        )


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr-ag", type=lambda x: int(x, 0), default=0x6A,
                    help="LSM6DS3TR-C address (0x6B if SDO pulled high)")
    ap.add_argument("--addr-mag", type=lambda x: int(x, 0), default=0x1C,
                    help="LIS3MDL address (0x1E if SDO pulled high)")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    ok = True
    with SMBus(args.bus) as bus:
        dev = Imu(bus, args.addr_ag, args.addr_mag)

        who_ag = dev.r(dev.ag, LSM6_WHO_AM_I)
        ok &= check("LSM6DS3TR-C id", who_ag == LSM6_ID,
                    f"0x{who_ag:02X} (expect 0x6A)")
        who_mag = dev.r(dev.mag, LIS3_WHO_AM_I)
        ok &= check("LIS3MDL id", who_mag == LIS3_ID,
                    f"0x{who_mag:02X} (expect 0x3D)")
        if not ok:
            print("\n== OVERALL: FAIL == (wrong chip -- check i2cdetect -y 1)")
            sys.exit(1)

        dev.init()
        status = dev.r(dev.ag, LSM6_STATUS)
        ok &= check("accel/gyro data ready", status & 0x03 == 0x03,
                    f"STATUS_REG 0x{status:02X} (bit0 XLDA, bit1 GDA)")

        acc_n, gyr, mag_n = [], [[], [], []], []
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < args.seconds:
            a, g, m = dev.read_sample()
            acc_n.append(math.sqrt(sum(v * v for v in a)))
            for i in range(3):
                gyr[i].append(g[i])
            mag_n.append(math.sqrt(sum(v * v for v in m)))
            n += 1
            time.sleep(0.005)  # poll ~200 Hz; data updates at 104/80 Hz
        rate = n / (time.monotonic() - t0)

        acc_mean = statistics.fmean(acc_n)
        acc_std = statistics.pstdev(acc_n)
        bias = [statistics.fmean(x) for x in gyr]
        mag_mean = statistics.fmean(mag_n)

        ok &= check("sample rate", rate > 90,
                    f"{rate:.0f} Hz effective polling")
        ok &= check("|accel| at rest", abs(acc_mean - 9.81) < 0.5,
                    f"{acc_mean:.2f} +/- {acc_std:.3f} m/s^2 (expect ~9.81)")
        ok &= check("gyro bias", all(abs(b) < 3.0 for b in bias),
                    "[" + " ".join(f"{b:+.2f}" for b in bias) + "] dps")
        ok &= check("mag norm", 15.0 < mag_mean < 100.0,
                    f"{mag_mean:.1f} uT (indoors ~25-65)")

    print("\n== OVERALL:", "PASS" if ok else "FAIL", "==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
