#!/usr/bin/env python3
"""Bench test for the Gravity BMX160 (DFRobot SEN0373) on the Pi's I2C1.

    python3 bmx160_i2c_test.py [--bus 1] [--addr 0x68] [--seconds 10]

No ROS, no MCU firmware, no serial -- talks straight to /dev/i2c-<bus>.
Checks, in order:
  1. chip ID (0xD8 = BMX160; 0xD1 means a plain BMI160 answered -- no mag),
  2. power-up of accel/gyro/mag incl. the magnetometer's indirect-interface
     init sequence (BMM150 behind MAG_IF, same sequence DFRobot's lib uses),
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

# Register map (BMX160 datasheet; BMI160-compatible core + BMM150 behind MAG_IF)
REG_CHIP_ID = 0x00      # 0xD8 BMX160, 0xD1 BMI160
REG_ERR = 0x02
REG_PMU_STATUS = 0x03   # bits [5:4] acc, [3:2] gyr, [1:0] mag; 0b01 = normal
REG_DATA = 0x04         # mag xyz (6) + rhall (2) + gyr xyz (6) + acc xyz (6)
REG_MAG_CONF = 0x44
REG_MAG_IF_0 = 0x4C     # mag interface: mode/burst
REG_MAG_IF_1 = 0x4D     # read address
REG_MAG_IF_2 = 0x4E     # write address
REG_MAG_IF_3 = 0x4F     # write data
REG_ACC_CONF = 0x40
REG_ACC_RANGE = 0x41
REG_GYR_CONF = 0x42
REG_GYR_RANGE = 0x43
REG_CMD = 0x7E

CMD_SOFTRESET = 0xB6
CMD_ACC_NORMAL = 0x11
CMD_GYR_NORMAL = 0x15
CMD_MAG_NORMAL = 0x19

ACC_ODR_100HZ_NORMAL = 0x28   # bwp=normal, odr=100 Hz
ACC_RANGE_4G = 0x05
GYR_ODR_100HZ_NORMAL = 0x28
GYR_RANGE_500DPS = 0x02

ACC_MS2_PER_LSB = 4.0 / 32768.0 * 9.80665   # +/-4 g
GYR_DPS_PER_LSB = 500.0 / 32768.0           # +/-500 dps
MAG_UT_PER_LSB = 0.3                        # BMM150 nominal, good enough here


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v


class Bmx160:
    def __init__(self, bus, addr):
        self.bus = bus
        self.addr = addr

    def w(self, reg, val, delay_s=0.001):
        self.bus.write_byte_data(self.addr, reg, val)
        time.sleep(delay_s)

    def r(self, reg):
        return self.bus.read_byte_data(self.addr, reg)

    def init(self):
        self.w(REG_CMD, CMD_SOFTRESET, 0.05)
        self.w(REG_CMD, CMD_ACC_NORMAL, 0.05)
        self.w(REG_CMD, CMD_GYR_NORMAL, 0.10)
        self.w(REG_CMD, CMD_MAG_NORMAL, 0.01)
        self.w(REG_ACC_CONF, ACC_ODR_100HZ_NORMAL)
        self.w(REG_ACC_RANGE, ACC_RANGE_4G)
        self.w(REG_GYR_CONF, GYR_ODR_100HZ_NORMAL)
        self.w(REG_GYR_RANGE, GYR_RANGE_500DPS)
        # Magnetometer = BMM150 behind the indirect MAG_IF: put the interface
        # in setup mode, poke the BMM150's own registers through IF_2/IF_3,
        # then switch to data (burst-read) mode. Same sequence as DFRobot's
        # library, with the ODR raised to 100 Hz.
        self.w(REG_MAG_IF_0, 0x80, 0.05)   # setup mode
        self.w(REG_MAG_IF_3, 0x01)         # BMM150 0x4B <- power on
        self.w(REG_MAG_IF_2, 0x4B)
        self.w(REG_MAG_IF_3, 0x04)         # BMM150 0x51 <- REPXY regular
        self.w(REG_MAG_IF_2, 0x51)
        self.w(REG_MAG_IF_3, 0x0E)         # BMM150 0x52 <- REPZ regular
        self.w(REG_MAG_IF_2, 0x52)
        self.w(REG_MAG_IF_3, 0x02)         # BMM150 0x4C <- forced mode
        self.w(REG_MAG_IF_2, 0x4C)
        self.w(REG_MAG_IF_1, 0x42)         # burst read from BMM150 data regs
        self.w(REG_MAG_CONF, 0x08)         # mag ODR 100 Hz
        self.w(REG_MAG_IF_0, 0x00, 0.05)   # data mode

    def read_sample(self):
        d = self.bus.read_i2c_block_data(self.addr, REG_DATA, 20)
        mag = [s16(d[0], d[1]), s16(d[2], d[3]), s16(d[4], d[5])]
        gyr = [s16(d[8], d[9]), s16(d[10], d[11]), s16(d[12], d[13])]
        acc = [s16(d[14], d[15]), s16(d[16], d[17]), s16(d[18], d[19])]
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
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x68)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    ok = True
    with SMBus(args.bus) as bus:
        dev = Bmx160(bus, args.addr)

        chip = dev.r(REG_CHIP_ID)
        name = {0xD8: "BMX160", 0xD1: "BMI160 (no magnetometer!)"}.get(
            chip, "UNKNOWN")
        ok &= check("chip id", chip == 0xD8, f"0x{chip:02X} = {name}")

        dev.init()
        err = dev.r(REG_ERR)
        pmu = dev.r(REG_PMU_STATUS)
        ok &= check("err reg", err == 0, f"0x{err:02X}")
        ok &= check("pmu status", pmu == 0x15,
                    f"0x{pmu:02X} (0x15 = acc/gyr/mag all normal)")

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
            time.sleep(0.005)  # poll ~200 Hz; data updates at 100 Hz
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
