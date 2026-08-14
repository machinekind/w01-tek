#!/usr/bin/env python3
"""How the computer running wojtek is doing, on /wojtek/sysinfo.

    ros2 run wojtek_telemetry sysinfo_node
    ros2 run wojtek_telemetry sysinfo_node --ros-args -p rate_hz:=2.0

CPU, memory, SoC temperature, Raspberry Pi throttling, free space where the
bag is written, and wifi traffic. One timer callback reads all of it and
stamps the message once, so the fields can be compared with each other.
Every run records all topics, so the same numbers are in the bag afterwards.

The throttle status comes from `vcgencmd get_throttled`, with the sysfs copy
of the same word as a fallback. A host with neither of those, such as a Mac
running the docker simulation, leaves the Pi fields at zero and gets one log
line saying so at startup.
"""

import os
import subprocess
import time

import psutil
import rclpy
from rclpy.node import Node

from wojtek_telemetry.msg import SysInfo

# Bits of the Raspberry Pi throttle word. The low bits are the current state.
# The high bits latch on the first occurrence and stay set until reboot.
_UNDERVOLTAGE_NOW = 1 << 0
_FREQ_CAPPED_NOW = 1 << 1
_THROTTLED_NOW = 1 << 2
_UNDERVOLTAGE_EVER = 1 << 16
_FREQ_CAPPED_EVER = 1 << 17
_THROTTLED_EVER = 1 << 18

_THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
# The Pi firmware exposes the same word vcgencmd prints, as hex text.
_SYSFS_THROTTLED = "/sys/devices/platform/soc/soc:firmware/get_throttled"


def _read_text(path):
    """Contents of a file, stripped. None when it is not readable."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _parse_throttled(text):
    """The hex word out of "throttled=0x0" or out of a bare "0x0"."""
    word = text.strip().rpartition("=")[2]
    try:
        return int(word, 16)
    except ValueError:
        return None


def _vcgencmd_throttled():
    """The throttle word from vcgencmd. None when that tool is not here."""
    try:
        done = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=1.0, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_throttled(done.stdout)


def _sysfs_throttled():
    """The throttle word straight from sysfs. None when the file is absent."""
    text = _read_text(_SYSFS_THROTTLED)
    return None if text is None else _parse_throttled(text)


def _pick_throttle_reader():
    """The first throttle source that answers on this host, or None."""
    for read in (_vcgencmd_throttled, _sysfs_throttled):
        if read() is not None:
            return read
    return None


class SysInfoNode(Node):
    def __init__(self):
        super().__init__("wojtek_sysinfo")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter(
            "disk_path", os.path.join(os.path.expanduser("~"), "wojtek_bags")
        )
        self.declare_parameter("net_iface", "wlan0")

        rate = float(self.get_parameter("rate_hz").value)
        if rate <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate}")
        self._disk_path = self.get_parameter("disk_path").value
        self._iface = self.get_parameter("net_iface").value

        # Which throttle source works here, decided once so a missing
        # vcgencmd does not cost a failed process launch every tick.
        self._read_throttled = _pick_throttle_reader()
        if self._read_throttled is None:
            self.get_logger().info(
                "no Raspberry Pi throttle status on this host, so the "
                "throttle flags stay false"
            )
        if not os.path.exists(_THERMAL_ZONE):
            self.get_logger().info(
                f"no {_THERMAL_ZONE} on this host, so the SoC temperature "
                f"stays at zero"
            )

        # Traffic counters from the previous tick. A rate needs two samples,
        # so the first message reports no traffic.
        self._net_prev = None
        self._net_prev_t = None
        # psutil measures CPU use between calls, and its first call has
        # nothing to compare against. Spend that one here.
        psutil.cpu_percent(percpu=True)

        self._pub = self.create_publisher(SysInfo, "wojtek/sysinfo", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"system info at {rate:g} Hz; disk {self._disk_path}, "
            f"network interface {self._iface}"
        )

    def _tick(self):
        msg = SysInfo()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.cpu_percent = [float(p) for p in psutil.cpu_percent(percpu=True)]
        msg.load1 = float(os.getloadavg()[0])
        msg.cpu_freq_mhz = self._cpu_freq_mhz()
        msg.soc_temp_c = self._soc_temp_c()

        word = self._throttled_word()
        msg.undervoltage_now = bool(word & _UNDERVOLTAGE_NOW)
        msg.undervoltage_ever = bool(word & _UNDERVOLTAGE_EVER)
        msg.throttled_now = bool(word & _THROTTLED_NOW)
        msg.throttled_ever = bool(word & _THROTTLED_EVER)
        msg.freq_capped_now = bool(word & _FREQ_CAPPED_NOW)
        msg.freq_capped_ever = bool(word & _FREQ_CAPPED_EVER)

        mem = psutil.virtual_memory()
        msg.mem_used_bytes = int(mem.used)
        msg.mem_total_bytes = int(mem.total)
        msg.swap_used_bytes = int(psutil.swap_memory().used)
        msg.disk_free_bytes = self._disk_free_bytes()

        msg.wifi_rx_bytes_per_s, msg.wifi_tx_bytes_per_s = self._net_rates()
        self._pub.publish(msg)

    def _cpu_freq_mhz(self):
        try:
            freq = psutil.cpu_freq()
        except Exception:
            # psutil raises rather than returns on hosts that report no
            # frequency at all, macOS among them.
            freq = None
        return float(freq.current) if freq else 0.0

    def _soc_temp_c(self):
        text = _read_text(_THERMAL_ZONE)
        if text is None:
            return 0.0
        try:
            return int(text) / 1000.0
        except ValueError:
            return 0.0

    def _throttled_word(self):
        if self._read_throttled is None:
            return 0
        word = self._read_throttled()
        return 0 if word is None else word

    def _disk_free_bytes(self):
        """Free space on the filesystem that holds disk_path.

        The bag directory does not exist until the first recording, so walk
        up to a parent that does. That parent sits on the same filesystem,
        which is the thing being measured.
        """
        path = self._disk_path
        while path:
            try:
                return int(psutil.disk_usage(path).free)
            except OSError:
                parent = os.path.dirname(path)
                if parent == path:
                    return 0
                path = parent
        return 0

    def _net_rates(self):
        """Wifi traffic in bytes per second, from the counter difference.

        A missing interface and the very first tick both report no traffic.
        The counters restart from zero when the interface goes down and
        comes back, which would otherwise show up as a negative rate.
        """
        counters = psutil.net_io_counters(pernic=True).get(self._iface)
        now = time.monotonic()
        if counters is None:
            self._net_prev = None
            return 0.0, 0.0
        prev, prev_t = self._net_prev, self._net_prev_t
        self._net_prev = (counters.bytes_recv, counters.bytes_sent)
        self._net_prev_t = now
        if prev is None or now <= prev_t:
            return 0.0, 0.0
        dt = now - prev_t
        return (
            max(0.0, (counters.bytes_recv - prev[0]) / dt),
            max(0.0, (counters.bytes_sent - prev[1]) / dt),
        )


def main():
    rclpy.init()
    node = SysInfoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
