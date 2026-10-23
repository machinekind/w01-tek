#!/usr/bin/env python3
"""One measurement window of the robot computer, as a single JSON document.

Runs ON the RPi next to a live stack (ROS environment sourced) and records,
over --duration seconds:

  * topics   -- rate and inter-arrival jitter of the control-path topics,
                and for stamped ones the publish-to-receive latency;
  * policy   -- /wojtek/policy_timing (inference_ms, period_ms), which the
                stack publishes with telemetry:=true;
  * cpu      -- utilisation per core, sampled every 0.5 s;
  * procs    -- CPU share of every busy process and of the threads of the
                stack's own processes: core, scheduling class, RT priority,
                allowed cores, involuntary context switches;
  * irqs     -- interrupts per core over the window (the isolated cores
                should take next to none);
  * board    -- temperature, clock, throttle flags, memory, network bytes.

It never commands anything: subscriptions only, no services, no params.
Pin it off the control cores (the perf.sh runner uses taskset -c 0); its
own CPU is reported under "probe" so it can be subtracted.

    python3 probe.py --duration 30 --label baseline > result.json
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Imu, JointState, Joy
from std_msgs.msg import Float64MultiArray

try:
    from wojtek_telemetry.msg import PolicyTiming
except ImportError:  # telemetry package not built: the rest still works
    PolicyTiming = None

CLK_TCK = os.sysconf("SC_CLK_TCK")

# Topic -> message type. Best-effort subscriptions: they match reliable
# publishers too and add no acknowledgement traffic to the stack.
TOPICS = {
    "/joint_states": JointState,
    "/wojtek/joint_states_abs": JointState,
    "/wojtek/joint_targets": JointState,
    "/forward_position_controller/commands": Float64MultiArray,
    "/forward_effort_controller/commands": Float64MultiArray,
    "/imu_sensor_broadcaster/imu": Imu,
    "/cmd_vel": Twist,
    "/joy": Joy,
}
# Subscribing to the compressed stream makes the camera node encode JPEGs
# (image_transport encodes only for subscribers), i.e. the probe itself
# becomes a viewer. Opt-in for that reason.
CAMERA_TOPIC = "/camera/camera/color/image_raw/compressed"

# Processes whose threads are listed one by one.
STACK_PATTERN = re.compile(
    r"ros2_control_node|policy_node|real_io_node|robot_state_publisher|"
    r"joy_node|gamepad_teleop|deck_gateway|realsense2_camera|sysinfo_node|"
    r"foxglove_bridge|ros2 bag|spawner|static_transform"
)


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))]


def summarize(values, digits=3):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), digits),
        "p50": round(pct(values, 50), digits),
        "p99": round(pct(values, 99), digits),
        "max": round(max(values), digits),
        "std": round(statistics.pstdev(values), digits),
    }


# -- /proc readers -------------------------------------------------------------

def read_cpu_times():
    """Per-core (busy, total) jiffies from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            vals = [int(v) for v in parts[1:]]
            idle = vals[3] + vals[4]  # idle + iowait
            out[parts[0]] = (sum(vals) - idle, sum(vals))
    return out


def read_task_stat(path):
    """Fields of a /proc/<pid>[/task/<tid>]/stat file, comm-safe."""
    with open(path) as f:
        raw = f.read()
    rparen = raw.rindex(")")
    comm = raw[raw.index("(") + 1:rparen]
    rest = raw[rparen + 2:].split()
    # rest[0] is field 3 (state); field n is rest[n - 3].
    return {
        "comm": comm,
        "ticks": int(rest[11]) + int(rest[12]),  # utime + stime
        "processor": int(rest[36]),
        "rt_priority": int(rest[37]),
        "policy": int(rest[38]),
    }


def read_status_fields(path, keys):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in keys:
                    out[k] = v.strip()
    except OSError:
        pass
    return out


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def snapshot_processes():
    """{pid: {...}} with per-thread data for the stack's processes."""
    procs = {}
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d.rsplit("/", 1)[1])
        try:
            st = read_task_stat(f"{d}/stat")
        except (OSError, ValueError):
            continue
        cmd = cmdline(pid)
        entry = {"comm": st["comm"], "cmd": cmd[:160], "ticks": st["ticks"]}
        if STACK_PATTERN.search(cmd):
            status = read_status_fields(f"{d}/status", {"Cpus_allowed_list", "VmRSS"})
            entry["cpus_allowed"] = status.get("Cpus_allowed_list")
            entry["rss"] = status.get("VmRSS")
            threads = {}
            for t in glob.glob(f"{d}/task/[0-9]*"):
                try:
                    ts = read_task_stat(f"{t}/stat")
                except (OSError, ValueError):
                    continue
                cs = read_status_fields(
                    f"{t}/status",
                    {"nonvoluntary_ctxt_switches", "voluntary_ctxt_switches"},
                )
                ts["nvcsw"] = int(cs.get("nonvoluntary_ctxt_switches", 0))
                ts["vcsw"] = int(cs.get("voluntary_ctxt_switches", 0))
                threads[int(t.rsplit("/", 1)[1])] = ts
            entry["threads"] = threads
        procs[pid] = entry
    return procs


def read_interrupts():
    """{irq label: [count per cpu]}."""
    out = {}
    with open("/proc/interrupts") as f:
        header = f.readline().split()
        ncpu = len(header)
        for line in f:
            parts = line.split()
            if not parts:
                continue
            counts = []
            for p in parts[1:1 + ncpu]:
                if not p.isdigit():
                    break
                counts.append(int(p))
            if len(counts) != ncpu:
                continue
            label = parts[0].rstrip(":") + " " + " ".join(parts[1 + ncpu:])
            out[label.strip()] = counts
    return out


def read_net():
    out = {}
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, _, data = line.partition(":")
            vals = data.split()
            out[name.strip()] = (int(vals[0]), int(vals[8]))  # rx, tx bytes
    return out


def read_meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0])  # kB
    return {
        "total_mb": out["MemTotal"] // 1024,
        "available_mb": out["MemAvailable"] // 1024,
        "swap_used_mb": (out["SwapTotal"] - out["SwapFree"]) // 1024,
    }


def vcgencmd(*args):
    try:
        return subprocess.run(
            ["vcgencmd", *args], capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def board_state():
    temp = vcgencmd("measure_temp")  # temp=41.3'C
    clock = vcgencmd("measure_clock", "arm")  # frequency(48)=1500345728
    m = re.search(r"([\d.]+)", temp)
    c = re.search(r"=(\d+)", clock)
    return {
        "temp_c": float(m.group(1)) if m else None,
        "arm_mhz": int(c.group(1)) // 1_000_000 if c else None,
        "throttled": vcgencmd("get_throttled").partition("=")[2] or None,
    }


# -- ROS side ------------------------------------------------------------------

def import_msg(type_name):
    """'sensor_msgs/msg/Image' -> the message class."""
    pkg, _, name = type_name.replace("/msg/", "/").partition("/")
    module = __import__(f"{pkg}.msg", fromlist=[name])
    return getattr(module, name)


class Probe(Node):
    def __init__(self, camera, extra_topics=()):
        super().__init__("wojtek_perf_probe")
        self.arrivals = {}
        self.latency_ms = {}
        # Extra topics are taken raw (serialized bytes, never deserialized):
        # rate, gaps and size of anything -- depth images, point clouds --
        # for the price of a copy.
        self.raw_bytes = {}
        builtin = set(TOPICS) | ({CAMERA_TOPIC} if camera else set())
        for spec in extra_topics:
            name, _, type_name = spec.partition("=")
            if name in builtin:
                # Already timed below; a second subscription would count
                # every message twice.
                continue
            self.arrivals[name] = []
            self.latency_ms[name] = []
            self.raw_bytes[name] = []
            self.create_subscription(
                import_msg(type_name), name,
                lambda data, n=name: self._on_raw(n, data),
                qos_profile_sensor_data, raw=True,
            )
        self.timing = {"inference_ms": [], "period_ms": []}
        self.recording = False
        topics = dict(TOPICS)
        if camera:
            topics[CAMERA_TOPIC] = CompressedImage
        self.camera_bytes = []
        for name, mtype in topics.items():
            self.arrivals[name] = []
            self.latency_ms[name] = []
            self.create_subscription(
                mtype, name, lambda m, n=name: self._on_msg(n, m),
                qos_profile_sensor_data,
            )
        if PolicyTiming is not None:
            self.create_subscription(
                PolicyTiming, "/wojtek/policy_timing", self._on_timing,
                qos_profile_sensor_data,
            )

    def _on_msg(self, name, msg):
        if not self.recording:
            return
        self.arrivals[name].append(time.monotonic())
        header = getattr(msg, "header", None)
        if header is not None and header.stamp.sec:
            now = time.time()
            stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            self.latency_ms[name].append((now - stamp) * 1e3)
        if isinstance(msg, CompressedImage):
            self.camera_bytes.append(len(msg.data))

    def _on_raw(self, name, data):
        if not self.recording:
            return
        self.arrivals[name].append(time.monotonic())
        self.raw_bytes[name].append(len(data))

    def _on_timing(self, msg):
        if not self.recording:
            return
        self.timing["inference_ms"].append(msg.inference_ms)
        if msg.period_ms > 0:
            self.timing["period_ms"].append(msg.period_ms)

    def topic_report(self, duration):
        out = {}
        for name, ts in self.arrivals.items():
            if not ts:
                out[name] = {"hz": 0.0}
                continue
            gaps = [(b - a) * 1e3 for a, b in zip(ts, ts[1:])]
            out[name] = {
                "hz": round(len(ts) / duration, 2),
                "gap_ms": summarize(gaps),
                "latency_ms": summarize(self.latency_ms[name]),
            }
        for name, sizes in self.raw_bytes.items():
            if sizes:
                out[name]["msg_kb"] = round(statistics.fmean(sizes) / 1024, 1)
                out[name]["MBps"] = round(sum(sizes) / duration / 2**20, 2)
        if self.camera_bytes:
            out[CAMERA_TOPIC]["frame_kb"] = round(
                statistics.fmean(self.camera_bytes) / 1024, 1
            )
        return out


def wait_for_topic(node, topic, timeout):
    """Wait until `topic` has a publisher, or give up after `timeout` s."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node.count_publishers(topic) > 0:
            return True
        time.sleep(0.5)
    return False


def start_executor(node):
    """Spin `node` on a background thread; the main thread only samples.

    The events executor, as policy_node uses: the default executor's per-
    wake overhead in rclpy costs a large share of a Pi core at these
    message rates, and the probe's own load belongs out of the picture.
    """
    try:
        from rclpy.experimental import EventsExecutor
        executor = EventsExecutor()
    except ImportError:
        executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    return executor


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--camera", action="store_true",
                    help="also subscribe to the compressed colour stream "
                         "(makes the probe a viewer)")
    ap.add_argument("--topic", action="append", default=[], metavar="NAME=TYPE",
                    help="extra topic to time, e.g. /camera/camera/depth/"
                         "image_rect_raw=sensor_msgs/msg/Image (repeatable)")
    ap.add_argument("--wait-for", default="/wojtek/joint_targets",
                    help="topic that must have a publisher before measuring")
    ap.add_argument("--wait-timeout", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds to let the stack settle before measuring")
    args = ap.parse_args()
    for spec in args.topic:
        name, _, type_name = spec.partition("=")
        if not name.startswith("/") or "/" not in type_name:
            ap.error(f"--topic wants /name=pkg/msg/Type, got '{spec}'")
        try:
            import_msg(type_name)
        except (ImportError, AttributeError, ValueError) as e:
            ap.error(f"--topic {spec}: cannot import {type_name} ({e})")

    rclpy.init()
    node = Probe(args.camera, args.topic)
    executor = start_executor(node)
    ready = wait_for_topic(node, args.wait_for, args.wait_timeout) if args.wait_for else True
    time.sleep(args.settle)

    cpu_samples = {}
    stop = threading.Event()

    def sample_cpu():
        prev = read_cpu_times()
        while not stop.wait(0.5):
            cur = read_cpu_times()
            for core, (busy, total) in cur.items():
                db, dt = busy - prev[core][0], total - prev[core][1]
                if dt > 0:
                    cpu_samples.setdefault(core, []).append(100.0 * db / dt)
            prev = cur

    board_start = board_state()
    procs0 = snapshot_processes()
    irq0 = read_interrupts()
    net0 = read_net()
    self_t0 = os.times()
    t0 = time.monotonic()
    wall0 = time.time()
    sampler = threading.Thread(target=sample_cpu, daemon=True)
    sampler.start()

    node.recording = True
    temps = [board_start["temp_c"]]
    next_temp = time.monotonic() + 5.0
    while time.monotonic() - t0 < args.duration:
        time.sleep(min(0.25, max(0.0, args.duration - (time.monotonic() - t0))))
        if time.monotonic() >= next_temp:
            temps.append(board_state()["temp_c"])
            next_temp += 5.0
    node.recording = False

    elapsed = time.monotonic() - t0
    stop.set()
    sampler.join()
    procs1 = snapshot_processes()
    irq1 = read_interrupts()
    net1 = read_net()
    self_t1 = os.times()
    board_end = board_state()

    # Per-process CPU over the window, % of ONE core.
    scale = 100.0 / (elapsed * CLK_TCK)
    processes = []
    for pid, p1 in procs1.items():
        p0 = procs0.get(pid)
        if p0 is None:
            continue
        cpu = (p1["ticks"] - p0["ticks"]) * scale
        entry = {"pid": pid, "comm": p1["comm"], "cmd": p1["cmd"],
                 "cpu_pct": round(cpu, 1)}
        if "threads" in p1:
            entry["cpus_allowed"] = p1["cpus_allowed"]
            entry["rss"] = p1["rss"]
            threads = []
            for tid, t1 in p1["threads"].items():
                t0_ = p0.get("threads", {}).get(tid)
                if t0_ is None:
                    continue
                threads.append({
                    "tid": tid,
                    "comm": t1["comm"],
                    "cpu_pct": round((t1["ticks"] - t0_["ticks"]) * scale, 1),
                    "core": t1["processor"],
                    "sched": {0: "OTHER", 1: "FIFO", 2: "RR"}.get(t1["policy"], t1["policy"]),
                    "rt_prio": t1["rt_priority"],
                    "nvcsw": t1["nvcsw"] - t0_["nvcsw"],
                })
            threads.sort(key=lambda t: -t["cpu_pct"])
            entry["threads"] = [t for t in threads if t["cpu_pct"] > 0.1 or t["sched"] != "OTHER"]
        if cpu >= 0.5 or "threads" in entry:
            processes.append(entry)
    processes.sort(key=lambda p: -p["cpu_pct"])

    irqs = []
    for label, c1 in irq1.items():
        c0 = irq0.get(label, [0] * len(c1))
        delta = [b - a for a, b in zip(c0, c1)]
        if sum(delta):
            irqs.append({"irq": label[:60], "per_s": [round(d / elapsed, 1) for d in delta]})
    irqs.sort(key=lambda i: -sum(i["per_s"]))

    net = {}
    for iface in ("eth0", "wlan0", "lo"):
        if iface in net0 and iface in net1:
            net[iface] = {
                "rx_kBps": round((net1[iface][0] - net0[iface][0]) / elapsed / 1024, 1),
                "tx_kBps": round((net1[iface][1] - net0[iface][1]) / elapsed / 1024, 1),
            }

    probe_cpu = ((self_t1.user - self_t0.user) + (self_t1.system - self_t0.system)) / elapsed * 100

    report = {
        "label": args.label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall0)),
        "duration_s": round(elapsed, 1),
        "stack_ready": ready,
        "cpu": {
            core: {"mean": round(statistics.fmean(v), 1), "p95": round(pct(v, 95), 1),
                   "max": round(max(v), 1)}
            for core, v in sorted(cpu_samples.items())
        },
        "policy": {k: summarize(v) for k, v in node.timing.items()},
        "topics": node.topic_report(elapsed),
        "board": {
            "start": board_start,
            "end": board_end,
            "temp_max_c": max(t for t in temps + [board_end["temp_c"]] if t is not None)
            if any(t is not None for t in temps) else None,
            "mem": read_meminfo(),
            "loadavg": open("/proc/loadavg").read().split()[:3],
        },
        "net": net,
        "irqs": irqs[:15],
        "processes": processes[:25],
        "probe": {"cpu_pct": round(probe_cpu, 1)},
    }
    print(json.dumps(report, indent=1))
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
