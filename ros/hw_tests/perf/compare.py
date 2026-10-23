#!/usr/bin/env python3
"""Summarize one probe result, or line several up side by side.

    python3 compare.py results/A.json                 # one run
    python3 compare.py results/A.json results/B.json  # before / after

The rows are the numbers that decide whether the robot "lags": control-loop
and policy rates, their worst gaps, policy inference time, the busiest
core, the camera as the viewer saw it, temperature and throttling.
"""

import argparse
import json
import os
import sys


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return None
        d = d[p]
    return d


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def proc_cpu(report, needle):
    total = 0.0
    found = False
    for p in report.get("processes", []):
        if needle in p.get("cmd", "") or needle in p.get("comm", ""):
            total += p["cpu_pct"]
            found = True
    return round(total, 1) if found else None


ROWS = [
    ("joint_states Hz", lambda r: get(r, "topics", "/joint_states", "hz")),
    ("joint_states max gap ms", lambda r: get(r, "topics", "/joint_states", "gap_ms", "max")),
    ("joint_targets Hz (policy)", lambda r: get(r, "topics", "/wojtek/joint_targets", "hz")),
    ("joint_targets p99 gap ms", lambda r: get(r, "topics", "/wojtek/joint_targets", "gap_ms", "p99")),
    ("joint_targets max gap ms", lambda r: get(r, "topics", "/wojtek/joint_targets", "gap_ms", "max")),
    ("policy period p99 ms", lambda r: get(r, "policy", "period_ms", "p99")),
    ("policy inference mean ms", lambda r: get(r, "policy", "inference_ms", "mean")),
    ("policy inference p99 ms", lambda r: get(r, "policy", "inference_ms", "p99")),
    ("joint_states->probe latency p99 ms",
     lambda r: get(r, "topics", "/joint_states", "latency_ms", "p99")),
    ("cmd_vel Hz", lambda r: get(r, "topics", "/cmd_vel", "hz")),
    ("cpu0 mean %", lambda r: get(r, "cpu", "cpu0", "mean")),
    ("cpu1 mean %", lambda r: get(r, "cpu", "cpu1", "mean")),
    ("cpu2 mean % (policy)", lambda r: get(r, "cpu", "cpu2", "mean")),
    ("cpu3 mean % (control)", lambda r: get(r, "cpu", "cpu3", "mean")),
    ("cpu2 max %", lambda r: get(r, "cpu", "cpu2", "max")),
    ("cpu3 max %", lambda r: get(r, "cpu", "cpu3", "max")),
    ("ros2_control_node %", lambda r: proc_cpu(r, "ros2_control_node")),
    ("policy_node %", lambda r: proc_cpu(r, "policy_node")),
    ("real_io_node %", lambda r: proc_cpu(r, "real_io_node")),
    ("realsense %", lambda r: proc_cpu(r, "realsense2_camera")),
    ("deck_gateway %", lambda r: proc_cpu(r, "deck_gateway")),
    ("depth Hz", lambda r: get(r, "topics", "/camera/camera/depth/image_rect_raw", "hz")),
    ("depth max gap ms",
     lambda r: get(r, "topics", "/camera/camera/depth/image_rect_raw", "gap_ms", "max")),
    ("depth MB/s (raw)",
     lambda r: get(r, "topics", "/camera/camera/depth/image_rect_raw", "MBps")),
    ("viewer fps", lambda r: get(r, "viewer", "fps")),
    ("viewer max gap ms", lambda r: get(r, "viewer", "gap_ms", "max")),
    ("viewer kB/s", lambda r: get(r, "viewer", "kBps")),
    ("eth0 tx kB/s", lambda r: get(r, "net", "eth0", "tx_kBps")),
    ("temp max C", lambda r: get(r, "board", "temp_max_c")),
    ("throttled", lambda r: get(r, "board", "end", "throttled")),
    ("mem available MB", lambda r: get(r, "board", "mem", "available_mb")),
    ("overrun log lines", lambda r: get(r, "log", "overrun_lines")),
    ("probe own cpu %", lambda r: get(r, "probe", "cpu_pct")),
]


def isolated_irqs(report):
    """Interrupts landing on the isolated cores 2 and 3, per second."""
    out = []
    for i in report.get("irqs", []):
        per = i["per_s"]
        if len(per) >= 4 and (per[2] > 1 or per[3] > 1):
            out.append(f"{i['irq']}: {per[2]:g}/s on cpu2, {per[3]:g}/s on cpu3")
    return out


def column_name(path):
    """The file name carries the stamp and the label (perf.sh names results
    <stamp>-<label>.json), which tells two runs of one scenario apart."""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n\n")[1:]),
    )
    ap.add_argument("paths", nargs="+", metavar="RESULT.json")
    args = ap.parse_args()
    reports = []
    for p in args.paths:
        try:
            with open(p) as f:
                reports.append(json.load(f))
        except (OSError, ValueError) as e:
            ap.error(f"{p}: {e}")
    names = [column_name(p) for p in args.paths]
    width = max(len(r[0]) for r in ROWS) + 2
    col = max(12, *(len(n) + 2 for n in names))
    print("".ljust(width) + "".join(n.rjust(col) for n in names))
    for title, fn in ROWS:
        vals = [fmt(fn(r)) for r in reports]
        if all(v == "-" for v in vals):
            continue
        print(title.ljust(width) + "".join(v.rjust(col) for v in vals))
    for name, r in zip(names, reports):
        irqs = isolated_irqs(r)
        warns = get(r, "log", "warnings") or []
        viewer_error = get(r, "viewer", "error")
        if irqs or warns or viewer_error or not r.get("stack_ready", True):
            print(f"\n[{name}]")
            if not r.get("stack_ready", True):
                print("  ! stack was not ready when the window started")
            if viewer_error:
                print(f"  ! viewer: {viewer_error}")
            for line in irqs:
                print(f"  irq on isolated core  {line}")
            for w in warns[:6]:
                print(f"  log  {w[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
