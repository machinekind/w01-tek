#!/usr/bin/env python3
"""The report measures latency at each stage using one robot bag.

    python3 latency_report.py <bag dir> [--json out.json] [--stale-s 0.02]

The script runs on a PC with the `rosbags` Python package. ROS is optional.
The robot records the bag with `telemetry:=true bag:=true`.

Measurements use header stamps and timing fields from the robot's nodes.
These values use the Pi's clock. Recorder receive times include delivery
and scheduling delays at the recorder. The report excludes those times.

Each stage has a sample count, p50, p95, p99 and maximum. Latencies are in
milliseconds. Policy timing covers joints_age, joints_wait, joints_delivery,
imu_age, inference and period. joints_delivery equals joints_age minus
joints_wait. IO timing covers target_age and state_relay for samples with
command_sent set to true.

The pipeline equals joints_age + inference + target_age for one policy tick.
A match requires policy_timing.header.stamp to equal io_timing.target_stamp
to the nanosecond. Only samples with command_sent set to true contribute.

The response_age summary gives each joint's maximum age. It also counts
samples above --stale-s. Diagnostics give hardware read and write execution
times. They include component periodicity where the manager reports it.
They also include the manager's own update periodicity. Unknown diagnostic
values appear as null in JSON and "n/a" in the table.

Each topic has a message count and a mean rate from its header stamps.
The notes count NaN ages excluded from the stage statistics. Missing stamps
can produce NaN ages. A policy that omits the IMU reports NaN for imu_age.
An older IoTiming layout omits command_sent. Its IO stages and pipeline
remain empty. Its sample counts and armed counts remain available.

The reader uses message definitions recorded in the bag. Bags without
definitions use rosbags' Jazzy store and the control_msgs definitions below.
They also use the wojtek_telemetry .msg files in ros/src/wojtek_telemetry/msg.
The --msg-dir option selects another directory for those files.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

TOPIC_POLICY = "/wojtek/policy_timing"
TOPIC_IO = "/wojtek/io_timing"
TOPIC_DJS = "/dynamic_joint_states"
TOPIC_DIAG = "/diagnostics"
TOPICS = (TOPIC_POLICY, TOPIC_IO, TOPIC_DJS, TOPIC_DIAG)

RESPONSE_AGE = "response_age"

# controller_manager 4.48.1 prefixes each status name with the node name.
STATUS_HW = "Hardware Components Activity"
STATUS_CM = "Controller Manager Activity"

DEFAULT_MSG_DIR = Path(__file__).resolve().parents[2] / "src/wojtek_telemetry/msg"

# Bags without control_msgs definitions need these types alongside the Jazzy store.
CONTROL_MSGS = {
    "control_msgs/msg/InterfaceValue": "string[] interface_names\nfloat64[] values\n",
    "control_msgs/msg/DynamicJointState": (
        "std_msgs/Header header\nstring[] joint_names\nInterfaceValue[] interface_values\n"
    ),
}

# controller_manager reports "Avg: 40.67 [6.25 - 4716.21] us, StdDev: 163.78".
# Periodicity values also include " -> Desired : 50 Hz".
# NaN values can carry a sign, as in "-nan".
_NUM = r"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
_STATS_RE = re.compile(
    rf"Avg:\s*(?P<avg>{_NUM})\s*\[\s*(?P<min>{_NUM})\s*-\s*(?P<max>{_NUM})\s*\]\s*"
    rf"(?P<unit>\S+?)\s*,\s*StdDev:\s*(?P<std>{_NUM})"
    rf"(?:\s*->\s*Desired\s*:\s*(?P<desired>{_NUM})\s*Hz)?",
    re.IGNORECASE,
)

_EXTRA_HINT = "latency_report.py needs the 'rosbags' package: pip install rosbags"


def telemetry_types(msg_dir):
    """The result contains wojtek_telemetry types from the .msg files in msg_dir."""
    from rosbags.typesys import get_types_from_msg

    types = {}
    for path in sorted(Path(msg_dir).glob("*.msg")):
        types.update(get_types_from_msg(path.read_text(), f"wojtek_telemetry/msg/{path.stem}"))
    return types


def control_msgs_types():
    from rosbags.typesys import get_types_from_msg

    types = {}
    for name, text in CONTROL_MSGS.items():
        types.update(get_types_from_msg(text, name))
    return types


def build_typestore(msg_dir=DEFAULT_MSG_DIR):
    """The Jazzy store includes control_msgs and local wojtek_telemetry types."""
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    store.register(control_msgs_types())
    if Path(msg_dir).is_dir():
        store.register(telemetry_types(msg_dir))
    return store


def stamp_ns(t):
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


def percentile(sorted_vals, q):
    """The percentile uses linear interpolation between adjacent ranks."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(values):
    """Finite values yield n, p50, p95, p99 and max. An empty set yields None."""
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    return {
        "n": len(vals),
        "p50_ms": percentile(vals, 50),
        "p95_ms": percentile(vals, 95),
        "p99_ms": percentile(vals, 99),
        "max_ms": vals[-1],
    }


def topic_rate(stamps):
    if not stamps:
        return {"count": 0, "span_s": None, "rate_hz": None}
    span = (max(stamps) - min(stamps)) * 1e-9
    rate = (len(stamps) - 1) / span if span > 0 else None
    return {"count": len(stamps), "span_s": span, "rate_hz": rate}


def parse_stats(value):
    """Recognized controller_manager statistics yield a dict. Other text yields None."""
    m = _STATS_RE.search(value)
    if not m:
        return None
    out = {k: _finite(float(m.group(k))) for k in ("avg", "min", "max", "std")}
    out["unit"] = m.group("unit")
    if m.group("desired") is not None:
        out["desired_hz"] = _finite(float(m.group("desired")))
    return out


def _finite(v):
    """Finite values pass through. NaN and infinity become None for JSON."""
    return v if v is not None and math.isfinite(v) else None


def _max_known(prev, v):
    """The running maximum ignores None on either side."""
    if prev is None:
        return v
    if v is None:
        return prev
    return max(prev, v)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_topics(bag_path, msg_dir=DEFAULT_MSG_DIR):
    """Each report topic maps to its messages. The result also lists all bag topics."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError(_EXTRA_HINT) from e

    fallback = build_typestore(msg_dir)
    rows = {t: [] for t in TOPICS}
    with AnyReader([Path(bag_path)], default_typestore=fallback) as reader:
        conns = [c for c in reader.connections if c.topic in rows]
        # Some bags carry definitions for only part of their connections.
        # The fallback store supplies the missing types.
        missing = {c.msgtype for c in conns if c.msgtype not in reader.typestore.fielddefs}
        if missing:
            reader.typestore.register(
                {k: v for k, v in fallback.fielddefs.items() if k not in reader.typestore.fielddefs}
            )
        # An empty connection list means "all connections" to rosbags.
        for conn, _t_recv, raw in reader.messages(connections=conns) if conns else ():
            # Recorder receive times include delays outside the robot's path.
            rows[conn.topic].append(reader.deserialize(raw, conn.msgtype))
        topics_in_bag = sorted({c.topic for c in reader.connections})
    return rows, topics_in_bag


def analyze_policy(msgs, notes):
    fields = ("joints_age_ms", "joints_wait_ms", "imu_age_ms", "inference_ms", "period_ms")
    series = {f: [] for f in fields}
    delivery = []
    absent = set()
    for m in msgs:
        for f in fields:
            v = getattr(m, f, None)
            if v is None:
                absent.add(f)
            else:
                series[f].append(float(v))
        age, wait = getattr(m, "joints_age_ms", None), getattr(m, "joints_wait_ms", None)
        if age is not None and wait is not None:
            delivery.append(float(age) - float(wait))
    if absent:
        notes.append(
            f"{TOPIC_POLICY}: fields {sorted(absent)} not in the recorded message "
            "(older stack); those stages are empty"
        )
    for f in ("joints_age_ms", "joints_wait_ms", "imu_age_ms"):
        vals = series[f]
        nan = sum(1 for v in vals if math.isnan(v))
        if not nan:
            continue
        if nan == len(vals) and f == "imu_age_ms":
            notes.append(f"{TOPIC_POLICY}: imu_age_ms is NaN in every sample (policy does not observe the IMU)")
        else:
            notes.append(f"{TOPIC_POLICY}: {f} NaN in {nan} of {len(vals)} samples, dropped")
    stages = {f[:-3]: summarize(series[f]) for f in fields}
    stages["joints_delivery"] = summarize(delivery)
    order = ("joints_age", "joints_wait", "joints_delivery", "imu_age", "inference", "period")
    return {k: stages[k] for k in order}


def analyze_io(msgs, notes):
    """IO stages include samples that sent a motor command.

    armed records the node's arm state when the target arrived.
    command_sent is false during dry_run. It is also false for targets with
    non-finite values or missing joints. Early IoTiming messages omit
    command_sent. Their IO stages remain empty. Their io_sent result is None.
    """
    known = all(hasattr(m, "command_sent") for m in msgs)
    armed = sum(1 for m in msgs if bool(m.armed))
    disarmed = len(msgs) - armed
    if not known:
        notes.append(
            f"{TOPIC_IO}: no command_sent field (early IoTiming); command delivery unknown, "
            "target_age, state_relay and pipeline not reported"
        )
        stages = {"target_age": None, "state_relay": None}
        counts = {
            "total": len(msgs),
            "armed": armed,
            "disarmed": disarmed,
            "command_sent_known": False,
            "command_sent": None,
            "armed_not_sent": None,
        }
        return stages, counts, None
    sent = [m for m in msgs if bool(m.command_sent)]
    armed_not_sent = sum(1 for m in msgs if bool(m.armed) and not bool(m.command_sent))
    if msgs and not sent:
        notes.append(f"{TOPIC_IO}: no sample sent a command; io stages and pipeline are empty")
    if disarmed:
        notes.append(f"{TOPIC_IO}: {disarmed} disarmed samples left out of the io stages")
    if armed_not_sent:
        notes.append(
            f"{TOPIC_IO}: {armed_not_sent} armed samples sent no command "
            "(dry_run, non-finite or rejected target), left out"
        )
    stages = {
        "target_age": summarize(float(m.target_age_ms) for m in sent),
        "state_relay": summarize(float(m.state_relay_ms) for m in sent),
    }
    counts = {
        "total": len(msgs),
        "armed": armed,
        "disarmed": disarmed,
        "command_sent_known": True,
        "command_sent": len(sent),
        "armed_not_sent": armed_not_sent,
    }
    return stages, counts, sent


def analyze_pipeline(policy_msgs, io_all, io_sent):
    by_stamp = {}
    for m in policy_msgs:
        by_stamp[stamp_ns(m.header.stamp)] = m
    io_stamps = {stamp_ns(m.target_stamp) for m in io_all}
    policy_without_io = sum(1 for s in by_stamp if s not in io_stamps)
    if io_sent is None:
        # The bag leaves command delivery unknown.
        return None, {"joined": None, "io_sent_unmatched": None, "policy_without_io": policy_without_io}
    values = []
    joined = 0
    for io in io_sent:
        pol = by_stamp.get(stamp_ns(io.target_stamp))
        if pol is None:
            continue
        joined += 1
        age = getattr(pol, "joints_age_ms", None)
        if age is None:
            continue
        values.append(float(age) + float(pol.inference_ms) + float(io.target_age_ms))
    join = {
        "joined": joined,
        "io_sent_unmatched": len(io_sent) - joined,
        "policy_without_io": policy_without_io,
    }
    return summarize(values), join


def analyze_response_age(msgs, stale_s, notes):
    per_joint = {}
    for m in msgs:
        for name, iv in zip(m.joint_names, m.interface_values):
            names = list(iv.interface_names)
            if RESPONSE_AGE in names:
                per_joint.setdefault(name, []).append(float(iv.values[names.index(RESPONSE_AGE)]))
    if not msgs:
        return {"available": False, "joints": {}}
    if not per_joint:
        notes.append(
            f"{TOPIC_DJS}: no '{RESPONSE_AGE}' interface (stack older than the drive-age "
            "interface); response age not reported"
        )
        return {"available": False, "joints": {}}
    joints = {}
    for name in sorted(per_joint):
        vals = [v for v in per_joint[name] if math.isfinite(v)]
        over = sum(1 for v in vals if v > stale_s)
        joints[name] = {
            "n": len(vals),
            "max_ms": max(vals) * 1e3 if vals else None,
            "over": over,
            "fraction": over / len(vals) if vals else None,
        }
    return {"available": True, "joints": joints}


def _status_suffix(status, suffix):
    return status.name == suffix or status.name.endswith(": " + suffix)


def analyze_diagnostics(msgs, notes):
    """The report includes hardware and update statistics from controller_manager.

    controller_manager 4.48.1 accumulates statistics from startup.
    A capture showed that minimum and maximum values persisted across messages.
    The report gives each hardware metric's last value and worst average in the bag.
    Manager metrics have a last value and a maximum.
    """
    hw = {}
    hw_levels = {}
    cm = {}
    seen_hw = seen_cm = False
    for m in msgs:
        for st in m.status:
            if _status_suffix(st, STATUS_HW):
                seen_hw = True
                lvl = int(st.level[0]) if isinstance(st.level, (bytes, bytearray)) else int(st.level)
                hw_levels[lvl] = hw_levels.get(lvl, 0) + 1
                for kv in st.values:
                    comp, sep, metric = kv.key.partition(".")
                    if not sep:
                        continue
                    entry = hw.setdefault(comp, {})
                    stats = parse_stats(kv.value)
                    if stats is None:
                        entry[metric] = {"last": kv.value}
                        continue
                    prev = entry.get(metric)
                    prev_worst = prev.get("worst_avg") if isinstance(prev, dict) else None
                    worst = _max_known(prev_worst, stats["avg"])
                    entry[metric] = {"last": stats, "worst_avg": worst}
            elif _status_suffix(st, STATUS_CM):
                seen_cm = True
                for kv in st.values:
                    v = _to_float(kv.value)
                    if v is None:
                        continue
                    v = _finite(v)  # The manager reports NaN before its first measurement.
                    prev = cm.get(kv.key)
                    cm[kv.key] = {"last": v, "max": _max_known(prev["max"] if prev else None, v)}
    if msgs and not seen_hw:
        notes.append(f"{TOPIC_DIAG}: no '{STATUS_HW}' status; hardware component statistics not reported")
    if msgs and not seen_cm:
        notes.append(f"{TOPIC_DIAG}: no '{STATUS_CM}' status")
    levels = {name: hw_levels.get(i, 0) for i, name in enumerate(("ok", "warn", "error", "stale"))}
    return {"hardware_components": hw, "hardware_levels": levels, "controller_manager": cm}


def analyze(rows, stale_s=0.02, topics_in_bag=()):
    notes = []
    for t in TOPICS:
        if not rows[t]:
            notes.append(f"{t}: not in the bag" if t not in topics_in_bag else f"{t}: no messages")
    policy = rows[TOPIC_POLICY]
    io = rows[TOPIC_IO]
    stages = analyze_policy(policy, notes)
    io_stages, io_counts, io_sent = analyze_io(io, notes)
    stages.update(io_stages)
    stages["pipeline"], join = analyze_pipeline(policy, io, io_sent)
    if io_sent is None and io:
        notes.append("pipeline: not reported, io_timing has no command_sent")
    if join["io_sent_unmatched"]:
        notes.append(
            f"pipeline: {join['io_sent_unmatched']} command_sent io samples had no policy_timing "
            "with the same stamp"
        )
    topics = {t: topic_rate([stamp_ns(m.header.stamp) for m in rows[t]]) for t in TOPICS}
    return {
        "stale_s": stale_s,
        "topics": topics,
        "stages": stages,
        "io_timing": io_counts,
        "pipeline_join": join,
        "response_age": analyze_response_age(rows[TOPIC_DJS], stale_s, notes),
        "diagnostics": analyze_diagnostics(rows[TOPIC_DIAG], notes),
        "notes": notes,
    }


def report(bag_path, stale_s=0.02, msg_dir=DEFAULT_MSG_DIR):
    rows, topics_in_bag = read_topics(bag_path, msg_dir)
    out = analyze(rows, stale_s, topics_in_bag)
    out["bag"] = str(bag_path)
    return out


def _f(v, digits=2):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _g(v):
    """Known diagnostic values appear as numbers. Unknown values appear as n/a."""
    return "n/a" if v is None else f"{v:g}"


def _n(v):
    return "unknown" if v is None else str(v)


def format_report(r):
    lines = [f"bag: {r.get('bag', '-')}", ""]
    lines.append(f"{'stage':<18}{'n':>8}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   [ms]")
    for name, s in r["stages"].items():
        if s is None:
            lines.append(f"{name:<18}{0:>8}{'-':>9}{'-':>9}{'-':>9}{'-':>9}")
        else:
            lines.append(
                f"{name:<18}{s['n']:>8}{_f(s['p50_ms']):>9}{_f(s['p95_ms']):>9}"
                f"{_f(s['p99_ms']):>9}{_f(s['max_ms']):>9}"
            )
    io, j = r["io_timing"], r["pipeline_join"]
    if io["command_sent_known"]:
        lines.append(
            f"io_timing: {io['total']} samples, {io['command_sent']} command_sent, "
            f"{io['disarmed']} armed=false, {io['armed_not_sent']} armed but no command"
        )
    else:
        lines.append(
            f"io_timing: {io['total']} samples, {io['armed']} armed, {io['disarmed']} armed=false, "
            "command delivery unknown (no command_sent field)"
        )
    lines.append(
        f"pipeline join: {_n(j['joined'])} joined, {_n(j['io_sent_unmatched'])} command_sent io "
        f"unmatched, {j['policy_without_io']} policy ticks without io"
    )

    ra = r["response_age"]
    lines.append("")
    lines.append(f"response_age (threshold {r['stale_s'] * 1e3:.0f} ms)")
    if ra["available"]:
        lines.append(f"  {'joint':<28}{'n':>8}{'max ms':>10}{'over':>8}{'frac':>9}")
        for name, s in ra["joints"].items():
            frac = "-" if s["fraction"] is None else f"{s['fraction'] * 100:.2f}%"
            lines.append(f"  {name:<28}{s['n']:>8}{_f(s['max_ms'], 1):>10}{s['over']:>8}{frac:>9}")
    else:
        lines.append("  not available")

    d = r["diagnostics"]
    lines.append("")
    lines.append("hardware components (controller_manager, since its start: last / worst avg)")
    if d["hardware_components"]:
        for comp, metrics in sorted(d["hardware_components"].items()):
            for metric, v in sorted(metrics.items()):
                last = v["last"]
                if isinstance(last, dict):
                    desired = f" desired {_g(last['desired_hz'])} Hz" if "desired_hz" in last else ""
                    lines.append(
                        f"  {comp}.{metric}: avg {_g(last['avg'])} [{_g(last['min'])} - {_g(last['max'])}] "
                        f"{last['unit']}, std {_g(last['std'])}{desired}; worst avg {_g(v['worst_avg'])}"
                    )
                else:
                    lines.append(f"  {comp}.{metric}: {last}")
        lv = d["hardware_levels"]
        lines.append(
            f"  status levels: {lv['ok']} ok, {lv['warn']} warn, {lv['error']} error, {lv['stale']} stale"
        )
    else:
        lines.append("  not available")
    if d["controller_manager"]:
        cm = d["controller_manager"]
        parts = [f"{k} {_g(v['last'])}" for k, v in cm.items()]
        lines.append("controller manager (last): " + ", ".join(parts))

    lines.append("")
    lines.append(f"{'topic':<26}{'count':>8}{'rate Hz':>10}{'span s':>9}   (header stamps)")
    for t, s in r["topics"].items():
        lines.append(f"{t:<26}{s['count']:>8}{_f(s['rate_hz'], 1):>10}{_f(s['span_s'], 2):>9}")

    if r["notes"]:
        lines.append("")
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in r["notes"])
    return "\n".join(lines)


def _positive_float(text):
    v = float(text)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("bag", type=Path, help="rosbag2 directory (sqlite3 or mcap)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON here")
    ap.add_argument(
        "--stale-s",
        type=_positive_float,
        default=0.02,
        help="response_age threshold in seconds (default 0.02: two 100 Hz controller periods)",
    )
    ap.add_argument(
        "--msg-dir",
        type=Path,
        default=DEFAULT_MSG_DIR,
        help="wojtek_telemetry .msg files for bags without embedded definitions",
    )
    args = ap.parse_args(argv)
    r = report(args.bag, args.stale_s, args.msg_dir)
    print(format_report(r))
    if args.json:
        args.json.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
