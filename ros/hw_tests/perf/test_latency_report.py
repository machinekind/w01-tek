"""The tests check latency_report.py with synthetic bags of known stamps.

The suite needs rosbags and pytest. rosbags supplies numpy. ROS is optional.

    python3 -m pytest ros/hw_tests/perf/test_latency_report.py

The wojtek_telemetry definitions below follow the P0 contract.
These tests keep their message layouts independent of the checkout's .msg files.
"""

import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("rosbags")

import numpy as np  # noqa: E402

from rosbags.rosbag2 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latency_report as lr  # noqa: E402

POLICY_TIMING = """std_msgs/Header header
float32 inference_ms
float32 period_ms
float32 joints_age_ms
float32 joints_wait_ms
float32 imu_age_ms
"""

IO_TIMING = """std_msgs/Header header
builtin_interfaces/Time target_stamp
float32 target_age_ms
float32 state_relay_ms
bool armed
bool command_sent
"""

# The first IoTiming layout omits command_sent.
IO_TIMING_NO_SENT = """std_msgs/Header header
builtin_interfaces/Time target_stamp
float32 target_age_ms
float32 state_relay_ms
bool armed
"""

T0 = 1_794_000_000 * 10**9  # The epoch time is arbitrary.
MS = 1_000_000


def make_store(io_timing=IO_TIMING):
    store = get_typestore(Stores.ROS2_JAZZY)
    store.register(get_types_from_msg(POLICY_TIMING, "wojtek_telemetry/msg/PolicyTiming"))
    store.register(get_types_from_msg(io_timing, "wojtek_telemetry/msg/IoTiming"))
    store.register(lr.control_msgs_types())
    return store


class Bag:
    """The bag writer accepts messages by type name."""

    def __init__(self, store, path):
        self.store = store
        self.writer = Writer(path, version=9)
        self.conns = {}

    def __enter__(self):
        self.writer.open()
        return self

    def __exit__(self, *exc):
        self.writer.close()

    def t(self, ns):
        return self.store.types["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9)

    def header(self, ns):
        return self.store.types["std_msgs/msg/Header"](stamp=self.t(ns), frame_id="")

    def msg(self, typ, **kw):
        return self.store.types[typ](**kw)

    def write(self, topic, typ, msg, recv_ns):
        if topic not in self.conns:
            self.conns[topic] = self.writer.add_connection(topic, typ, typestore=self.store)
        self.writer.write(self.conns[topic], recv_ns, self.store.serialize_cdr(msg, typ))


PT = "wojtek_telemetry/msg/PolicyTiming"
IO = "wojtek_telemetry/msg/IoTiming"
DJS = "control_msgs/msg/DynamicJointState"
IV = "control_msgs/msg/InterfaceValue"
DA = "diagnostic_msgs/msg/DiagnosticArray"
DS = "diagnostic_msgs/msg/DiagnosticStatus"
KV = "diagnostic_msgs/msg/KeyValue"

# Recorder receive times differ from header stamps by several seconds.
# Measurements based on receive times would give different rates and ages.
RECV_SKEW = 5 * 10**9


def write_policy(b):
    """Six ticks are 20 ms apart. Tick 5 has NaN ages for a missing stamp."""
    for k in range(6):
        stamp = T0 + k * 20 * MS
        nan = float("nan")
        m = b.msg(
            PT,
            header=b.header(stamp),
            inference_ms=3.0,
            period_ms=20.0 + k,
            joints_age_ms=nan if k == 5 else 10.0 + k,
            joints_wait_ms=nan if k == 5 else 2.0,
            imu_age_ms=nan if k in (0, 5) else 5.0,
        )
        b.write(lr.TOPIC_POLICY, PT, m, stamp + RECV_SKEW + k * 7 * MS)


def write_io(b, with_command_sent=True):
    """Tick 0 is disarmed. Tick 1 is armed with no command sent.

    Ticks 2 through 4 send commands. One extra command has an unmatched
    target stamp. The older layout omits command_sent from the same samples.
    """
    rows = [
        # Each row contains target ns, armed, command_sent and target_age_ms.
        (T0 + 0 * 20 * MS, False, False, 9.0),
        (T0 + 1 * 20 * MS, True, False, 9.0),
        (T0 + 2 * 20 * MS, True, True, 1.0),
        (T0 + 3 * 20 * MS, True, True, 2.0),
        (T0 + 4 * 20 * MS, True, True, 3.0),
        (T0 + 4 * 20 * MS + 1, True, True, 4.0),  # A 1 ns mismatch leaves this sample unmatched.
    ]
    for i, (target, armed, sent, age) in enumerate(rows):
        stamp = target + int(age * MS)
        m = b.msg(
            IO,
            header=b.header(stamp),
            target_stamp=b.t(target),
            target_age_ms=age,
            state_relay_ms=0.5 * (i + 1),
            armed=armed,
            **({"command_sent": sent} if with_command_sent else {}),
        )
        b.write(lr.TOPIC_IO, IO, m, stamp + RECV_SKEW)


def write_djs(b, with_response_age=True):
    a = [0.0, 0.01, 0.03, 0.0]
    c = [0.05, 0.05, 0.0, 0.0]
    for k in range(4):
        stamp = T0 + k * 10 * MS
        names_j = ["position", "velocity", "effort"] + ([lr.RESPONSE_AGE] if with_response_age else [])
        ivs = [
            b.msg(IV, interface_names=names_j, values=np.array([0.0, 0.0, 0.0] + ([a[k]] if with_response_age else []))),
            b.msg(IV, interface_names=names_j, values=np.array([0.0, 0.0, 0.0] + ([c[k]] if with_response_age else []))),
            b.msg(IV, interface_names=["orientation.x"], values=np.array([0.0])),
        ]
        m = b.msg(DJS, header=b.header(stamp), joint_names=["a_joint", "b_joint", "imu"], interface_values=ivs)
        b.write(lr.TOPIC_DJS, DJS, m, stamp + RECV_SKEW)


def _status(b, level, name, values):
    kvs = [b.msg(KV, key=k, value=v) for k, v in values]
    return b.msg(DS, level=level, name=name, message="", hardware_id="ros2_control", values=kvs)


def write_diag(b):
    """Two messages follow the captured controller_manager 4.48.1 format."""
    reads = ["Avg: 40.67 [6.25 - 4716.21] us, StdDev: 163.78", "Avg: 50.43 [6.25 - 13222.83] us, StdDev: 368.76"]
    for k in range(2):
        stamp = T0 + k * 1000 * MS
        hw = _status(
            b,
            1 if k == 0 else 2,
            "controller_manager: Hardware Components Activity",
            [
                ("MD80HardwareInterface.state", "active"),
                ("MD80HardwareInterface.read_cycle.execution_time", reads[k]),
                ("MD80HardwareInterface.write_cycle.execution_time", "Avg: 0.15 [0.00 - 4.83] us, StdDev: 0.17"),
                ("imu.state", "active"),
                ("imu.read_cycle.execution_time", "Avg: 900.5 [800.0 - 2500.0] us, StdDev: 50.0"),
                ("imu.read_cycle.periodicity", "Avg: 99.5 [80.1 - 120.2] Hz, StdDev: 3.3 -> Desired : 100 Hz"),
            ],
        )
        cm = _status(
            b,
            0,
            "controller_manager: Controller Manager Activity",
            [
                ("update_rate", "100"),
                ("periodicity.average", "101.649527" if k == 0 else "99.5"),
                ("periodicity.standard_deviation", "19.983140"),
                ("periodicity.min", "27.860166"),
                ("periodicity.max", "526.523628"),
            ],
        )
        ctl = _status(
            b,
            0,
            "controller_manager: Controllers Activity",
            [("joint_state_broadcaster.state", "active")],
        )
        m = b.msg(DA, header=b.header(stamp), status=[ctl, hw, cm])
        b.write(lr.TOPIC_DIAG, DA, m, stamp + RECV_SKEW)


@pytest.fixture
def full_bag(tmp_path):
    path = tmp_path / "full"
    with Bag(make_store(), path) as b:
        write_policy(b)
        write_io(b)
        write_djs(b)
        write_diag(b)
    return path


def test_policy_stages(full_bag):
    r = lr.report(full_bag)
    s = r["stages"]
    assert s["joints_age"]["n"] == 5
    assert s["joints_age"]["p50_ms"] == pytest.approx(12.0)
    assert s["joints_age"]["p95_ms"] == pytest.approx(13.8)
    assert s["joints_age"]["p99_ms"] == pytest.approx(13.96)
    assert s["joints_age"]["max_ms"] == pytest.approx(14.0)
    assert s["joints_wait"]["n"] == 5
    assert s["joints_delivery"]["p50_ms"] == pytest.approx(10.0)
    assert s["joints_delivery"]["max_ms"] == pytest.approx(12.0)
    assert s["imu_age"]["n"] == 4
    assert s["imu_age"]["max_ms"] == pytest.approx(5.0)
    assert s["inference"]["n"] == 6
    assert s["period"]["p50_ms"] == pytest.approx(22.5)
    assert s["period"]["max_ms"] == pytest.approx(25.0)
    assert any("joints_age_ms NaN in 1 of 6" in n for n in r["notes"])
    assert any("imu_age_ms NaN in 2 of 6" in n for n in r["notes"])


def test_io_stages_use_command_sent(full_bag):
    r = lr.report(full_bag)
    assert r["io_timing"] == {
        "total": 6,
        "armed": 5,
        "disarmed": 1,
        "command_sent_known": True,
        "command_sent": 4,
        "armed_not_sent": 1,
    }
    s = r["stages"]
    # Only samples with command_sent set to true contribute to target_age.
    assert s["target_age"]["n"] == 4
    assert s["target_age"]["p50_ms"] == pytest.approx(2.5)
    assert s["target_age"]["max_ms"] == pytest.approx(4.0)
    assert s["state_relay"]["p50_ms"] == pytest.approx(2.25)
    assert s["state_relay"]["max_ms"] == pytest.approx(3.0)


def test_pipeline_join(full_bag):
    r = lr.report(full_bag)
    p = r["stages"]["pipeline"]
    assert p["n"] == 3
    assert p["p50_ms"] == pytest.approx(18.0)
    assert p["max_ms"] == pytest.approx(20.0)
    assert r["pipeline_join"] == {"joined": 3, "io_sent_unmatched": 1, "policy_without_io": 1}


def test_response_age(full_bag):
    r = lr.report(full_bag)
    ra = r["response_age"]
    assert ra["available"]
    assert set(ra["joints"]) == {"a_joint", "b_joint"}
    assert ra["joints"]["a_joint"] == {"n": 4, "max_ms": pytest.approx(30.0), "over": 1, "fraction": 0.25}
    assert ra["joints"]["b_joint"]["over"] == 2
    assert ra["joints"]["b_joint"]["max_ms"] == pytest.approx(50.0)
    # All samples fall below the higher threshold.
    r2 = lr.report(full_bag, stale_s=0.06)
    assert r2["response_age"]["joints"]["b_joint"]["over"] == 0


def test_diagnostics(full_bag):
    d = lr.report(full_bag)["diagnostics"]
    hw = d["hardware_components"]
    assert set(hw) == {"MD80HardwareInterface", "imu"}
    read = hw["MD80HardwareInterface"]["read_cycle.execution_time"]
    assert read["last"]["avg"] == pytest.approx(50.43)
    assert read["last"]["max"] == pytest.approx(13222.83)
    assert read["last"]["unit"] == "us"
    assert read["worst_avg"] == pytest.approx(50.43)
    assert hw["MD80HardwareInterface"]["state"] == {"last": "active"}
    per = hw["imu"]["read_cycle.periodicity"]["last"]
    assert per["avg"] == pytest.approx(99.5)
    assert per["unit"] == "Hz"
    assert per["desired_hz"] == pytest.approx(100.0)
    assert d["hardware_levels"] == {"ok": 0, "warn": 1, "error": 1, "stale": 0}
    cm = d["controller_manager"]
    assert cm["periodicity.average"] == {"last": pytest.approx(99.5), "max": pytest.approx(101.649527)}


def test_topic_rates_from_header_stamps(full_bag):
    t = lr.report(full_bag)["topics"]
    assert t[lr.TOPIC_POLICY]["count"] == 6
    assert t[lr.TOPIC_POLICY]["rate_hz"] == pytest.approx(50.0)
    assert t[lr.TOPIC_DJS]["rate_hz"] == pytest.approx(100.0)
    assert t[lr.TOPIC_DIAG]["rate_hz"] == pytest.approx(1.0)


def test_cli_table_and_json(full_bag, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert lr.main([str(full_bag), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "pipeline" in text and "MD80HardwareInterface.read_cycle.execution_time" in text
    data = json.loads(out.read_text())
    assert data["stages"]["pipeline"]["n"] == 3


def test_older_stack_without_response_age_or_diagnostics(tmp_path):
    path = tmp_path / "old"
    with Bag(make_store(), path) as b:
        write_policy(b)
        write_djs(b, with_response_age=False)
    r = lr.report(path)
    assert r["response_age"]["available"] is False
    assert any("no 'response_age' interface" in n for n in r["notes"])
    assert any(n.startswith(f"{lr.TOPIC_DIAG}: not in the bag") for n in r["notes"])
    assert any(n.startswith(f"{lr.TOPIC_IO}: not in the bag") for n in r["notes"])
    assert r["stages"]["pipeline"] is None
    assert r["diagnostics"]["hardware_components"] == {}
    lr.format_report(r)  # The formatter accepts missing data.


def strip_definitions(bag_dir):
    """The bag retains its messages without embedded type definitions."""
    for db in Path(bag_dir).glob("*.db3"):
        with sqlite3.connect(db) as con:
            con.execute("DELETE FROM message_definitions")


def test_bag_without_definitions_uses_msg_dir(tmp_path):
    """Bags without embedded definitions get telemetry types from --msg-dir."""
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "PolicyTiming.msg").write_text(POLICY_TIMING)
    (msg_dir / "IoTiming.msg").write_text(IO_TIMING)
    path = tmp_path / "nodefs"
    with Bag(make_store(), path) as b:
        write_policy(b)
        write_io(b)
        write_djs(b)
        write_diag(b)
    strip_definitions(path)
    from rosbags.highlevel import AnyReader, AnyReaderError

    with pytest.raises(AnyReaderError, match="no type definitions"):
        with AnyReader([path]):
            pass
    r = lr.report(path, msg_dir=msg_dir)
    assert r["stages"]["pipeline"]["n"] == 3
    assert math.isclose(r["stages"]["joints_age"]["max_ms"], 14.0)
    assert r["response_age"]["joints"]["a_joint"]["over"] == 1
    assert "MD80HardwareInterface" in r["diagnostics"]["hardware_components"]


def test_default_msg_dir_is_the_checkouts_telemetry_package():
    types = lr.telemetry_types(lr.DEFAULT_MSG_DIR)
    assert "wojtek_telemetry/msg/PolicyTiming" in types


def load_json_strict(path):
    """The JSON reader rejects NaN and Infinity tokens."""

    def reject(token):
        raise ValueError(f"non-finite token {token} in the JSON")

    with open(path) as f:
        return json.load(f, parse_constant=reject)


def write_diag_values(b, hw_reads, cm_averages):
    """Each diagnostics message has a read statistic and a manager average."""
    for k, (read, avg) in enumerate(zip(hw_reads, cm_averages)):
        stamp = T0 + k * 1000 * MS
        hw = _status(
            b,
            0,
            "controller_manager: Hardware Components Activity",
            [
                ("MD80HardwareInterface.read_cycle.execution_time", read),
                ("imu.read_cycle.periodicity", read.replace(" us,", " Hz,") + " -> Desired : 100 Hz"),
            ],
        )
        cm = _status(b, 0, "controller_manager: Controller Manager Activity", [("periodicity.average", avg)])
        m = b.msg(DA, header=b.header(stamp), status=[hw, cm])
        b.write(lr.TOPIC_DIAG, DA, m, stamp + RECV_SKEW)


NAN_STATS = "Avg: nan [nan - -nan] us, StdDev: nan"


def test_diagnostics_nan_then_valid(tmp_path, capsys):
    """controller_manager reports NaN before its first measurement."""
    path = tmp_path / "nan_then_valid"
    with Bag(make_store(), path) as b:
        write_diag_values(
            b,
            [NAN_STATS, "Avg: 40.0 [6.0 - 900.0] us, StdDev: 10.0", "Avg: 30.0 [6.0 - 950.0] us, StdDev: 9.0"],
            ["nan", "101.5", "99.5"],
        )
    d = lr.report(path)["diagnostics"]
    read = d["hardware_components"]["MD80HardwareInterface"]["read_cycle.execution_time"]
    assert read["worst_avg"] == pytest.approx(40.0)
    assert read["last"]["avg"] == pytest.approx(30.0)
    per = d["hardware_components"]["imu"]["read_cycle.periodicity"]
    assert per["worst_avg"] == pytest.approx(40.0)
    assert per["last"]["desired_hz"] == pytest.approx(100.0)
    assert d["controller_manager"]["periodicity.average"] == {
        "last": pytest.approx(99.5),
        "max": pytest.approx(101.5),
    }
    out = tmp_path / "r.json"
    assert lr.main([str(path), "--json", str(out)]) == 0
    capsys.readouterr()
    data = load_json_strict(out)
    assert data["diagnostics"]["controller_manager"]["periodicity.average"]["max"] == pytest.approx(101.5)


def test_diagnostics_nan_only(tmp_path, capsys):
    path = tmp_path / "nan_only"
    with Bag(make_store(), path) as b:
        write_diag_values(b, [NAN_STATS, NAN_STATS], ["nan", "nan"])
    out = tmp_path / "r.json"
    assert lr.main([str(path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "MD80HardwareInterface.read_cycle.execution_time: avg n/a [n/a - n/a] us, std n/a" in text
    assert "worst avg n/a" in text
    assert "periodicity.average n/a" in text
    data = load_json_strict(out)
    read = data["diagnostics"]["hardware_components"]["MD80HardwareInterface"]["read_cycle.execution_time"]
    assert read["worst_avg"] is None
    assert read["last"] == {"avg": None, "min": None, "max": None, "std": None, "unit": "us"}
    per = data["diagnostics"]["hardware_components"]["imu"]["read_cycle.periodicity"]
    assert per["last"]["desired_hz"] == pytest.approx(100.0)
    assert data["diagnostics"]["controller_manager"]["periodicity.average"] == {"last": None, "max": None}


def test_io_without_command_sent_leaves_sent_stages_unknown(tmp_path, capsys):
    """Targets received during dry_run can have armed set to true."""
    path = tmp_path / "no_sent"
    with Bag(make_store(IO_TIMING_NO_SENT), path) as b:
        write_policy(b)
        write_io(b, with_command_sent=False)
    r = lr.report(path)
    assert r["io_timing"] == {
        "total": 6,
        "armed": 5,
        "disarmed": 1,
        "command_sent_known": False,
        "command_sent": None,
        "armed_not_sent": None,
    }
    s = r["stages"]
    assert s["target_age"] is None
    assert s["state_relay"] is None
    assert s["pipeline"] is None
    assert s["joints_age"]["n"] == 5  # Policy stages remain available.
    assert r["pipeline_join"] == {"joined": None, "io_sent_unmatched": None, "policy_without_io": 1}
    assert r["topics"][lr.TOPIC_IO]["count"] == 6
    assert any("no command_sent field" in n and "command delivery unknown" in n for n in r["notes"])
    assert any(n.startswith("pipeline: not reported") for n in r["notes"])
    out = tmp_path / "r.json"
    assert lr.main([str(path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "6 samples, 5 armed, 1 armed=false, command delivery unknown" in text
    assert "pipeline join: unknown joined" in text
    assert load_json_strict(out)["io_timing"]["command_sent"] is None


def test_bag_with_only_an_unrelated_topic(tmp_path, capsys):
    """A bag with only unrelated topics yields an empty report."""
    path = tmp_path / "unrelated"
    with Bag(make_store(), path) as b:
        b.write("/chatter", "std_msgs/msg/String", b.msg("std_msgs/msg/String", data="hello"), T0)
    out = tmp_path / "r.json"
    assert lr.main([str(path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "notes:" in text
    for t in lr.TOPICS:
        assert f"{t}: not in the bag" in text
    data = load_json_strict(out)
    assert all(v["count"] == 0 for v in data["topics"].values())
    assert all(v is None for v in data["stages"].values())
