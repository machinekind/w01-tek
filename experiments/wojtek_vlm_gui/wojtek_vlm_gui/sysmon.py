"""The robot's computer: per-core usage, load, memory, temperature.

Plain reads of /proc and /sys, no ROS, no UI. Parsing is separate from
reading so the tests run on strings. `sysmon_node` runs this on the RPi
and publishes it; the page, on the PC, only shows what arrives, and the
panel names the machine it came from (`Snapshot.host`).

The robot's RPi isolates cores for the 200 Hz control loop and the policy
(`isolcpus`); those are read from /sys and marked, so an idle RT core
reads as "reserved", not as a machine with nothing to do.
"""

from __future__ import annotations

import socket
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

STAT_PATH = "/proc/stat"
LOADAVG_PATH = "/proc/loadavg"
MEMINFO_PATH = "/proc/meminfo"
THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
ISOLATED_PATH = "/sys/devices/system/cpu/isolated"

Jiffies = Tuple[int, int]  # (busy, total)


def parse_stat(text: str) -> Dict[str, Jiffies]:
    """`/proc/stat` -> {"cpu": (busy, total), "cpu0": ..., ...} in jiffies.
    Idle is idle + iowait, the kernel's own reading of "nothing to run"."""
    out: Dict[str, Jiffies] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        vals = [int(v) for v in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        out[parts[0]] = (total - idle, total)
    return out


def core_usage(before: Dict[str, Jiffies], after: Dict[str, Jiffies]) -> List[float]:
    """Percent busy per core between two samples, in core order (cpu0
    first). A core with no elapsed jiffies reads 0."""
    cores = sorted((k for k in after if k != "cpu"), key=lambda k: int(k[3:]))
    usage: List[float] = []
    for core in cores:
        busy0, total0 = before.get(core, (0, 0))
        busy1, total1 = after[core]
        elapsed = total1 - total0
        usage.append(100.0 * (busy1 - busy0) / elapsed if elapsed > 0 else 0.0)
    return usage


def parse_loadavg(text: str) -> Tuple[float, float, float]:
    parts = text.split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_meminfo(text: str) -> Tuple[int, int]:
    """(used, total) in MB, used = total - MemAvailable (what the kernel
    could not hand out right now, caches excluded)."""
    fields: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            fields[key] = int(rest.split()[0])
    total_kb = fields.get("MemTotal", 0)
    avail_kb = fields.get("MemAvailable", total_kb)
    return (total_kb - avail_kb) // 1024, total_kb // 1024


def parse_temp(text: str) -> Optional[float]:
    """The thermal zone's millidegrees as degrees C; None for garbage."""
    try:
        return int(text.strip()) / 1000.0
    except ValueError:
        return None


def parse_isolated(text: str) -> List[int]:
    """`isolcpus` as the kernel lists it: "2-3", "1,3", "" -> core numbers."""
    cores: List[int] = []
    for chunk in text.strip().split(","):
        if not chunk:
            continue
        lo, _, hi = chunk.partition("-")
        cores.extend(range(int(lo), int(hi or lo) + 1))
    return cores


@dataclass
class Snapshot:
    host: str
    cores: List[float] = field(default_factory=list)      # percent busy per core
    isolated: List[int] = field(default_factory=list)     # the RT cores
    load: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    temp_c: Optional[float] = None


def snapshot_to_dict(snap: "Snapshot") -> Dict[str, Any]:
    """The snapshot as the JSON the sysmon node publishes."""
    return asdict(snap)


def snapshot_from_dict(data: Dict[str, Any]) -> "Snapshot":
    """The inverse; a missing field takes the default, never raises on a
    reading from an older node."""
    return Snapshot(
        host=str(data.get("host", "?")),
        cores=[float(c) for c in data.get("cores", [])],
        isolated=[int(c) for c in data.get("isolated", [])],
        load=tuple(float(v) for v in data.get("load", (0.0, 0.0, 0.0)))[:3],  # type: ignore[arg-type]
        mem_used_mb=int(data.get("mem_used_mb", 0)),
        mem_total_mb=int(data.get("mem_total_mb", 0)),
        temp_c=None if data.get("temp_c") is None else float(data["temp_c"]),
    )


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


class SysMon:
    """Keeps the previous /proc/stat sample, so every `read()` reports the
    usage since the last one (the first call reports since boot)."""

    def __init__(self, host: Optional[str] = None) -> None:
        self.host = host or socket.gethostname()
        self._prev: Dict[str, Jiffies] = {}
        self.isolated = parse_isolated(_read(ISOLATED_PATH))

    def read(self) -> Snapshot:
        now = parse_stat(_read(STAT_PATH))
        cores = core_usage(self._prev, now) if now else []
        self._prev = now
        used, total = parse_meminfo(_read(MEMINFO_PATH))
        load_text = _read(LOADAVG_PATH)
        return Snapshot(
            host=self.host,
            cores=cores,
            isolated=[c for c in self.isolated if c < len(cores)],
            load=parse_loadavg(load_text) if load_text else (0.0, 0.0, 0.0),
            mem_used_mb=used,
            mem_total_mb=total,
            temp_c=parse_temp(_read(THERMAL_PATH)),
        )


def core_label(index: int, isolated: Sequence[int]) -> str:
    return f"core {index}" + (" (RT, reserved)" if index in isolated else "")
