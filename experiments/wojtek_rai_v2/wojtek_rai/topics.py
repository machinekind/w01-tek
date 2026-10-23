"""Step 1 smoke: list the live ROS 2 graph through RAI's own connector.

    ./experiments/wojtek_rai_v2/run.sh topics

Exit status 1 when any of the topics/services this experiment relies on is
missing, so the check is scriptable.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Sequence, Tuple

from wojtek_rai.limits import REQUIRED_SERVICES, REQUIRED_TOPICS

NamesAndTypes = Sequence[Tuple[str, Sequence[str]]]

# DDS discovery is not instantaneous; poll a little before judging.
DISCOVERY_TIMEOUT_S = 10.0
DISCOVERY_POLL_S = 0.5


def format_graph(topics: NamesAndTypes, services: NamesAndTypes) -> str:
    """Render the graph as sorted `name  type` lines."""
    lines = ["topics:"]
    lines += [f"  {n}  {', '.join(t)}" for n, t in sorted(topics)]
    lines.append("services:")
    lines += [
        f"  {n}  {', '.join(t)}"
        for n, t in sorted(services)
        if "parameter" not in n
    ]
    return "\n".join(lines)


def missing(names_and_types: NamesAndTypes, required: Iterable[str]) -> list[str]:
    """Required names absent from a names-and-types listing."""
    present = {n for n, _ in names_and_types}
    return [r for r in required if r not in present]


def main() -> int:
    # Imported here so the pure helpers above stay importable without ROS.
    from rai.communication.ros2 import ROS2Connector

    connector = ROS2Connector(node_name="wojtek_rai_topics")
    try:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
        while True:
            topics = connector.get_topics_names_and_types()
            services = connector.get_services_names_and_types()
            gaps = missing(topics, REQUIRED_TOPICS) + missing(services, REQUIRED_SERVICES)
            if not gaps or time.monotonic() > deadline:
                break
            time.sleep(DISCOVERY_POLL_S)
    finally:
        connector.shutdown()

    print(format_graph(topics, services))
    if gaps:
        hint = (
            "/wojtek/nav_command exists only while text_commander runs (always in the "
            "simulation; on the physical robot start it on the PC: "
            "ros2 run wojtek_teleop text_commander)"
            if gaps == [REQUIRED_TOPICS[0]] else "is ./ros/sim.sh running, or the robot stack up?"
        )
        print(f"MISSING: {' '.join(gaps)}  ({hint})", file=sys.stderr)
        return 1
    print("OK: every required topic and service is visible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
