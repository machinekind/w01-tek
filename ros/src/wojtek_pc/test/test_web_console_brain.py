"""The web console forgets the brain's status once the brain is gone.

The bug this guards: the console kept the brain's last status JSON in
memory for ever and showed it to every page that connected. Stop the brain
with Ctrl-C mid-task and the page went on saying "step 3 ask ..." as if it
were still working. Now a 1 Hz check on the ROS thread counts the
publishers on wojtek/vlm/status, and when there are none it drops the copy
and tells the open pages once.

Needs rclpy (skipped on a host without ROS); run in the dev container:

    pytest ros/src/wojtek_pc/test/test_web_console_brain.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

# Never let a test node reach the robot's graph: own domain, this host only.
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

rclpy = pytest.importorskip("rclpy")
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

# The sources in this tree, not whatever an older install has.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "wojtek_policy"))
sys.path.insert(0, str(SRC / "wojtek_pc"))
from wojtek_pc.web_console import ConsoleNode  # noqa: E402

TEST_DOMAIN = 88
GONE = {"t": "vlm", "status": None}


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init(domain_id=TEST_DOMAIN)
    yield
    rclpy.shutdown()


@pytest.fixture
def console():
    emitted = []
    node = ConsoleNode(emitted.append, lambda: False)
    yield node, emitted
    node.destroy_node()


def test_brain_status_forgotten_when_no_publisher(console):
    node, emitted = console
    node.vlm_status = {"action": "ask", "step": 3}
    node.count_publishers = lambda name: 0
    node._check_brain()
    assert node.vlm_status is None
    assert emitted == [GONE]
    assert node.vlm_snapshot()[0] == GONE
    # Once, not every second: the next ticks find nothing to forget.
    node._check_brain()
    assert len(emitted) == 1


def test_brain_status_kept_while_brain_publishes(console):
    node, emitted = console
    node.vlm_status = {"action": "ask", "step": 3}
    node.count_publishers = lambda name: 1
    node._check_brain()
    assert node.vlm_status == {"action": "ask", "step": 3}
    assert emitted == []


def test_no_status_no_broadcast(console):
    node, emitted = console
    node.vlm_status = None
    node.count_publishers = lambda name: 0
    node._check_brain()
    assert emitted == []


def test_a_real_brain_coming_and_going(console):
    """The whole path on a real graph: the latched status arrives, stays
    while its publisher lives (the check must see it, or the panel would
    blank every second), and goes once the publisher is destroyed."""
    node, emitted = console
    assert node._vlm_sub.topic_name == "/wojtek/vlm/status"
    brain = rclpy.create_node("fake_vlm_brain")
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = brain.create_publisher(String, "wojtek/vlm/status", latched)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(brain)

    def spin_until(done, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not done():
            ex.spin_once(timeout_sec=0.05)
        return done()

    try:
        pub.publish(String(data=json.dumps({"action": "ask"})))
        assert spin_until(lambda: node.vlm_status is not None, 5.0), \
            "the brain's status never arrived"
        assert node.vlm_status == {"action": "ask"}
        # Longer than one check: a live brain must keep its status.
        spin_until(lambda: False, 1.5)
        assert node.vlm_status == {"action": "ask"}, \
            "the check forgot a brain that is still publishing"

        brain.destroy_publisher(pub)
        assert spin_until(lambda: node.vlm_status is None, 5.0), \
            "the status outlived its publisher"
        vlm_frames = [f for f in emitted if isinstance(f, dict) and f.get("t") == "vlm"]
        assert vlm_frames[-1] == GONE
        assert vlm_frames.count(GONE) == 1
    finally:
        ex.shutdown()
        brain.destroy_node()
