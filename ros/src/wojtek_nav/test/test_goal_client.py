import math

import pytest

from wojtek_nav.goal_client import local_goal, main


def test_goal_is_frozen_in_odom_at_submission():
    assert local_goal(10, 20, math.pi / 2, 2, 1) == pytest.approx((9, 22))


@pytest.mark.parametrize("args", [["--timeout", "0"], ["--forward", "nan"], ["--left", "inf"]])
def test_invalid_inputs_are_rejected_before_connecting(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


class FakeROS:
    """Only the ROS boundary is fake; the client runs its actual control loop."""

    def __init__(self, monkeypatch):
        import copy
        import sys
        from types import SimpleNamespace as NS

        from wojtek_nav import goal_client

        self.now = 0.0
        self.alive = False
        self.destroyed = self.shutdown = False
        self.messages = []
        self.logs = []
        self.subscribers = {"/wojtek/nav/goal": 1, "/wojtek/nav/cancel": 1}
        self.status_publishers = 1
        self.position = (10.0, 20.0, math.pi / 2)
        self.tf_available = True
        self.on_spin = self.on_publish = None
        self.destroy_error = None
        self.signal_options = None
        self.status_callback = None
        self.cancel_callback = None
        self.service_ready = True
        self.service_done = True
        self.service_result = NS(success=True, message="autonomy_enabled")
        self.service_requests = []

        runtime = self

        class Publisher:
            def __init__(self, topic):
                self.topic = topic

            def publish(self, message):
                runtime.messages.append((self.topic, copy.deepcopy(message), runtime.now))
                if runtime.on_publish:
                    runtime.on_publish(self.topic)

            def get_subscription_count(self):
                own = self.topic == "/wojtek/nav/cancel" and runtime.cancel_callback is not None
                return runtime.subscribers[self.topic] + int(own)

        class Node:
            def __init__(self, name):
                self.name = name

            def create_publisher(self, message_type, topic, qos):
                return Publisher(topic)

            def create_subscription(self, message_type, topic, callback, qos):
                if topic == "/wojtek/nav/status":
                    runtime.status_callback = callback
                else:
                    assert topic == "/wojtek/nav/cancel"
                    runtime.cancel_callback = callback
                return object()

            def create_client(self, service_type, name):
                assert name == "/wojtek/autonomy_enable"

                def call_async(request):
                    runtime.service_requests.append((request.data, runtime.now))
                    return NS(done=lambda: runtime.service_done,
                              result=lambda: runtime.service_result)

                return NS(service_is_ready=lambda: runtime.service_ready, call_async=call_async)

            def count_publishers(self, topic):
                return runtime.status_publishers

            def get_logger(self):
                return NS(**{level: lambda message, level=level: runtime.logs.append((level, message))
                             for level in ("info", "error", "warning")})

            def get_clock(self):
                return NS(now=lambda: NS(to_msg=lambda: runtime.now))

            def destroy_node(self):
                runtime.destroyed = True
                if runtime.destroy_error:
                    raise runtime.destroy_error

        class Buffer:
            def lookup_transform(self, target, source, stamp):
                assert (target, source) == ("odom", "base_link")
                if not runtime.tf_available:
                    raise RuntimeError("no transform")
                x, y, yaw = runtime.position
                return NS(transform=NS(translation=NS(x=x, y=y), rotation=NS(
                    x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))))

        def init(*, args, signal_handler_options):
            runtime.signal_options = signal_handler_options
            runtime.alive = True

        def spin_once(node, *, timeout_sec):
            assert 0 < timeout_sec <= 0.05
            runtime.now += timeout_sec
            if runtime.on_spin:
                runtime.on_spin()

        def try_shutdown():
            runtime.shutdown = True
            runtime.alive = False

        def pose_stamped():
            return NS(header=NS(frame_id=None, stamp=None), pose=NS(
                position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(w=0.0)))

        modules = {
            "rclpy": NS(init=init, ok=lambda: self.alive, spin_once=spin_once,
                        try_shutdown=try_shutdown, time=NS(Time=object)),
            "rclpy.node": NS(Node=Node),
            "rclpy.qos": NS(QoSProfile=lambda **kwargs: NS(**kwargs),
                            ReliabilityPolicy=NS(RELIABLE="reliable"),
                            DurabilityPolicy=NS(VOLATILE="volatile")),
            "rclpy.signals": NS(SignalHandlerOptions=NS(NO="no-signal-handlers")),
            "geometry_msgs.msg": NS(PoseStamped=pose_stamped),
            "std_msgs.msg": NS(Empty=object, String=object),
            "std_srvs.srv": NS(SetBool=NS(Request=NS)),
            "tf2_ros": NS(Buffer=Buffer, TransformListener=lambda *args: object()),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(goal_client, "time", NS(monotonic=lambda: self.now))

    @property
    def goals(self):
        return [message for topic, message, _ in self.messages if topic == "/wojtek/nav/goal"]

    @property
    def cancels(self):
        return [stamp for topic, _, stamp in self.messages if topic == "/wojtek/nav/cancel"]

    def status(self, value):
        from types import SimpleNamespace
        self.status_callback(SimpleNamespace(data=value))

    def cancel(self):
        if self.cancel_callback:
            self.cancel_callback(object())

    def assert_closed(self):
        assert self.destroyed and self.shutdown
        assert self.signal_options == "no-signal-handlers"


@pytest.fixture
def ros(monkeypatch):
    return FakeROS(monkeypatch)


@pytest.mark.parametrize("missing", ["cancel", "goal", "status", "tf"])
def test_startup_timeout_is_bounded_and_sends_no_goal(ros, missing):
    if missing in ("cancel", "goal"):
        ros.subscribers[f"/wojtek/nav/{missing}"] = 0
    elif missing == "status":
        ros.status_publishers = 0
    else:
        ros.tf_available = False
    assert main(["--timeout", "1"]) == 1
    assert not ros.goals
    assert ros.now == pytest.approx(1.0)
    ros.assert_closed()


def test_timeout_shorter_than_cancel_holdoff_cannot_send_goal(ros):
    assert main(["--timeout", "0.1"]) == 1
    assert not ros.goals
    assert ros.now == pytest.approx(0.1)
    ros.assert_closed()


def test_cancel_only_does_not_require_tf_or_status(ros):
    ros.tf_available = False
    ros.status_publishers = 0
    assert main(["--cancel", "--timeout", "1"]) == 0
    assert len(ros.cancels) == 1
    assert not ros.goals
    assert ros.now == pytest.approx(0.2)
    ros.assert_closed()


def test_goal_refresh_preserves_destination_and_success_checks_position(ros):
    def advance():
        if not ros.goals:
            return
        if ros.now >= 1.8:
            ros.status("reached")
            ros.position = (10.0, 20.5, 0.0)
        else:
            ros.status("driving")
            ros.position = (8.0, 19.0, 0.0)

    ros.on_spin = advance
    assert main(["--timeout", "3"]) == 0
    assert len(ros.goals) == 2
    for goal in ros.goals:
        assert goal.header.frame_id == "odom"
        assert (goal.pose.position.x, goal.pose.position.y) == pytest.approx((10, 20.5))
    assert len(ros.cancels) == 2
    ros.assert_closed()


@pytest.mark.parametrize("event, expected", [
    ("timeout", 1), ("blocked", 1), ("disconnect", 1),
    ("interrupt", 130), ("context_shutdown", 1),
])
def test_active_goal_stops_and_cleans_up(ros, event, expected):
    def trigger():
        if not ros.goals:
            return
        ros.on_spin = None
        if event == "blocked":
            ros.status("blocked")
        elif event == "disconnect":
            ros.subscribers["/wojtek/nav/goal"] = 0
        elif event == "interrupt":
            raise KeyboardInterrupt
        elif event == "context_shutdown":
            ros.alive = False

    ros.on_spin = trigger
    assert main(["--timeout", "1"]) == expected
    assert len(ros.goals) == 1
    assert len(ros.cancels) == (1 if event == "context_shutdown" else 2)
    assert ros.now <= 1.2 + 1e-9
    ros.assert_closed()


def test_interrupt_immediately_after_goal_publish_still_cancels(ros):
    def interrupt(topic):
        if topic == "/wojtek/nav/goal":
            raise KeyboardInterrupt

    ros.on_publish = interrupt
    assert main(["--timeout", "1"]) == 130
    assert len(ros.cancels) == 2
    ros.assert_closed()


@pytest.mark.parametrize("failure", [RuntimeError("transport lost"), KeyboardInterrupt()])
def test_cleanup_failure_still_destroys_node_and_shuts_down(ros, failure):
    def fail_cancel(topic):
        if topic == "/wojtek/nav/cancel" and ros.goals:
            raise failure

    ros.on_publish = fail_cancel
    ros.destroy_error = RuntimeError("destroy failed")
    assert main(["--timeout", "1"]) == 1
    assert len(ros.cancels) == 2
    ros.assert_closed()


def test_autonomy_enables_after_cancel_holdoff_and_before_first_goal(ros):
    assert main(["--autonomy", "--timeout", "1"]) == 1
    assert len(ros.goals) == 1
    [(enabled, stamp)] = ros.service_requests
    assert enabled
    first_goal_time = next(stamp for topic, _, stamp in ros.messages if topic == "/wojtek/nav/goal")
    assert ros.cancels[0] + 0.6 <= stamp <= first_goal_time
    assert len(ros.cancels) == 2
    ros.assert_closed()


@pytest.mark.parametrize("failure", ["unavailable", "no_reply", "manual_active"])
def test_autonomy_failure_never_sends_goal(ros, failure):
    if failure == "unavailable":
        ros.service_ready = False
    elif failure == "no_reply":
        ros.service_done = False
    else:
        ros.service_result.success = False
        ros.service_result.message = "manual_owner_active"
    assert main(["--autonomy", "--timeout", "1"]) == 1
    assert not ros.goals
    assert len(ros.cancels) == (1 if failure == "unavailable" else 2)
    assert ros.now <= 1.2 + 1e-9
    if failure == "manual_active":
        assert ("error", "autonomy refused: manual_owner_active") in ros.logs
    ros.assert_closed()


def test_legacy_mode_does_not_call_autonomy_service(ros):
    assert main(["--timeout", "1"]) == 1
    assert not ros.service_requests


@pytest.mark.parametrize("signal", ["cancel", "driving_then_idle", "turning_then_idle"])
def test_cancelled_goal_is_never_refreshed(ros, signal):
    def advance():
        if not ros.goals:
            return
        if signal == "cancel":
            # Cancellation can precede the first movement status.
            ros.cancel()
        elif ros.now < 1.6:
            ros.status(signal.removesuffix("_then_idle"))
        else:
            # Arrives just before the next one-second refresh.
            ros.status("idle")

    ros.on_spin = advance
    assert main(["--timeout", "5"]) == 1
    assert ("error", "navigation cancelled") in ros.logs
    assert len(ros.goals) == 1
    assert len(ros.cancels) == 2
    assert ros.now < 2.0
    ros.assert_closed()


@pytest.mark.parametrize("waiting", ["readiness", "autonomy_reply"])
def test_cancellation_while_preparing_never_submits_a_goal(ros, waiting):
    if waiting == "readiness":
        ros.status_publishers = 0
    else:
        ros.service_done = False

    def cancel():
        if waiting == "readiness" or ros.service_requests:
            ros.cancel()

    ros.on_spin = cancel
    assert main(["--autonomy", "--timeout", "5"]) == 1
    assert ("error", "navigation cancelled") in ros.logs
    assert not ros.goals
    assert ros.now < 1.0
    ros.assert_closed()


@pytest.mark.parametrize("startup_status", ["idle", "driving", "turning", "reached", "blocked"])
def test_discovery_status_and_initial_idle_do_not_cancel_new_goal(ros, startup_status):
    goal_ticks = 0

    def advance():
        nonlocal goal_ticks
        if not ros.goals:
            ros.status(startup_status)
            return
        goal_ticks += 1
        if goal_ticks == 1:
            ros.status("idle")  # May arrive before the goal has been processed.
        elif goal_ticks == 2:
            ros.status("driving")
        else:
            ros.position = (10.0, 20.5, 0.0)
            ros.status("reached")

    ros.on_spin = advance
    assert main(["--timeout", "5"]) == 0
    assert len(ros.goals) == 1
    assert len(ros.cancels) == 2
    ros.assert_closed()


def test_reached_then_idle_waits_for_tf_without_resubmitting(ros):
    reached = False

    def advance():
        nonlocal reached
        if not ros.goals:
            return
        if not reached:
            reached = True
            ros.status("reached")  # A nearby goal need not report driving first.
            ros.tf_available = False
        else:
            ros.status("idle")  # Normal tick after reached, not a cancellation.
            if ros.now >= 1.9:  # Past the next refresh deadline.
                ros.tf_available = True
                ros.position = (10.0, 20.5, 0.0)

    ros.on_spin = advance
    assert main(["--timeout", "3"]) == 0
    assert len(ros.goals) == 1
    assert ros.now >= 1.9
    ros.assert_closed()


def test_reached_without_matching_position_times_out_without_refresh(ros):
    def advance():
        if ros.goals:
            ros.status("reached")

    ros.on_spin = advance
    assert main(["--timeout", "3"]) == 1
    assert len(ros.goals) == 1
    assert ("error", "goal timeout") in ros.logs
    ros.assert_closed()
