"""Send one local goal and report its result; see docs/integration.md."""

import argparse
import math
import sys
import time


def local_goal(x, y, yaw, forward, left):
    """Freeze a body-relative displacement into odom coordinates."""
    return (x + math.cos(yaw) * forward - math.sin(yaw) * left,
            y + math.sin(yaw) * forward + math.cos(yaw) * left)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=float, default=0.5, help="metres ahead")
    parser.add_argument("--left", type=float, default=0.0, help="metres left")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="seconds, plus at most 0.2 s for cancel cleanup")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cancel", action="store_true", help="cancel navigation and exit")
    mode.add_argument("--autonomy", action="store_true",
                      help="enable autonomy through the opt-in command mux before sending")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) for v in (args.forward, args.left, args.timeout)) or args.timeout <= 0:
        parser.error("distances must be finite and timeout must be positive")

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Empty, String
    from tf2_ros import Buffer, TransformListener

    node, cancel_pub = None, None
    deadline = time.monotonic() + args.timeout
    cancel_needed, result = False, 1

    try:
        # Keep the ROS context alive through Python's KeyboardInterrupt so
        # Ctrl-C can publish cancellation before destroying its publisher.
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = Node("wojtek_goal_client")
        goal_pub = node.create_publisher(PoseStamped, "/wojtek/nav/goal", 10)
        cancel_pub = node.create_publisher(Empty, "/wojtek/nav/cancel", 10)
        # Volatile ignores the previous session's retained result.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        state = {"goal_sent": False, "started": False, "terminal": None,
                 "cancelled": False}

        def on_status(msg):
            node.get_logger().info(msg.data)
            if not state["goal_sent"] or state["terminal"] is not None:
                return
            if msg.data in ("turning", "driving"):
                state["started"] = True
            elif msg.data in ("blocked", "reached"):
                # Goto reports idle on the tick after reached. Keep reached
                # while waiting for the corresponding TF update; never resend.
                state["terminal"] = msg.data
            elif msg.data == "idle" and state["started"]:
                state["cancelled"] = True

        def on_cancel(_msg):
            state["cancelled"] = True

        subscription = node.create_subscription(String, "/wojtek/nav/status", on_status, qos)
        transforms = Buffer()
        listener = TransformListener(transforms, node)

        def spin_until(end):
            while rclpy.ok():
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(node, timeout_sec=min(0.05, remaining))

        def spin_step():
            if state["cancelled"]:
                raise RuntimeError("navigation cancelled")
            if not rclpy.ok():
                raise RuntimeError("ROS context shut down")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("goal timeout")
            rclpy.spin_once(node, timeout_sec=min(0.05, remaining))
            if state["cancelled"]:
                raise RuntimeError("navigation cancelled")

        def pose():
            transform = transforms.lookup_transform("odom", "base_link", rclpy.time.Time()).transform
            q = transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            value = transform.translation.x, transform.translation.y, yaw
            if not all(math.isfinite(v) for v in value):
                raise ValueError("non-finite odom pose")
            return value

        while not cancel_pub.get_subscription_count():
            spin_step()
        cancel_pub.publish(Empty())
        # A cancel-only request needs a short delivery window, not a goal hold-off.
        spin_until(min(deadline, time.monotonic() + (0.2 if args.cancel else 0.6)))
        if not rclpy.ok():
            raise RuntimeError("ROS context shut down")
        if args.cancel:
            node.get_logger().info("cancel requested; this does not disable motor torque")
            return 0
        # Empty has no sender/request ID. After settling our initial cancel,
        # any cancel (including a delayed echo) ends this invocation.
        # Subscribe only now: our own subscription must not satisfy discovery.
        cancel_subscription = node.create_subscription(Empty, "/wojtek/nav/cancel", on_cancel, 10)
        while True:
            spin_step()
            if not goal_pub.get_subscription_count() or not node.count_publishers("/wojtek/nav/status"):
                continue
            try:
                start = pose()
                break
            except Exception:
                continue
        if args.autonomy:
            from std_srvs.srv import SetBool

            autonomy = node.create_client(SetBool, "/wojtek/autonomy_enable")
            while not autonomy.service_is_ready():
                spin_step()
            request = SetBool.Request()
            request.data = True
            # A request may reach the mux even if its reply is lost.
            cancel_needed = True
            response = autonomy.call_async(request)
            while not response.done():
                spin_step()
            reply = response.result()
            if not reply.success:
                raise RuntimeError(f"autonomy refused: {reply.message}")
            start = pose()
        x, y = local_goal(*start, args.forward, args.left)
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.pose.position.x, goal.pose.position.y = x, y
        goal.pose.orientation.w = 1.0
        next_send = 0.0
        node.get_logger().info(f"goal in odom: x={x:.3f}, y={y:.3f}; robot must already be armed")
        while True:
            if not rclpy.ok():
                raise RuntimeError("ROS context shut down")
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("goal timeout")
            if not goal_pub.get_subscription_count() or not node.count_publishers("/wojtek/nav/status"):
                raise RuntimeError("navigation disconnected")
            if state["terminal"] is None and now >= next_send:
                goal.header.stamp = node.get_clock().now().to_msg()
                # Mark intent first: an interrupt immediately after publish
                # must still send cancellation during cleanup.
                cancel_needed = True
                state["goal_sent"] = True
                goal_pub.publish(goal)
                next_send = now + 1.0  # controller goal lease is 3 s
            spin_step()
            if state["terminal"] == "blocked":
                raise RuntimeError("route blocked; choose another waypoint")
            if state["terminal"] == "reached":
                try:
                    px, py, _ = pose()
                except Exception:
                    continue
                if math.hypot(px - x, py - y) <= 0.15:
                    result = 0
                    break
        # Keep these ROS entities alive for the entire exchange.
        del subscription, cancel_subscription, listener
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("interrupted")
        result = 130
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(str(exc), file=sys.stderr)
    finally:
        try:
            if cancel_needed and rclpy.ok():
                cancel_pub.publish(Empty())
                spin_until(time.monotonic() + 0.2)
        except (Exception, KeyboardInterrupt) as exc:
            if node is not None:
                node.get_logger().warning(f"cancel delivery interrupted: {type(exc).__name__}")
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            except (Exception, KeyboardInterrupt):
                pass
            finally:
                try:
                    rclpy.try_shutdown()
                except (Exception, KeyboardInterrupt):
                    pass
    return result


if __name__ == "__main__":
    raise SystemExit(main())
