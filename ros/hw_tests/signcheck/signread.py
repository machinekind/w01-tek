#!/usr/bin/env python3
"""Change of every actuated joint since start, in the drives' sense (URDF),
one line a second. Run on the robot next to a dry-run stack (see README).
Columns per leg: abduction, hip, lever; legs rear_left, rear_right,
front_right, front_left. A dot is less than 0.02 rad from the start."""
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ORDER = ("rear_left", "rear_right", "front_right", "front_left")
JOINTS = ("first", "second", "third")


class SignRead(Node):
    def __init__(self):
        super().__init__("signcheck")
        self.ref = None
        self.t = time.time()
        self.create_subscription(JointState, "/joint_states", self.cb, 10)

    def cb(self, msg):
        q = dict(zip(msg.name, msg.position))
        if self.ref is None:
            self.ref = q
            print(time.strftime("%H:%M:%S"), "reference taken; move one joint at a time", flush=True)
        if time.time() - self.t < 1.0:
            return
        self.t = time.time()
        cells = []
        for leg in ORDER:
            row = []
            for j in JOINTS:
                d = q[f"{leg}_{j}_joint"] - self.ref[f"{leg}_{j}_joint"]
                row.append(f"{d:+.2f}" if abs(d) > 0.02 else "  .  ")
            cells.append(f"{leg}[" + " ".join(row) + "]")
        print(time.strftime("%H:%M:%S"), "  ".join(cells), flush=True)


def main():
    rclpy.init()
    node = SignRead()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
