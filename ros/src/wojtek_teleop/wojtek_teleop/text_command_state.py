"""Timed text-command state, independent of ROS."""


class CommandState:
    """Pure command core -- all behaviour, no rclpy (tested without ROS).

    `handle` arms a command, `twist_to_publish` is the timer's question:
    (vx, wz) while active, exactly one (0, 0) on stop/timeout, then None.
    """

    def __init__(self, v_forward, w_turn, command_timeout):
        self._timeout = float(command_timeout)
        # The whole vocabulary: motion commands map to (vx, wz); "stop"
        # (and, defensively, anything unknown) clears instead of mapping.
        self._map = {
            "forward": (float(v_forward), 0.0),
            "left": (0.0, float(w_turn)),
            "right": (0.0, -float(w_turn)),
        }
        self._twist = None        # (vx, wz) while a command is active
        self._deadline = 0.0
        self._stop_sent = True    # nothing to zero out yet

    @property
    def known_commands(self):
        return (*self._map, "stop")

    def handle(self, command, now):
        """Arm `command`; False for unknown ones (caller logs, we stop)."""
        command = command.strip().lower()
        twist = self._map.get(command)
        if twist is not None:
            self._twist = twist
            self._deadline = now + self._timeout
            self._stop_sent = False
            return True
        # "stop" and unknown commands both stop -- but only if a command
        # of ours is active; a zero burst from idle would shout over
        # whichever other source currently drives /cmd_vel.
        self._twist = None
        return command == "stop"

    def cancel(self):
        """Forget the active command; emit its stop once, then stay silent."""
        self._twist = None

    def twist_to_publish(self, now):
        if self._twist is not None and now <= self._deadline:
            return self._twist
        self._twist = None        # timed out (or stopped by handle)
        if self._stop_sent:
            return None
        self._stop_sent = True
        return (0.0, 0.0)
