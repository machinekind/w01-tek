#!/usr/bin/env python3
"""Operator console -- one PC-side GUI for every manual robot capability.

    ros2 run wojtek_viz console

Instead of typing raw `ros2 service call` / `ros2 topic pub`, this gathers all
the hand-operated controls in one window. It talks to the SAME services/topics
the nodes already expose -- nothing in wojtek_bringup / wojtek_policy changes:

  * Services (M1): /wojtek/arm (SetBool), /wojtek/zero, /wojtek/stand_up,
    /wojtek/lie_down (Trigger), /wojtek/enable (SetBool), /wojtek/reset (Trigger).
  * Telemetry (M2): measured joint angles from /wojtek/joint_states_abs and
    orientation/angular-rate from /imu_sensor_broadcaster/imu, shown live.
  * Joint jog (M3): per-joint sliders publish /wojtek/joint_targets (absolute
    URDF JointState). real_io_node applies them ONLY when armed, and does NOT
    ramp -- so we ramp here, nudging the published target toward the slider set-
    point in small steps. Needs: armed + policy DISABLED (else policy_node's own
    joint_targets fight ours). Measured is shown next to each commanded slider.
  * Drive pad (M4): an XY pad (vx,yaw) + a strafe slider + a height slider
    publish /cmd_vel (Twist) at a fixed rate; releasing the pad/strafe
    recenters to zero (stop), height holds its set-point. Needs: armed +
    policy ENABLED. This is the opposite mode to jog.

There is NO dedicated arm/enable status topic, so the console tracks that state
locally from the service responses it gets back.

Threading: rclpy spins in a background thread; all ROS->GUI updates cross over
through Qt signals (queued, thread-safe). Service calls are async so the UI
never blocks.
"""
import signal
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Console-side cap on the XY-pad drive commands [vx, vy, yaw]. Deliberately
# conservative; policy_node additionally clips /cmd_vel to the loaded
# policy's trained command box.
CMD_LIMIT = (0.6, 0.4, 0.7)
# Commanded standing height (m): trained envelope and default stance. Keep in
# sync with the loaded policy contract's command_height and the training env's
# command.height range. Published on Twist linear.z (0 = use policy default).
HEIGHT_RANGE = (0.09, 0.17)
HEIGHT_DEFAULT = 0.125

# Jog: how fast the published target chases the slider set-point (rad/s), and
# the publish/ramp tick rate. Deliberately slow -- this drives real motors.
JOG_RAMP_RATE = 0.8      # rad/s
JOG_TICK_HZ = 50.0
DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate while driving
SLIDER_SCALE = 1000.0    # rad <-> integer slider units
DEFAULT_JOINT_LIMIT = 3.14  # fallback slider half-range when URDF has none


# --------------------------------------------------------------------------- #
# ROS side                                                                     #
# --------------------------------------------------------------------------- #
class Signals(QObject):
    """Marshals ROS-thread events onto the Qt (GUI) thread."""
    joints = pyqtSignal(object)          # (names, positions, velocities)
    imu = pyqtSignal(object)             # (quat xyzw, angular_velocity xyz)
    limits = pyqtSignal(object)          # {joint_name: (lower, upper)}
    service_result = pyqtSignal(str, bool, str)   # label, success, message


class RosNode(Node):
    def __init__(self, sig: Signals):
        super().__init__("operator_console")
        self.sig = sig

        # --- service clients (M1) ---
        self._cli = {
            "arm": self.create_client(SetBool, "wojtek/arm"),
            "enable": self.create_client(SetBool, "wojtek/enable"),
            "zero": self.create_client(Trigger, "wojtek/zero"),
            "stand_up": self.create_client(Trigger, "wojtek/stand_up"),
            "lie_down": self.create_client(Trigger, "wojtek/lie_down"),
            "reset": self.create_client(Trigger, "wojtek/reset"),
        }

        # --- publishers (M3/M4) ---
        self._pub_targets = self.create_publisher(JointState, "wojtek/joint_targets", 10)
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)

        # Canonical actuated joint names, from the robot constants in
        # wojtek_policy.poses (same source real_io uses). Lets the jog UI
        # build immediately, even in sim where wojtek/joint_states_abs is
        # never published.
        self.joint_names = self._load_joint_names()

        # --- telemetry (M2) ---
        # Measured angles: prefer wojtek/joint_states_abs (real_io, absolute
        # URDF). In sim real_io is absent and mujoco publishes plain
        # joint_states -- also absolute URDF there -- so fall back to it until
        # abs appears. On the real robot joint_states is boot-relative, so once
        # we've seen abs we must ignore joint_states.
        self._have_abs = False
        self.create_subscription(
            JointState, "wojtek/joint_states_abs", self._on_joints_abs, 10)
        self.create_subscription(
            JointState, "joint_states", self._on_joints_raw, 10)
        self.create_subscription(
            Imu, "imu_sensor_broadcaster/imu", self._on_imu, 10)

        # URDF (for jog slider limits): latched String from robot_state_publisher.
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "robot_description", self._on_urdf, latched)

    def _load_joint_names(self):
        try:
            from wojtek_policy import poses
            return list(poses.ACTUATOR_NAMES)
        except ImportError as e:
            self.get_logger().warning(f"no canonical joint names ({e}); "
                                      "jog will build from telemetry instead")
            return []

    # ---- telemetry callbacks (run in ROS thread -> emit to GUI) -----------
    def _on_joints_abs(self, msg: JointState):
        self._have_abs = True
        self._emit_joints(msg)

    def _on_joints_raw(self, msg: JointState):
        if self._have_abs:      # real robot: joint_states is boot-relative, skip
            return
        self._emit_joints(msg)

    def _emit_joints(self, msg: JointState):
        vel = list(msg.velocity) if len(msg.velocity) == len(msg.name) else []
        self.sig.joints.emit((list(msg.name), list(msg.position), vel))

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        w = msg.angular_velocity
        self.sig.imu.emit(((q.x, q.y, q.z, q.w), (w.x, w.y, w.z)))

    def _on_urdf(self, msg: String):
        limits = {}
        try:
            root = ET.fromstring(msg.data)
            for j in root.iter("joint"):
                lim = j.find("limit")
                name = j.get("name")
                if name is not None and lim is not None and lim.get("lower") is not None:
                    limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
        except ET.ParseError:
            return
        if limits:
            self.sig.limits.emit(limits)

    # ---- commands (called from GUI thread; publish/async are thread-safe) --
    def call_setbool(self, key: str, value: bool):
        self._call(key, self._cli[key], SetBool.Request(data=value))

    def call_trigger(self, key: str):
        self._call(key, self._cli[key], Trigger.Request())

    def _call(self, label, client, req):
        if not client.service_is_ready():
            self.sig.service_result.emit(label, False, "service unavailable")
            return
        fut = client.call_async(req)
        fut.add_done_callback(lambda f: self._on_response(label, f))

    def is_ready(self, key: str) -> bool:
        return self._cli[key].service_is_ready()

    def _on_response(self, label, fut):
        try:
            resp = fut.result()
            self.sig.service_result.emit(label, bool(resp.success), resp.message)
        except Exception as e:  # noqa: BLE001 -- surface any RPC failure to the UI
            self.sig.service_result.emit(label, False, str(e))

    def publish_targets(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._pub_targets.publish(msg)

    def publish_cmd(self, vx, vy, yaw, height=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(yaw)
        # Standing-height command; policy_node treats 0 as "use the default".
        t.linear.z = float(height)
        self._pub_cmd.publish(t)


# --------------------------------------------------------------------------- #
# Drive pad widget (M4)                                                        #
# --------------------------------------------------------------------------- #
class DrivePad(QWidget):
    """2D pad: drag = (vx, yaw) command; release recenters to zero (stop).

    Screen up = +vx (forward), screen right = -wz (turn right; ROS +z yaw is
    counter-clockwise). Normalized to [-1, 1] on each axis; the window scales
    by CMD_LIMIT before publishing.
    """
    moved = pyqtSignal(float, float)  # normalized (fwd, ccw) in [-1, 1]

    def __init__(self):
        super().__init__()
        self.setMinimumSize(180, 180)
        self._x = 0.0  # right(+)/left(-)
        self._y = 0.0  # down(+)/up(-)

    def _set_from_pos(self, pos):
        r = min(self.width(), self.height()) / 2 - 8
        cx, cy = self.width() / 2, self.height() / 2
        self._x = max(-1.0, min(1.0, (pos.x() - cx) / r))
        self._y = max(-1.0, min(1.0, (pos.y() - cy) / r))
        self.update()
        self.moved.emit(-self._y, -self._x)  # fwd = -screenY, ccw = -screenX

    def mousePressEvent(self, e):
        self._set_from_pos(e.pos())

    def mouseMoveEvent(self, e):
        self._set_from_pos(e.pos())

    def mouseReleaseEvent(self, e):
        self._x = self._y = 0.0
        self.update()
        self.moved.emit(0.0, 0.0)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) / 2 - 8
        cx, cy = w / 2, h / 2
        p.setPen(QColor(120, 120, 120))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        p.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        p.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
        kx, ky = cx + self._x * r, cy + self._y * r
        p.setBrush(QColor(70, 140, 220))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(kx - 9), int(ky - 9), 18, 18)


# --------------------------------------------------------------------------- #
# Main window                                                                  #
# --------------------------------------------------------------------------- #
class Console(QWidget):
    def __init__(self, node: RosNode, sig: Signals):
        super().__init__()
        self.node = node
        self.setWindowTitle("wojtek operator console")

        # local, service-response-tracked state (no status topic exists)
        self._armed = False
        self._policy_on = False

        # jog state
        self._joint_names = []
        self._measured = {}          # name -> measured position (rad)
        self._limits = {}            # name -> (lower, upper) from URDF
        self._jog_rows = {}          # name -> (slider, cmd_label, meas_label)
        self._jog_target = None      # np.array of currently-published targets
        self._jog_built = False

        root = QVBoxLayout(self)
        root.addWidget(self._build_services())
        self._imu_label = QLabel("IMU:  (waiting) ")
        root.addWidget(self._imu_label)
        self._jog_box = self._build_jog()
        root.addWidget(self._jog_box)
        # Build jog rows up front from the canonical joint list (works in sim,
        # where no joint_states_abs arrives). Falls back to a lazy build off the
        # first telemetry message if the canonical list was unavailable.
        if self.node.joint_names:
            self._build_jog_rows(self.node.joint_names)
        root.addWidget(self._build_drive())
        self._status = QLabel("ready")
        self._status.setStyleSheet("color: #888;")
        root.addWidget(self._status)

        # wire ROS signals -> GUI slots
        sig.joints.connect(self._on_joints)
        sig.imu.connect(self._on_imu)
        sig.limits.connect(self._on_limits)
        sig.service_result.connect(self._on_service_result)

        # jog ramp + publish tick
        self._jog_timer = QTimer(self)
        self._jog_timer.timeout.connect(self._jog_tick)
        self._jog_timer.start(int(1000 / JOG_TICK_HZ))

        # drive publish tick
        self._drive_vx = self._drive_vy = self._drive_yaw = 0.0
        self._drive_height = HEIGHT_DEFAULT
        self._drive_timer = QTimer(self)
        self._drive_timer.timeout.connect(self._drive_tick)
        self._drive_timer.start(int(1000 / DRIVE_TICK_HZ))

        # service-availability poll: grey out buttons with no live server
        self._avail_timer = QTimer(self)
        self._avail_timer.timeout.connect(self._refresh_availability)
        self._avail_timer.start(1000)
        self._refresh_availability()

    # ---- M1: services ------------------------------------------------------
    def _build_services(self):
        box = QGroupBox("Motor power / poses / policy")
        lay = QHBoxLayout(box)
        # key -> button, so a poll can grey out controls whose service has no
        # server yet (e.g. arm/pose need real_io_node -- absent in --sim, where
        # only policy's enable/reset are live). Avoids "click -> failed".
        self._svc_buttons = {}
        self._arm_btn = QPushButton("ARM")
        self._arm_btn.setCheckable(True)
        self._arm_btn.clicked.connect(self._toggle_arm)
        lay.addWidget(self._arm_btn)
        self._svc_buttons["arm"] = self._arm_btn
        for label, key in (("Zero", "zero"), ("Stand up", "stand_up"),
                           ("Lie down", "lie_down"), ("Policy reset", "reset")):
            b = QPushButton(label)
            b.clicked.connect(lambda _c, k=key: self.node.call_trigger(k))
            lay.addWidget(b)
            self._svc_buttons[key] = b
        self._policy_btn = QPushButton("Policy ON")
        self._policy_btn.setCheckable(True)
        self._policy_btn.clicked.connect(self._toggle_policy)
        lay.addWidget(self._policy_btn)
        self._svc_buttons["enable"] = self._policy_btn
        return box

    def _refresh_availability(self):
        """Grey out buttons whose service currently has no server."""
        for key, btn in self._svc_buttons.items():
            ready = self.node.is_ready(key)
            if btn.isEnabled() != ready:
                btn.setEnabled(ready)
                btn.setToolTip("" if ready
                               else f"/wojtek/{key} unavailable (no server -- "
                                    f"real_io not up? sim has no arm/pose services)")

    def _toggle_arm(self):
        want = self._arm_btn.isChecked()
        self.node.call_setbool("arm", want)  # state confirmed in _on_service_result

    def _toggle_policy(self):
        want = self._policy_btn.isChecked()
        self.node.call_setbool("enable", want)

    # ---- M3: jog -----------------------------------------------------------
    def _build_jog(self):
        box = QGroupBox("Joint jog  (needs: ARMED + Policy OFF)")
        self._jog_grid = QGridLayout(box)
        self._jog_enable = QCheckBox("enable jog (publish targets)")
        self._jog_enable.toggled.connect(self._on_jog_enable)
        self._jog_grid.addWidget(self._jog_enable, 0, 0, 1, 4)
        self._jog_grid.addWidget(QLabel("waiting for joint list ..."), 1, 0)
        return box

    def _build_jog_rows(self, names):
        # clear the placeholder row
        placeholder = self._jog_grid.itemAtPosition(1, 0)
        if placeholder is not None:
            placeholder.widget().deleteLater()
        for row, name in enumerate(names, start=1):
            lo, hi = self._limits.get(name, (-DEFAULT_JOINT_LIMIT, DEFAULT_JOINT_LIMIT))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(lo * SLIDER_SCALE))
            slider.setMaximum(int(hi * SLIDER_SCALE))
            cmd_lbl = QLabel("cmd —")
            meas_lbl = QLabel("meas —")
            meas_lbl.setStyleSheet("color: #4a8;")
            self._jog_grid.addWidget(QLabel(name), row, 0)
            self._jog_grid.addWidget(slider, row, 1)
            self._jog_grid.addWidget(cmd_lbl, row, 2)
            self._jog_grid.addWidget(meas_lbl, row, 3)
            self._jog_rows[name] = (slider, cmd_lbl, meas_lbl)
        self._joint_names = list(names)
        self._jog_built = True

    def _on_jog_enable(self, on):
        if on:
            # start from measured so we don't jump the robot
            if not self._joint_names:
                self._jog_enable.setChecked(False)
                self._set_status("no joint list yet -- cannot jog")
                return
            # Need measured angles to start from, or we'd jump the robot from 0.
            if not any(n in self._measured for n in self._joint_names):
                self._jog_enable.setChecked(False)
                self._set_status("no measured joint angles yet -- cannot jog safely")
                return
            cur = np.array([self._measured.get(n, 0.0) for n in self._joint_names])
            self._jog_target = cur.copy()
            for n in self._joint_names:
                slider = self._jog_rows[n][0]
                slider.blockSignals(True)
                slider.setValue(int(self._measured.get(n, 0.0) * SLIDER_SCALE))
                slider.blockSignals(False)
            if not self._armed:
                self._set_status("jog on, but NOT armed -- targets ignored until you ARM")
        else:
            self._jog_target = None

    def _jog_tick(self):
        if self._jog_target is None or not self._joint_names:
            return
        setpoint = np.array(
            [self._jog_rows[n][0].value() / SLIDER_SCALE for n in self._joint_names])
        step = JOG_RAMP_RATE / JOG_TICK_HZ
        delta = np.clip(setpoint - self._jog_target, -step, step)
        self._jog_target = self._jog_target + delta
        self.node.publish_targets(self._joint_names, self._jog_target)
        for i, n in enumerate(self._joint_names):
            self._jog_rows[n][1].setText(f"cmd {self._jog_target[i]:+.3f}")

    # ---- M4: drive ---------------------------------------------------------
    def _build_drive(self):
        box = QGroupBox("Drive  (needs: ARMED + Policy ON)")
        lay = QHBoxLayout(box)
        self._drive_enable = QCheckBox("enable drive")
        self._drive_enable.toggled.connect(self._on_drive_enable)
        lay.addWidget(self._drive_enable)
        self._pad = DrivePad()
        self._pad.moved.connect(self._on_pad)
        lay.addWidget(self._pad)
        col = QVBoxLayout()
        col.addWidget(QLabel("strafe"))
        self._strafe = QSlider(Qt.Horizontal)
        self._strafe.setMinimum(-1000)
        self._strafe.setMaximum(1000)
        self._strafe.sliderReleased.connect(lambda: self._strafe.setValue(0))
        self._strafe.valueChanged.connect(self._on_strafe)
        col.addWidget(self._strafe)
        col.addWidget(QLabel("height"))
        self._height = QSlider(Qt.Horizontal)
        self._height.setMinimum(int(HEIGHT_RANGE[0] * SLIDER_SCALE))
        self._height.setMaximum(int(HEIGHT_RANGE[1] * SLIDER_SCALE))
        self._height.setValue(int(HEIGHT_DEFAULT * SLIDER_SCALE))
        self._height.valueChanged.connect(self._on_height)
        col.addWidget(self._height)
        self._drive_label = QLabel(
            f"vx 0.00  vy 0.00  yaw 0.00  h {HEIGHT_DEFAULT:.3f}")
        col.addWidget(self._drive_label)
        lay.addLayout(col)
        return box

    def _on_drive_enable(self, on):
        if not on:
            self._drive_vx = self._drive_vy = self._drive_yaw = 0.0
            # explicit stop; keep the height set-point so the stance holds
            self.node.publish_cmd(0.0, 0.0, 0.0, self._drive_height)
        elif not self._policy_on:
            self._set_status("drive on, but policy is OFF -- enable policy to move")

    def _on_pad(self, fwd, ccw):
        self._drive_vx = fwd * CMD_LIMIT[0]
        self._drive_yaw = ccw * CMD_LIMIT[2]  # pad left = rotate left (+wz)
        self._update_drive_label()

    def _on_strafe(self, val):
        # slider right = strafe right = -vy (ROS +y is left)
        self._drive_vy = -(val / 1000.0) * CMD_LIMIT[1]
        self._update_drive_label()

    def _on_height(self, val):
        self._drive_height = val / SLIDER_SCALE
        self._update_drive_label()

    def _update_drive_label(self):
        self._drive_label.setText(
            f"vx {self._drive_vx:+.2f}  vy {self._drive_vy:+.2f}  "
            f"yaw {self._drive_yaw:+.2f}  h {self._drive_height:.3f}")

    def _drive_tick(self):
        if not self._drive_enable.isChecked():
            return
        self.node.publish_cmd(
            self._drive_vx, self._drive_vy, self._drive_yaw, self._drive_height)

    # ---- telemetry slots ---------------------------------------------------
    def _on_joints(self, data):
        names, positions, velocities = data
        self._measured = dict(zip(names, positions))
        if not self._jog_built and names:
            self._build_jog_rows(names)
        for n in self._joint_names:
            if n in self._measured and n in self._jog_rows:
                self._jog_rows[n][2].setText(f"meas {self._measured[n]:+.3f}")

    def _on_imu(self, data):
        (qx, qy, qz, qw), (wx, wy, wz) = data
        # quaternion -> roll/pitch/yaw for a readable orientation line
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self._imu_label.setText(
            f"IMU:  roll {np.degrees(roll):+6.1f}°  pitch {np.degrees(pitch):+6.1f}°  "
            f"yaw {np.degrees(yaw):+6.1f}°   gyro [{wx:+.2f} {wy:+.2f} {wz:+.2f}] rad/s")

    def _on_limits(self, limits):
        self._limits = limits
        # apply to any already-built sliders
        for n, (slider, _c, _m) in self._jog_rows.items():
            if n in limits:
                slider.setMinimum(int(limits[n][0] * SLIDER_SCALE))
                slider.setMaximum(int(limits[n][1] * SLIDER_SCALE))

    def _on_service_result(self, label, success, message):
        if success:
            if label == "arm":
                self._armed = self._arm_btn.isChecked()
                self._arm_btn.setText("ARMED" if self._armed else "ARM")
                self._arm_btn.setStyleSheet(
                    "background:#c33; color:white;" if self._armed else "")
            elif label == "enable":
                self._policy_on = self._policy_btn.isChecked()
                self._policy_btn.setText("Policy ON" if self._policy_on else "Policy OFF")
            self._set_status(f"{label}: {message or 'ok'}")
        else:
            # revert the toggle button if the call failed
            if label == "arm":
                self._arm_btn.setChecked(self._armed)
            elif label == "enable":
                self._policy_btn.setChecked(self._policy_on)
            self._set_status(f"{label} FAILED: {message}")

    def _set_status(self, text):
        self._status.setText(text)


# --------------------------------------------------------------------------- #
def main():
    rclpy.init()
    sig = Signals()
    node = RosNode(sig)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    # let Ctrl-C in the terminal kill the GUI
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    win = Console(node, sig)
    win.resize(560, 720)
    win.show()
    try:
        rc = app.exec_()
    finally:
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
