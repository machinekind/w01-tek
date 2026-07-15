#!/usr/bin/env python3
"""One-command robot bring-up, run FROM the PC dev container.

    ros2 run wojtek_bringup robot [--dry-run] [--sim] [--no-viz] [--plotjuggler]

You work from one container shell (./dev.sh). This starts BOTH sides for a
session and tears them down on Ctrl-C:
  * the RPi control stack, started over SSH via its RT systemd service
    (taskset on the isolated cores + RT limits) -- this is the default and the
    only "real" way to run it; there is no non-RT path, and
  * visualization on the PC (RViz/PlotJuggler), locally in this container.

The RPi is NOT set to auto-start on boot -- the stack runs only for the duration
of this command, leaving the RPi free for other launches during development.

  --dry-run   BENCH mode: launch on the RPi WITHOUT RT and with no torque, for
              testing. --sim: MuJoCo sim locally, no RPi.

The stack comes up DISARMED -- arming stays manual (that's when torque reaches
the motors); the commands are printed once it's up. Needs SSH to the RPi from
the container (docker/compose.yaml mounts ~/.ssh; network_mode:host reaches .2).
"""
import argparse
import os
import signal
import subprocess
import sys
import time

RPI_HOST = os.environ.get("RPI_HOST", "rpi@10.42.0.2")
REMOTE_WS = os.environ.get("REMOTE_WS", "wojtek_ws")
ROS_DISTRO = os.environ.get("ROS_DISTRO", "jazzy")
SERVICE = "wojtek-robot.service"

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "LogLevel=ERROR"]
SSH = ["ssh"] + _SSH_OPTS + [RPI_HOST]              # non-interactive
SSH_TTY = ["ssh", "-tt"] + _SSH_OPTS + [RPI_HOST]   # foreground w/ signal propagation

ARMING_HINT = """
>> Stack is up and DISARMED. Use the operator console (opens with this command,
   unless --no-console): pose the robot in the boot pose (home), check it matches
   RViz, then Zero -> Stand up -> ARM with its buttons; the XY pad drives it.
   Raw service calls still work as a fallback:
     ros2 service call /wojtek/zero      std_srvs/srv/Trigger
     ros2 service call /wojtek/stand_up  std_srvs/srv/Trigger
     ros2 service call /wojtek/arm       std_srvs/srv/SetBool "{data: true}"
   Stop:   disarm, then /wojtek/lie_down -- or Ctrl-C here to tear everything down.
"""


def main():
    ap = argparse.ArgumentParser(prog="ros2 run wojtek_bringup robot",
                                 description="One-command robot bring-up from the dev container.")
    ap.add_argument("--dry-run", action="store_true",
                    help="BENCH: launch on the RPi WITHOUT RT, no torque (testing)")
    ap.add_argument("--sim", action="store_true", help="MuJoCo sim locally; no RPi, no SSH")
    ap.add_argument("--no-viz", action="store_true", help="skip RViz/PlotJuggler")
    ap.add_argument("--no-console", action="store_true",
                    help="skip the operator console GUI (arm/pose/jog/drive)")
    ap.add_argument("--plotjuggler", action="store_true", help="also open PlotJuggler")
    args = ap.parse_args()

    procs = []
    stop_remote = None  # callable to stop the RPi side on exit

    def spawn(cmd):
        p = subprocess.Popen(cmd)
        procs.append(p)
        return p

    # ---- robot side --------------------------------------------------------
    if args.sim:
        print(">> [sim] MuJoCo sim (local, in container)")
        spawn(["ros2", "launch", "wojtek_viz", "sim.launch.py", "rviz:=false"])
    elif args.dry_run:
        print(f">> [BENCH] launching on {RPI_HOST} WITHOUT RT, no torque")
        remote = (f"source /opt/ros/{ROS_DISTRO}/setup.bash && "
                  f"source ~/{REMOTE_WS}/install/setup.bash && "
                  f"ros2 launch wojtek_bringup robot.launch.py dry_run:=true")
        spawn(SSH_TTY + [remote])
        stop_remote = lambda: subprocess.run(SSH + ["pkill -f robot.launch.py"])
    else:
        # Default: start the RT stack via its systemd service (RT-pinned), for
        # this session only. Stopped again on exit.
        print(f">> starting RPi RT stack on {RPI_HOST} (systemd, RT-pinned)")
        subprocess.run(SSH + [f"sudo systemctl start {SERVICE}"])
        spawn(SSH_TTY + [f"journalctl -u {SERVICE} -f -n 20"])
        stop_remote = lambda: subprocess.run(SSH + [f"sudo systemctl stop {SERVICE}"])

    # ---- PC viz (local, in this container) ---------------------------------
    if not args.no_viz:
        pj = str(args.plotjuggler).lower()
        print(f">> launching visualization (rviz, plotjuggler={pj})")
        spawn(["ros2", "launch", "wojtek_viz", "viz.launch.py", f"plotjuggler:={pj}"])

    # ---- operator console (local GUI: arm/pose/jog/drive) ------------------
    # The manual-control surface for this session; comes up alongside viz so the
    # operator never types raw `ros2 service call`. GUI, so it needs the X mount
    # ../dev.sh sets up -- same as RViz.
    if not args.no_console:
        print(">> launching operator console (ros2 run wojtek_viz console)")
        spawn(["ros2", "run", "wojtek_viz", "console"])

    if not args.sim:
        print(ARMING_HINT)

    # ---- teardown ----------------------------------------------------------
    def shutdown(*_):
        print("\n>> stopping ...")
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        deadline = time.time() + 5
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if stop_remote is not None:
            stop_remote()
        print(">> down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
