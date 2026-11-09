# Wojtek ROS workspace

ROS 2 Jazzy controls the robot and runs the same policy/I/O stack in MuJoCo.
The PC uses Docker; the robot uses native Ubuntu with RT configuration.
CycloneDDS domain 42 connects the Linux PC and robot.

| Task | Guide |
|---|---|
| Choose hardware, profile and policy | [Compatibility](../docs/compatibility.md) |
| Run locally without hardware | [Simulation](../docs/simulation.md) |
| Drive with a pad or Steam Deck | [Operations](../docs/operations.md) |
| Connect your software | [Integration](../docs/integration.md) |
| Understand the node graph | [Architecture](../docs/ros-architecture.md) |
| Diagnose or recover a run | [Troubleshooting](../docs/troubleshooting.md) |
| Configure visual navigation | [Navigation](src/wojtek_nav/README.md) |

All host commands below run from the repository root.

## PC setup

Install Docker with Compose. On first setup, copy `.env.example` to `.env` and
set `HF_ORGANIZATION` plus credentials for private keeper access. Existing files
should be edited, not overwritten. Alternatively use a local exported policy
pair; see [policy lifecycle](../training/docs/policy-lifecycle.md).

```bash
./ros/build.sh
./ros/sim.sh --foxglove
```

The simulation loads the stock profile's pinned policy. It opens the web console
on `http://localhost:8080` and Deck panel on `http://localhost:8090`.
Foxglove connects to `ws://localhost:8765`. Stand/arm/drive using the
[operator sequence](../docs/operations.md); launch does not arm the robot.

For ROS commands, open a container shell:

```bash
./ros/dev.sh
# Inside the container:
ros2 launch wojtek_pc sim.launch.py --show-args
```

The workspace is already sourced. Rebuild the image when system/package
requirements change. Linux supports native input devices and direct DDS
networking. macOS simulation uses published browser/Foxglove ports; use browser
pad input. The supplied cable-network setup uses Linux NetworkManager.

Useful sim arguments:

```bash
./ros/sim.sh hw:=mock rviz:=false console:=none  # graph/lifecycle, no physics
./ros/sim.sh --foxglove robot:=wojtek_v2         # new-leg profile and matching pin
./ros/sim.sh --foxglove telemetry:=true         # CPU and policy timing topics
./ros/sim.sh --rviz gamepad:=true               # Linux joy driver
./ros/sim.sh --foxglove command_mux:=true       # opt-in command ownership
```

The mux starts with autonomy inhibited; see [command ownership](../docs/command-ownership.md)
for enabling it, handing over and routing external clients.

`WOJTEK_POLICY` in the repo-root `.env` overrides the sim policy reference.
The robot ignores it. A launch accepts `policy:=<org/name@commit>` or a local
directory visible **inside** the container. `ros/policies/` is mounted at
`/ros2_ws/policies/`; a host home-directory path is not automatically mounted.
Missing private access is not a Docker failure: supply an authorized reference
or export a compatible local policy through the training workflow.

## Provision the robot

Confirm the physical revision and power arrangement before provisioning or
powered tests. The [hardware guide](../docs/hardware.md) records commissioning
requirements. For the image and network, see [RPi image](deploy/rpi/IMAGE.md)
and [network setup](deploy/rpi/README.md).

From a Linux operator PC:

```bash
./ros/deploy/pc/setup-net.sh
./ros/deploy/rpi/flash-card.sh /dev/sdX  # replace with the verified target card
./ros/deploy.sh --provision
```

Provisioning requires the settings in `ros/.env.example`, including the Ubuntu
Pro token for the RT kernel and AP password. Shared policy/profile settings live
in the repo-root `.env`. `WOJTEK_ROBOT=wojtek_v2` selects new legs;
`--provision` writes that selection into the service drop-in.

Provisioning installs the RT service and CM watchdog. Without `--enable-robot`,
it disables robot boot autostart, including when previously enabled. Add that
flag only for a commissioned robot intended to start its stack on boot.
The command may reboot the RPi after kernel changes.

Routine deployment:

```bash
./ros/deploy.sh --no-restart          # sync and build without a service restart
./ros/deploy.sh --policy <reference> # fetch/sync a policy override
./ros/deploy.sh                     # return to the profile's pinned policy
```

Deployment builds `--packages-up-to wojtek_bringup`, which includes gamepad,
Deck gateway and camera dependencies. It also fetches/syncs policy and detector
assets. Restart occurs only when requested and the robot service is enabled;
an active but disabled service is not automatically restarted. Inspect its state
instead of assuming that a completed deploy changed a running process.

## Run the physical robot

After the physical boot pose and commissioning checks in the
[operator runbook](../docs/operations.md):

```bash
./ros/dev.sh
# Inside the container:
ros2 run wojtek_bringup robot --web-console --foxglove
```

The wrapper starts the configured RPi service over SSH and local visualization.
Ctrl-C stops both. The service starts disarmed, but the hardware can already
hold torque. Stand/lie ramps also move while disarmed.

`--sim` runs locally. `--dry-run` is a separate non-RT bench launch with drive
commands suppressed. `--gamepad` documents the robot-paired pad path; the
installed service already includes that driver. `--vlm` starts the PC-side brain,
but the robot service must separately include its camera and `nav:=true`.
`--policy` and launch tokens affect sim/bench launches; they do not rewrite the
RT service's launch arguments. Use deployment/service configuration for that.

## Recording and live diagnostics

On the robot, inspect the effective unit:

```bash
systemctl cat wojtek-robot.service
```

The base unit enables bags, telemetry and Foxglove. The installed
`10-local.conf` replaces its command. The drop-in disables bags.
It enables telemetry, the Foxglove bridge on port 8765, Deck, camera and gamepad.
Check the board's resource budget before recording a planned run.
Add `bag:=true bag_cpus:=0,1` to the effective launch to record the run.
Manual launches leave recording, telemetry and Foxglove disabled by default.

`/wojtek/sysinfo` reports computer health; `/wojtek/policy_timing` reports control
tick timing. Import the [Foxglove layout](foxglove/layouts/README.md). Robot bags
default to `~/wojtek_bags/run_<timestamp>`; PC-container bags use the persistent
`wojtek_bags` Docker volume. See [troubleshooting](../docs/troubleshooting.md) for
inspection and recovery.
