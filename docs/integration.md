# Integrate with Wojtek

Start against ROS MuJoCo with a matching model and policy. The supported shared
interfaces are ROS topics and services. They do not provide authenticated
clients, exclusive task ownership or a versioned action API.

## Choose the level

| Input | Interface | Meaning |
|---|---|---|
| Body velocity | `/cmd_vel`, `geometry_msgs/Twist` | `linear.x/y`: m/s; `angular.z`: rad/s |
| Stance height | Same Twist, `linear.z` | Metres, only for policies with the height command |
| Short motion command | `/wojtek/nav_command`, `std_msgs/String` | `forward`, `left`, `right` or `stop`; timed velocity commands |
| Local destination | `/wojtek/nav/goal`, `geometry_msgs/PoseStamped` | Position in a TF-resolvable frame; `odom` is the simplest |
| Image target | `/wojtek/nav/pixel_goal`, `geometry_msgs/PointStamped` | Pixel coordinates and source-image stamp; see nav contract |
| Language task | `/wojtek/vlm/instruction`, `std_msgs/String` | Replaces the brain's current instruction |
| Cancel navigation | `/wojtek/nav/cancel`, `std_msgs/Empty` | Ends goto, pixel tracking, the brain task and text-command motion |
| Pose/arm control | `/wojtek/zero`, `/stand_up`, `/lie_down`, `/arm` under `/wojtek` | See operating-state rules before calling |

Do not publish motor-controller commands from an application. The policy converts
commands into absolute joint targets; `real_io_node` applies the boot offsets and
arm gate. [Architecture](ros-architecture.md) documents those internal contracts.

## Run a complete waypoint client

Start a stock simulation with navigation, but without the brain:

```bash
# Host, repository root; set up a compatible policy first.
./ros/sim.sh --foxglove model_xml:=scene_nav.xml leg_odom:=true nav:=true
```

Use the console to zero, stand and arm. Leave other controllers idle. In a second
host terminal run `./ros/dev.sh`, then:

```bash
ros2 run wojtek_nav goal_client --forward 0.5 --left 0 --timeout 30
```

The [client source](../ros/src/wojtek_nav/wojtek_nav/goal_client.py) shows the whole
exchange: discover subscribers, cancel any old task, wait out the cancel
hold-off, read `odom→base_link`, freeze the relative displacement into `odom`,
refresh the same goal every second, observe status, and cancel on exit.
It never zeroes, stands or arms the robot itself.

For the opt-in [command mux](command-ownership.md), add `command_mux:=true` to
the simulation launch and use:

```bash
ros2 run wojtek_nav goal_client --autonomy --forward 0.5 --timeout 30
```

`--autonomy` requests mux enable **after** the client's initial cancellation and
hold-off. A manual owner or unsettled cancellation can refuse enable; no goal
is then sent. This flag never arms the robot. Enabling the mux beforehand is
insufficient because the client's initial cancel inhibits it again.

Success is exit code 0 after `reached` and a position check within 0.15 m of the
target. A `blocked` status immediately fails the client and cancels the waypoint;
it does not wait for the obstacle to clear. Cancelled navigation, timeout or a
disconnected graph also returns failure. Ctrl-C requests cancel and returns 130.
Cancellation can also be sent separately:

```bash
ros2 run wojtek_nav goal_client --cancel
```

After its initial hold-off, the client exits on any received cancel or on `idle`
after movement starts; resuming needs a new invocation. Cancel messages have no
request ID, so a delayed initial cancel also ends it. After `reached`, the client
stops refreshing and waits for the position check, allowing the subsequent
`idle` and a delayed TF update.

Status uses reliable delivery. The example deliberately subscribes with volatile
durability so a retained result from a previous session is not mistaken for this
one. Observers wanting the latest displayed status can use transient-local QoS.
Sensors use sensor-data QoS; inspect `ros2 topic info -v TOPIC` before subscribing.

## Send a short motion command

The simulation always starts `text_commander`. The web console's text-command
panel and ROS clients use the same `/wojtek/nav_command` interface. `forward`
requests 0.3 m/s; `left` and `right` turn at +0.5 and −0.5 rad/s by default.
Parameters `v_forward`, `w_turn` and `command_timeout` configure these defaults.
This path has no camera or costmap check and never arms the robot.

After standing and arming in simulation, leave other sources idle and run inside
the ROS container:

```bash
ros2 topic pub --once /wojtek/nav_command std_msgs/msg/String '{data: forward}'
ros2 topic pub --once /wojtek/nav_command std_msgs/msg/String '{data: stop}'
```

Motion is published at 20 Hz for at most 2 seconds after the last command;
repeat the command to continue. `stop`, an unknown command, timeout or
`/wojtek/nav/cancel` clears it, emits one zero if active, then goes silent.
With `command_mux:=true`, output goes to `/wojtek/cmd_vel/text`; explicitly
[enable autonomy](command-ownership.md#autonomy-and-cancellation) before sending.

## Timing and ownership

Goto expires a goal after 3 seconds without refresh. A goal in `base_link` is
transformed at its message stamp; repeatedly sending a new relative goal moves
the destination. Freeze it in `odom` for a fixed waypoint, as the example does.
ROS simulation currently runs against wall time; do not enable `use_sim_time`
without a clock source. Synchronize clocks for offboard camera/TF consumers.

The current default graph shares `/cmd_vel` between controllers. It has no
exclusive lease. Keep one writer active, cancel the brain before manual
takeover, and stop old publishers before starting a new client. Disarming closes
the joint-target gate; it does not by itself cancel autonomy. The policy's
command timeout stops stale velocity commands after 0.5 seconds; it is not a
substitute for source ownership or a physical torque-off mechanism.

With the shared launch's `command_mux:=true`, manual takeover and disarm cancel
navigation and inhibit autonomy. Route external velocity commands to
`/wojtek/cmd_vel/external`; publishing directly on `/cmd_vel` bypasses the mux.

Goal/status topics have no request ID. This example assumes one client; matching
a status word cannot distinguish concurrent goals. Language status is JSON in a
String with `instruction`, `step`, `t`, `action` and action-specific fields.
`action=finished` carries a `result`. Instruction text and wall-clock time are
not unique request identifiers. Build a versioned action adapter before relying
on concurrent or retryable task submission.

## Add another input or model

A velocity source should publish at a steady rate while active, send a zero
velocity when it stops, then go silent. Keep the policy's height channel intact.
The runtime clips commands to its trained box; inspect the policy metadata for
the allowed range. A depth-aware application should use the existing pixel
resolver rather than invent a pixel-to-distance conversion.

For language tasks, use the [brain contract](../ros/src/wojtek_nav/README.md).
An alternative model must satisfy the structured answer and verification
contracts and pass the same pointing/navigation evaluation. The Deck's external
detector protocol is documented in its [package guide](../ros/src/wojtek_deck/README.md).
For network boundaries and observer-only connections, see [network access](network-access.md).
