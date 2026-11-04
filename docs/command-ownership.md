# Command ownership

The default ROS graph lets several sources publish `/cmd_vel`. Its consumer sees
the latest message; there is no ownership in that path. The command mux is an
**opt-in** alternative. Control logic, ROS routing and simulated handover have
been tested; physical handover still needs validation on the assembled unit.
See the [validation record](validation.md) for the tested scope.

The mux selects one source, expires stale input and inhibits autonomy on manual
takeover. It does not arm/disarm drives, enforce motor limits or authenticate
publishers. Those remain separate functions.

## Ownership rules

Each source has a fixed priority. Higher numbers win; a lower-priority source
cannot preempt a fresh owner. The same owner refreshes its lease with every
accepted message. Equal-source messages are not separate operators: use exactly
one publisher per input topic.

| Source | Priority | Mode | Input topic |
|---|---:|---|---|
| Pad | 100 | Manual | `/wojtek/cmd_vel/pad` |
| Deck | 90 | Manual | `/wojtek/cmd_vel/deck` |
| Console | 80 | Manual | `/wojtek/cmd_vel/console` |
| External client | 70 | Manual | `/wojtek/cmd_vel/external` |
| VLM brain turns | 30 | Autonomous | `/wojtek/cmd_vel/brain` |
| Navigation driver | 20 | Autonomous | `/wojtek/cmd_vel/nav` |
| Text commander | 10 | Autonomous | `/wojtek/cmd_vel/text` |

All inputs are `geometry_msgs/msg/Twist`. The default lease is 0.5 seconds from
receipt, measured with a monotonic clock. Invalid/non-finite messages are
discarded and do not refresh a lease. Zero velocity is still a command and keeps
ownership; idle sources must eventually stop publishing.

On expiry, the mux emits one zero-velocity command, preserving the last emitted
stance-height channel (`linear.z`), then goes silent. It does not retain commands
from losing sources. A different source must send again to acquire ownership.
If it arrives after expiry but before the next tick, the stop is emitted before
its new command. The policy's own freshness guard remains necessary if the mux
process itself stops.

## Autonomy and cancellation

Autonomy starts inhibited. Enable it explicitly when no manual lease is active:

```bash
ros2 service call /wojtek/autonomy_enable std_srvs/srv/SetBool '{data: true}'
```

The service refuses enable while a cancel is pending and for 0.5 seconds after
the last cancel was published or received. Check `success` and retry after that
interval; a rejected request does not enable autonomy.
This quiet period is not a cancel acknowledgment: `Empty` carries no request ID.
A delayed cancel arriving after enable still inhibits autonomy, including a
delayed echo of the mux's own cancel. Wait for cancellation to settle and retry
the explicit enable if that happens.

Any accepted manual acquisition publishes `/wojtek/nav/cancel` and inhibits
autonomy. Manual release or timeout does **not** re-enable it. A new instruction
also does not enable it automatically. To resume, stop the manual source, wait
for its lease to expire, enable autonomy and issue the intended new task.

Switch the Qt console's drive toggle off first: it keeps publishing even with
zero velocity. After releasing ROS pad sticks, allow roughly 3 seconds from the
last nonzero input: 0.5 seconds of input timeout, a 2-second zero burst, then the
0.5-second mux lease. Check that `owner` is null before enabling autonomy.

Disable autonomy and cancel its current task with:

```bash
ros2 service call /wojtek/autonomy_enable std_srvs/srv/SetBool '{data: false}'
```

The mux also inhibits autonomy on `/wojtek/nav/cancel` (`std_msgs/msg/Empty`),
without echoing that message. A manual owner continues to operate. The brain
uses this topic when finishing or replacing a task, so wait for that cancellation
to settle before enabling the next/replacement task. The text commander also
clears its timed command on cancel, so enabling autonomy cannot resume that old
command. An expired autonomous lease
alone does not disable the gate; a newly received command may acquire ownership
while autonomy remains enabled.

With the shared launch's `command_mux:=true`, `real_io_node` publishes
`/wojtek/nav/cancel` when disarmed. This also inhibits the mux's autonomy gate.
That connection is off in the default graph. A manually assembled launch must
enable `cancel_navigation_on_disarm` on `real_io_node`, or explicitly cancel
navigation during disarm. The mux cannot observe service calls by subscribing
to topics. Disarm does not remove torque; see [Operations](operations.md).

## Use the shared simulation launch

After [simulation setup](simulation.md), start a separate simulator session:

```bash
./ros/sim.sh --foxglove command_mux:=true
```

The shared launch routes its Deck and pad publishers; the simulation launch
also routes its console, text commander and VLM brain. Navigation uses its own
input. The mux becomes the only publisher on `/cmd_vel` within this launch.
The option defaults to false and does not change existing sessions.

In the container, inspect `ros2 topic info -v /cmd_vel` before driving. Only
`wojtek_command_mux` should publish there. Then validate:

1. Stand and arm using the operator interface; drive from one manual source.
2. Start a lower-priority source and confirm it cannot take ownership.
3. Release/stop the owner, wait for its zero burst and 0.5-second lease expiry,
   then send a new command from the other source.
4. Enable autonomy with the service, send an intended task, and take over
   manually. Confirm cancellation and `autonomy_enabled: false` in mux status.
5. Disarm and confirm autonomy remains inhibited. Re-arming must not resume the
   old autonomous task.

For navigation, add `model_xml:=scene_nav.xml leg_odom:=true nav:=true` to the
launch. Clients that cancel an old task before starting must enable autonomy
**after** that cancellation; enabling first would be undone by the cancel.

The robot launch supports the same option, but an offboard console, brain or
client is outside that launch and still needs an explicit source remap. Validate
the complete graph in simulation before choosing this option on hardware.

## Inspect the mux without driving a robot

Use an unused ROS domain in every test shell, separate from the robot's domain.
For example, after sourcing the workspace:

```bash
export ROS_DOMAIN_ID=73
ros2 launch wojtek_teleop command_mux.launch.py
```

The standalone launch publishes on `/wojtek/cmd_vel/selected`, leaving the normal
`/cmd_vel` consumer disconnected. In another shell with the same test domain:

```bash
ros2 topic echo /wojtek/cmd_vel/selected
```

Publish a test command in a third shell, then stop it with Ctrl-C:

```bash
ros2 topic pub --rate 20 /wojtek/cmd_vel/external geometry_msgs/msg/Twist \
  '{linear: {x: 0.1, z: 0.125}}'
```

Expect the command while input arrives, one zero-velocity command after expiry,
then silence. Inspect the latched JSON status:

```bash
ros2 topic echo /wojtek/command_mux/status \
  --qos-durability transient_local --qos-reliability reliable
```

Status schema 1 contains `owner` (source or null), `autonomy_enabled`, `reason`,
`timeout_s` and the rejection count. Status changes are published; command output
uses `publish_hz` (20 Hz by default). `source_timeout_s` sets the lease duration.

## Integrate every writer

Remap each publisher's `/cmd_vel` to its dedicated mux input:

| Publisher | Remap to |
|---|---|
| `gamepad_teleop` | `/wojtek/cmd_vel/pad` |
| `deck_gateway` | `/wojtek/cmd_vel/deck` |
| Web or Qt console | `/wojtek/cmd_vel/console` |
| Your client | `/wojtek/cmd_vel/external` |
| `vlm_brain_node` | `/wojtek/cmd_vel/brain` |
| `goto_node` | `/wojtek/cmd_vel/nav` |
| `text_commander` | `/wojtek/cmd_vel/text` |

For a standalone publisher, an explicit ROS remap works:

```bash
ros2 run wojtek_teleop text_commander --ros-args \
  -r /cmd_vel:=/wojtek/cmd_vel/text
```

The `gamepad.launch.py`, navigation `costmap.launch.py` and `brain.launch.py`
files also accept `cmd_vel_topic:=` for their respective publisher. For example,
an offboard brain uses:

```bash
ros2 launch wojtek_nav brain.launch.py cmd_vel_topic:=/wojtek/cmd_vel/brain
```

In a Python launch, put the remap on that publisher's `Node`, for example:

```python
Node(
    package="wojtek_teleop", executable="gamepad_teleop",
    remappings=[("/cmd_vel", "/wojtek/cmd_vel/pad")],
)
```

Once all writers are routed, either remap the policy's `/cmd_vel` subscription
to `/wojtek/cmd_vel/selected`, or connect the mux output to `/cmd_vel`:

```bash
ros2 launch wojtek_teleop command_mux.launch.py output:=/cmd_vel
ros2 topic info -v /cmd_vel
```

For the latter setup, `/cmd_vel` must have exactly one publisher:
`wojtek_command_mux`. Check each source topic for exactly its intended publisher.
A writer left on `/cmd_vel` bypasses arbitration. Avoid a global launch remap
that merges several publishers onto one source or sends mux output back to an
input. Validate the wiring first in ROS MuJoCo, including manual takeover,
stale input, cancellation and re-enabling autonomy.

## Limits and tests

Command subscriptions use reliable, volatile `KEEP_LAST(1)` QoS. This limits
queued data and excludes retained historical commands. `Twist` has no timestamp
or task identifier, so receipt freshness cannot detect an old network packet
arriving late. The no-replay rule applies to commands stored inside this mux.
Adding task/epoch provenance requires a different input contract.

Sources share the graph's network trust. A client can spoof a source topic or
bypass the mux unless deployment access controls prevent it. This module is not
an emergency stop or proof of a physical stopping distance.

Run the model-free scenarios from the repository root:

```bash
PYTHONPATH=ros/src/wojtek_teleop training/.venv/bin/python -m pytest \
  ros/src/wojtek_teleop/test/test_command_mux.py -q
```

The suite checks concurrent manual writers, priority, lease renewal/expiry,
discarded commands, manual takeover, explicit enable/disable, cancellation
echoes, height preservation and invalid input. It does not exercise ROS
transport, a physical robot or timing under load.
