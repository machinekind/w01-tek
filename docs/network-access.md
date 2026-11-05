# Network access

The normal robot network is a trusted operator network. DDS domain 42 separates
discovery; it does not authenticate a controller. The console on port 8080 and
Deck gateway on 8090 can issue commands and services. Their HTTP/WebSocket
connections are not an authorization boundary. Keep them off public networks.

| Connection | Purpose | Control access |
|---|---|---|
| DDS between robot and PC | ROS topics/services | Participants can publish and call available services |
| Operator console, 8080 | Browser controls | Commands and services |
| Deck gateway, 8090 | Pad, camera and panel | Commands and services |
| Standard Foxglove bridge, 8765 | ROS inspection | Bridge defaults may allow publish/services/parameters |
| Observer bridge, 8766 | Restricted inspection | Subscriptions, graph and selected model assets only |
| Ollama endpoint | VLM requests | Model inference, not a direct motor interface |

See [robot networking](../ros/deploy/rpi/README.md) for the dock/AP configuration.
The robot's mobile AP has no internet; policy and detector assets are copied
before a run. Colour JPEGs go to the offboard VLM; raw depth stays with the
navigation pipeline in the core deployment.

## Observer-only connection

Inside the ROS container, run:

```bash
ros2 launch wojtek_pc observe.launch.py
```

This starts a separate bridge on loopback port 8766. Its
[configuration](../ros/src/wojtek_pc/config/observer.yaml) removes client-publish,
service and parameter capabilities, and restricts assets to robot description
meshes/images. These are the bridge's documented
[capability controls](https://github.com/foxglove/foxglove-sdk/blob/main/ros/src/foxglove_bridge/README.md).

For a Linux host-network container, connect locally to `ws://localhost:8766`.
For a remote Linux machine, forward the port through SSH:

```bash
ssh -N -L 8766:127.0.0.1:8766 <observer-host>
```

The Mac Compose bridge does not expose this loopback socket. Use a dedicated
container/network setup for remote observation rather than widening the bind
address silently. The normal Mac simulation UI uses the published 8765 port.

Confirm with `ros2 param get /wojtek_observer capabilities` and inspect the
WebSocket server-info message. It must contain neither `clientPublish`,
`services` nor parameter capabilities. Subscribing to an allowed topic should
work; advertising a command topic or making a service call should be refused.
Record the installed bridge version with any acceptance result.

This profile restricts this bridge only. Another unrestricted bridge, DDS
participant or command gateway on the same network still has its own access.
It supplies no user authentication; SSH controls remote access to its socket.

## VLM server access

Keep the model server on loopback and tunnel it to the workstation. On Linux,
the host-network ROS container uses the workstation's loopback. On macOS it
uses `host.docker.internal`. The [VLM setup](../ros/src/wojtek_nav/README.md)
gives the commands. Do not put server credentials or private host identities
in checked-in examples or diagnostic bundles.
