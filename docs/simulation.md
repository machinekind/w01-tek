# Run Wojtek in simulation

Use ROS MuJoCo to test the deployed control stack. Use MJX training to learn or
evaluate policies in batches. The training room app and Isaac prototype are
separate applications; neither exercises the full ROS bring-up.

| Goal | Path |
|---|---|
| Test interfaces without physics | `./ros/sim.sh hw:=mock` |
| Operate the robot against physics | `./ros/sim.sh --foxglove` |
| Train/evaluate a walking policy | [Training](../training/README.md) |
| Inspect a model with joint sliders | [Variant viewer](../training/docs/robots.md) |
| Explore a scanned room | [Training demo](../training/demo/README.md) |
| Explore Isaac rendering | [Prototype status](../training/demo/isaac/README.md) |

## Prepare the host

Run host commands from the repository root. Install Docker with Compose.
Linux uses host networking; macOS uses Docker Desktop's bridge and published
ports. Browser consoles work on both. Linux can pass a pad to `joy`; on macOS
pair it with the Mac and use the browser. RViz on macOS additionally needs
XQuartz; start with Foxglove/browser mode to avoid that dependency.

Choose a policy route before launching:

- **Keeper access:** copy `.env.example` to `.env` if it does not already exist.
  Set `HF_ORGANIZATION` and authenticate with a token that can read its keepers.
  The launch resolves the selected profile's pinned revision.
- **Local artifact:** obtain `policy.npz` and `policy_meta.json` from your own
  [training/export run](../training/docs/policy-lifecycle.md). Copy the pair to
  `ros/policies/local-example/`. The container sees that directory as
  `/ros2_ws/policies/local-example`; pass that path through `policy:=`.

The public source checkout contains no trained weights. Mock mode also resolves
a policy because it runs the same policy node. An untrained random network is
not a working locomotion example.

```bash
./ros/build.sh
./ros/sim.sh --foxglove
# Local-artifact alternative:
./ros/sim.sh --foxglove policy:=/ros2_ws/policies/local-example
```

The first build downloads ROS dependencies and compiles the workspace. The
wrapper rebuilds changed packages on later runs. Use `--no-build` only when the
container overlay already matches the source. No host ROS installation is needed.

## Stand and drive

Open `http://localhost:8080` for the operator console or
`http://localhost:8090` for the Deck panel. Foxglove connects to
`ws://localhost:8765`. Follow **Zero → Stand → Arm**, then drive with the console
or pad. Disarm and lie down before ending the session with Ctrl-C.

The default `home` start is the simplest first run. `boot_pose:=folded` rehearses
startup, but the new-leg folded pose currently differs from the exported model;
see [twin limits](../training/docs/digital-twin.md).

In a second host terminal, `./ros/dev.sh` enters the same container. Inspect:

```bash
ros2 control list_controllers
ros2 topic hz /wojtek/joint_states_abs
ros2 topic echo /sim/rtf
ros2 topic info -v /cmd_vel
```

Expect active controllers, changing absolute joint states and a real-time factor
near 1 when the host keeps up. The absolute-state topic includes separate passive
joint messages, so its message count is not simply the broadcaster rate. Low
RTF means timing and camera results need interpretation; it is not a valid
real-time performance result. Disable the camera or visualizer to isolate load.

## Variants and reduced sessions

```bash
./ros/sim.sh --foxglove robot:=wojtek_v2
./ros/sim.sh --no-viz hw:=mock camera:=false console:=none deck:=false
./ros/sim.sh --foxglove camera:=false
```

The new-leg profile selects its own model, joint map and policy. Do not override
it with `scene_nav.xml`, which contains the stock robot. Mock mode echoes joint
commands; it cannot demonstrate walking, falls, contacts or a camera image.

The stock profile's default `scene_sim.xml` adds six props for the Deck detector:
a ball, fire hydrant, stop sign, traffic light, clock and person. They have solid
collision geometry. For the stock robot on an empty checkerboard floor, use:

```bash
./ros/sim.sh --foxglove robot:=wojtek model_xml:=scene_mjx.xml
```

An empty `model_xml:=` selects the profile's default scene, so it does not remove
the stock props. The new-leg profile has its own scene; keep that default.

## Navigation

Configure Ollama and `VLM_URL` using the [navigation guide](../ros/src/wojtek_nav/README.md).
Then run the stock navigation scene:

```bash
./ros/sim.sh --foxglove model_xml:=scene_nav.xml \
  leg_odom:=true nav:=true vlm:=true
```

Stand and arm before submitting a task through the console. Stop autonomy before
manual driving. A local waypoint example needs `nav:=true` but no VLM server;
see [integration](integration.md).

For explicit source priority and manual takeover, use the opt-in
[command mux](command-ownership.md). Its shared launch option routes the built-in
publishers; external clients still need their own source remap.

## Restart and troubleshoot

The current hardware-plugin simulation has no `/sim/reset` service. End the
launch and run it again to reset the world. Ctrl-C leaves the development
container available for reuse; `docker compose -f ros/docker/compose.yaml down`
stops the Linux Compose session. On macOS include the same
`-f ros/docker/compose.mac.yaml` override used to start it.

An empty policy reference means configuration/access is missing. A missing ROS
package often means a stale overlay. A blank camera needs physics and a working
EGL renderer. See [troubleshooting](troubleshooting.md) and the
[simulation contract](sim-test-contract.md) before treating a sim result as
physical validation.
