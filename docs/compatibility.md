# Robot and software compatibility

Select a robot before selecting its policy or scene. A policy trained for one
leg family is rejected by a ROS profile for the other. Compatible names do not
certify that the model matches every physical stop or collision surface.

## Names and selectors

| Item | Stock | New legs |
|---|---|---|
| Training variant | `wojtek` | `legs_v627` |
| ROS profile | `wojtek` | `wojtek_v2` |
| URDF legs | `stock` | `legs_v627` |
| Joint conversion | `joint_map.yaml` | `joint_map_v627.yaml` |
| Default policy repository suffix | `wojtek-quiet-locomotion` | `wojtek_v2-quiet-locomotion` |
| Default sim world | Furnished stock scene | New-leg training scene, no props |
| Nominal model standing height | About 0.12 m | About 0.345 m |

The ROS profile describes the installed v6.37a legs as sharing v6.27 kinematics.
Keep that software mapping separate from the exact CAD and assembly revision of
each physical unit. See [profiles](../ros/src/wojtek_policy/wojtek_policy/robots.py)
for pinned policy revisions and [variants](../training/docs/robots.md) for mechanics.

```bash
# Host, repository root: select the training model.
./training/run.sh build --robot legs_v627
./training/run.sh train robot=legs_v627 --cfg job --resolve

# Host: select the ROS model and its pinned policy.
./ros/sim.sh --foxglove robot:=wojtek_v2
```

For actual training, use the new-leg experiment presets from the training guide;
stock presets can carry incompatible height ranges. `WOJTEK_ROBOT=wojtek_v2`
in the root `.env` selects the deployment profile. Provisioning writes it into
the robot service; launch files themselves take `robot:=`, not that environment
variable. `WOJTEK_POLICY` is a desk override and is not read by the robot launch.

## Capability status

| Capability | Stock | New legs |
|---|---|---|
| ROS mock and MuJoCo launch | Implemented | Implemented |
| Deployable flat joystick policy | Pinned export | Pinned quiet export |
| Stand/lie and named tricks | Implemented | Implemented; folded-pose fidelity has open measurements |
| Jump, get-up, biped training | Implemented task paths; check individual deployment support | Rejected |
| Terrain training | Implemented experiment family | Rejected |
| Course scores | Stock normalization | Runs may be evaluated, but scores are not comparable to stock |
| Navigation demo geometry | Stock scenes and footprint | Planner/camera assumptions need revision-specific validation |
| Calibrated physical twin | Identification tools available; no general certification | Joint range and lying collisions remain open |

An older new-leg keeper named `wojtek_v2-locomotion` uses a gait clock unavailable
in the deployment runtime. Its checkpoint is not interchangeable with the
exported quiet keeper. Check `policy_meta.json`, not a model name alone.

## Runtime configuration

These are repository defaults, not measurements of a connected unit:

| Setting | Current source value |
|---|---|
| ROS distribution | Jazzy |
| Controller manager | 100 Hz |
| Joint and IMU broadcasters | 50 Hz each |
| Magnetometer broadcaster | 100 Hz |
| Policy | From exported `ctrl_dt`; typical locomotion 50 Hz |
| CAN | CANdle SPI, 1 Mbit/s default |
| IMU | I2C enabled by default |
| Arm position tolerance | Real 0.15 rad; sim 0.35 rad |
| Manual launch boot pose | `home` |
| Installed service boot pose | `folded` |
| Installed service recording | Off in the local drop-in |

Sources: [controller YAML](../ros/src/wojtek_bringup/config/real_controllers.yaml),
[shared launch](../ros/src/wojtek_bringup/wojtek_bringup/launch_common.py),
[service drop-in](../ros/deploy/rpi/wojtek-robot-local.conf).
The configuration records Pi 3 tuning and Pi 4/Pi 5 measurements. Confirm
the board actually installed before copying CPU or performance expectations.
