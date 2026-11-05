# Wojtek

Wojtek is a twelve-actuator quadruped with four-bar legs, ROS 2 control and a
learned walking policy. This repository contains the robot software, MuJoCo
simulation, MJX/Brax training pipeline and experimental visual navigation.

**[Start with the documentation](docs/README.md).**

| Task | Guide |
|---|---|
| Run it on your computer | [Simulation](docs/simulation.md) |
| Drive with a pad or Steam Deck | [Operations](docs/operations.md) |
| Integrate your software | [ROS architecture](docs/ros-architecture.md) · [Client tutorial](docs/integration.md) |
| Choose stock or new legs | [Compatibility](docs/compatibility.md) |
| Train and deploy a policy | [Training](training/README.md) · [Policy lifecycle](training/docs/policy-lifecycle.md) |
| Build and commission hardware | [Hardware](docs/hardware.md) · [Commissioning](docs/commissioning.md) |
| Explore VLM navigation | [Navigation](ros/src/wojtek_nav/README.md) |
| Validate a digital twin | [Twin workflow](training/docs/digital-twin.md) |

ROS simulation runs the robot's control graph against a simulated hardware
plugin. It needs a compatible policy artifact; weights are not included in the
public checkout. The new legs use training variant `legs_v627` and ROS profile
`wojtek_v2`. Check the compatibility guide before mixing models and policies.

Hardware construction and physical twin validation have
[explicit missing inputs](docs/hardware-evidence.md). The docs distinguish
implemented behavior from measured results and remaining work.

## Origins

The robot began as **4BarBot**, a quadruped designed in The Mechatronics
and Robotics Laboratory of the Wrocław University of Science and
Technology, within the Mechatronics and Theory of Mechanisms Team.
[Jakub Delicat](https://github.com/delipl)'s master thesis,
[*Comparative studies on walking control algorithms for the quadruped robot
4BarBot*](docs/comparative-studies-on-walking-control-algorithms-for-the-quadruped-robot-4barbot.pdf)
(2025), covers the robot and its first walking controllers, and his
[delipl/quadruped_ros2](https://github.com/delipl/quadruped_ros2) repository
is the ROS 2 stack this workspace grew out of.

## Documentation

- [Documentation home](docs/README.md)
- [Contributing and checks](CONTRIBUTING.md)
- [Validation record](docs/validation.md)
- [Agent guide](AGENTS.md)
- [Training configuration reference](training/docs/configuration.md)
- [Guided RL notebook](learning/wojtek_rl_course/README.md) — Colab, step by step: how Wojtek
  is trained, and your policy against the deployed keeper
- [ROS workspace guide](ros/README.md)
- [Documentation backlog](docs/plans/documentation-backlog.md) — prioritized improvements and acceptance criteria

## License

Apache-2.0; see [LICENSE](LICENSE) and the attribution notes in
[NOTICE](NOTICE). One exception: the IMU firmware directories under
`ros/src/bmi160_serial_hardware_interface/` and
`ros/src/bmx160_serial_hardware_interface/` contain GPL-licensed AHRS
code; the `LICENSE.md` in each firmware directory is the authoritative
statement of what applies there.
