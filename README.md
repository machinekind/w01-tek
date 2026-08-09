# Wojtek

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

- [Agent guide](CLAUDE.md)
- [Training configuration reference](training/docs/configuration.md)
- [ROS workspace guide](ros/README.md)

## License

Apache-2.0; see [LICENSE](LICENSE) and the attribution notes in
[NOTICE](NOTICE). One exception: the IMU firmware directories under
`ros/src/bmi160_serial_hardware_interface/` and
`ros/src/bmx160_serial_hardware_interface/` contain GPL-licensed AHRS
code; the `LICENSE.md` in each firmware directory is the authoritative
statement of what applies there.
