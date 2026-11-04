# Wojtek documentation

Wojtek is a twelve-actuator quadruped with four-bar legs. A ROS 2 Jazzy stack
runs its learned walking policy on the robot; MuJoCo runs the same control graph
on a development machine. MJX and Brax provide the separate training pipeline.
Vision-language navigation turns an instruction and camera image into local
waypoints. It is an experimental autonomy layer, not a complete navigation system.

## Start with your task

| I want to… | Start here |
|---|---|
| Drive a virtual robot | [Simulation](simulation.md) |
| Operate with a pad or Steam Deck | [Operations](operations.md) |
| Recover from a fault or inspect a run | [Troubleshooting](troubleshooting.md) |
| Connect my software | [Integration](integration.md) |
| Coordinate manual and autonomous control | [Command ownership](command-ownership.md) |
| Understand the node graph | [ROS architecture](ros-architecture.md) |
| Choose the right legs, model and policy | [Compatibility](compatibility.md) |
| Train and export a policy | [Training quickstart](../training/README.md) |
| Deploy or roll back a policy | [Policy lifecycle](../training/docs/policy-lifecycle.md) |
| Try VLM navigation | [Navigation guide](../ros/src/wojtek_nav/README.md) |
| Build or modify the robot | [Hardware](hardware.md), then [commissioning](commissioning.md) |
| Maintain an assembled unit | [Maintenance](maintenance.md) |
| Calibrate a digital twin | [Twin scope and workflow](../training/docs/digital-twin.md) |
| Observe over a network | [Network access](network-access.md) |

Start in simulation. A physical unit additionally needs a confirmed hardware
revision, wiring, zero pose and calibration. The [hardware evidence register](hardware-evidence.md)
lists missing build inputs; the repository does not yet contain a complete
manufacturing pack or a measured twin of every revision.

## References and evidence

- [Training configuration](../training/docs/configuration.md): Hydra settings,
  observations, rewards, randomization and evaluation commands.
- [Robot variants](../training/docs/robots.md): model import, mechanics and results.
- [System identification](../training/docs/sysid.md): actuator fitting and limits.
- [Simulation test contract](sim-test-contract.md): what a sim run demonstrates.
- [Camera measurements](perception/d435-noise.md): dated D435 observations.
- [Guided RL course](../learning/wojtek_rl_course/README.md): the Polish Colab course.
- [Original thesis](comparative-studies-on-walking-control-algorithms-for-the-quadruped-robot-4barbot.pdf):
  the 2025 design and original controllers, not the current build specification.
- [Research comparison](research/isaac-vs-mujoco.md) and
  [MJWarp report](plans/mjwarp-phase0-report.md): historical decisions and measurements.

Experimental projects carry their own status and setup in their directory under
`experiments/`. They are not deployment dependencies. A result from one route
does not validate another route with different sensors or obstacle checks.

## Contribute

Use [CONTRIBUTING](../CONTRIBUTING.md) for checks and documentation conventions.
The [backlog](plans/documentation-backlog.md) tracks unfinished acceptance work;
the [validation record](validation.md) states what this documentation change
actually exercised. Hardware measurements and human walkthroughs remain separate
from code tests.
