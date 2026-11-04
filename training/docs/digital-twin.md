# Build and validate a digital twin

Wojtek has a CAD-derived physics model, a ROS simulation using the deployed
control stack, simulated sensors and actuator identification. A calibrated twin
requires evidence that these match one identified physical robot.

| Capability | Current support |
|---|---|
| Run the ROS stack against mock or MuJoCo hardware | Implemented; see [simulation](../../docs/simulation.md). |
| Fit actuator parameters from an air-rig bag | Implemented; see [sysid](sysid.md). Variant selection chooses the matching coordinate map. |
| Render a scanned room with the exported policy | Implemented demo; geometry and camera/planner assumptions are stock-sized. |
| Reproduce a measured whole-robot fidelity report | No complete reference dataset/report is committed. |
| Synchronize a physical robot and simulator live | No supported end-to-end workflow is documented or validated. |
| Replay arbitrary walking bags with quantitative fidelity scores | Requires a dedicated comparison workflow; the sysid fitter is fixed-base and is not this tool. |

## Identify the unit

Record the physical unit and leg revision, code/model revision, ROS profile,
policy reference, firmware, joint map and calibration files. Distinguish measured
values from CAD values and assumptions. The software maps physical v6.37a legs
to training `legs_v627` and ROS `wojtek_v2`; verify that mapping against the
assembled hardware before applying its model.

Measure joint limits and zero poses, mass and load distribution, foot/contact
geometry, actuator response, sensor mount/axes and transport latency. Keep the
measurement method, instrument, date and uncertainty with each result. Do not
replace an unknown electrical rating or mechanical limit with a model default.

## Fit and test separate data

1. Complete [commissioning](../../docs/commissioning.md), including drive signs and
   absolute encoder coordinates.
2. Record repeatable air-rig excitation with the selected model and joint map.
   Use [sysid](sysid.md) to fit actuator parameters. Keep the original bag and
   result metadata; do not copy effective fitted gains into motor firmware.
3. Reserve a separate recording for validation. Repeat fits across seeds and
   windows to expose weakly identified parameters.
4. Compare default and identified models using the same recorded input. For
   fixed-base data, report per-joint trajectory RMSE, bias, lag and peak error.
   Ground-contact validation additionally needs measured base/contact state and
   a replay tool; a lower air-rig fitting error cannot establish ground fidelity.
5. Test standing, folded/stand transitions and walking under the same policy and
   command sequence. Record stance height, attitude, travel, slip, torque and
   timing errors. Use independent pose measurements where possible: odometry
   drift is not ground truth.

Set acceptance tolerances before fitting. Report both successful and failed
cases, the test surface/load, and the conditions outside the claim. Sensor-image
quality and physical-motion fidelity need separate measurements.

## New-leg fidelity limits

The documented folded hip target is 2.903 rad; the exported model limits it to
2.695 rad. Whether the physical stop allows the target still needs measurement.
The training collision primitives also extend below the visual body and thigh.
A previous lying-pose check used a separate mesh-collision setup that is not
part of the committed model. Therefore the current model does not establish
accurate physical folded/stand behavior for these legs.

Resolve the measured range and collision geometry in the source model/builder,
then rerun the transition and contact checks. Do not widen a modeled limit
solely to make the simulated pose match a desired target. See
[variant evidence](robots.md#ros-description) for the measurements and
[unsupported features](robots.md#what-does-not-support-a-variant-yet) for task,
terrain, camera and planner restrictions.

## Release a validation bundle

Keep these together for each claimed fidelity level:

- Physical/model revision, policy, firmware and calibration manifest.
- Measurement methods and original held-out bags/data.
- Exact replay/comparison commands, tool versions and random seeds.
- Measured errors against predeclared tolerances, including failures.
- Known mismatches, excluded conditions and the next measurement needed.

Another developer should be able to reproduce the report before calling that
configuration a validated twin. Until then, describe it as a simulated robot
with the listed calibration and fidelity limits.
