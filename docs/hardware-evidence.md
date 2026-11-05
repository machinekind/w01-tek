# Hardware evidence register

This register identifies the physical information needed to complete the build
pack and validate a particular robot. The software/model references already in
the repository are linked from [Hardware](hardware.md). Open rows below require
assembly records, supplied design files or measurements; they are not claims
that the hardware is defective.

## Missing inputs

| ID | Required evidence | Source to obtain it from | Completion criterion |
|---|---|---|---|
| H01 | Current physical leg/body revision and its CAD revision | Mechanical team and unit inspection | Assembly photos and a revision map; confirm the v6.37a/v6.27 kinematic correspondence |
| H02 | Source CAD, dimensioned drawings, BOM and licenses | Mechanical design owner | A versioned, accessible manufacturing pack for each supported assembly |
| H03 | Fasteners, bearings, materials, print/fabrication settings, tolerances and assembly order | Mechanical design owner | Illustrated assembly with part quantities and tightening/fit requirements |
| H04 | Battery, charger, DC/DC, fuse, switches, connector pinouts and wire specifications | Electrical owner and unit inspection | Annotated power/harness drawing with ratings and a confirmed power-isolation method |
| H05 | Actual motor/controller identities, firmware and saved configuration | Drive labels and configuration backup | One restorable configuration per drive type, with IDs and baud recorded |
| H06 | Computer board, RAM, cooling, OS and kernel | Unit readout | Versioned commissioning record; resolve Pi 3/Pi 4/Pi 5 assumptions for that unit |
| H07 | Folded hip range, stops, standoffs and collision clearances | Physical measurement plus CAD | Explain 2.903 rad folded target versus 2.695 rad exported limit; reproduce the accepted geometry in the model |
| H08 | Mass, centre of mass, actuator response, friction and timing | Mechanical measurements and identification runs | Measured/modelled comparison with units, uncertainty and held-out validation |
| H09 | IMU calibration and camera extrinsics | Calibration captures and mount measurement | Replace identity/design placeholders with unit-specific values and independent checks |
| H10 | Charging/storage instructions and operating limits | Component documentation and supervised measurements | Accepted voltage/temperature/payload/runtime/terrain limits tied to the physical revision |

Until H02–H04 are supplied, the repository is not a complete manufacturing or
electrical build pack. Until H07–H09 are validated, new-leg folded-pose and
calibrated-twin fidelity remain limited.

## Unit record

Create one record for each physical robot, held with its calibration and test
artifacts. Include these fields:

| Field | Record |
|---|---|
| Identity | Unit identifier; body, leg and harness revisions; dated assembly photos |
| Software | Git commit, ROS profile, policy reference and revision, model revision |
| Computer | Board/RAM, OS image, kernel, service unit and drop-ins |
| Drives | CAN ID, joint, motor/controller model, firmware, stored configuration and backup |
| Power | Pack/converter/protection/charger identity, approved settings and isolation location |
| Calibration | Encoder-zero definition, joint signs, IMU mount/magnetometer file, camera extrinsics |
| Tests | Commands, fixture, conditions, expected results, measurements, logs/bags and deviations |
| Acceptance | Physical operating limits, reviewer and date; open checks |

Keep original measurements alongside derived values. A model parameter should
name its source and whether it was measured, fitted, supplied by design or
assumed. A calibration file copied from another unit remains unvalidated until
checked on this one.

## Existing evidence

- [Robot variants](../training/docs/robots.md): mechanism changes, mechanical-team
  corrections and model limitations.
- [Manual joint signs](../ros/hw_tests/signcheck/README.md): physical v6.37a
  measurements from 2026-10-23.
- [Hardware bench tests](../ros/hw_tests/README.md): CAN and IMU procedures.
- [Camera noise](perception/d435-noise.md): prior measurements; the measurement
  scripts still need to be supplied for full reproduction.
- [System identification](../training/docs/sysid.md): actuator fitting and
  held-out validation.
- [2025 thesis](comparative-studies-on-walking-control-algorithms-for-the-quadruped-robot-4barbot.pdf):
  original design history, not evidence for every current component.
