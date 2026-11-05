# Documentation backlog

Make Wojtek understandable and reproducible for a new builder, operator and
developer. A reader should be able to identify the robot, assemble and commission
it, operate it with a pad or Steam Deck, integrate software, run simulation,
train and deploy a policy, and understand the evidence behind a digital twin.

This backlog comes from the documentation and code audit of commit `207480e`.
The original scopes and acceptance criteria are retained below. Status lines
record the implemented work; the [validation record](../validation.md) records
checks, environments and limitations. **Done** closes the stated scope;
**Partial** has implementation but lacks required evidence or acceptance;
**Open** has no completed demonstration. Checkmarks do not certify hardware.

Work types: **Docs** can be written from existing evidence; **Code** requires an
implementation change; **Hardware** requires component information, measurements
or physical validation. A documented limitation can complete a documentation
task, but it does not complete the engineering work that removes that limitation.

| Priority | Outcome |
|---|---|
| P0 | Operating claims and current configuration are trustworthy. |
| P1 | A newcomer can follow the main operator and developer workflows. |
| P2 | Physical construction, commissioning and twin validation are reproducible. |
| P3 | Documentation remains accurate as hardware and software change. |

All **H01–H10 remain pending** in the [hardware evidence register](../hardware-evidence.md).
Fresh human walkthroughs remain open under D32. Supplied design files,
measurements and independent use of the guides cannot be replaced by code tests.

## P0 Correct claims that affect operation

- [x] **D01 Fix and verify navigation cancellation.** **Code + Docs**
  **Status: Done.** One [cancel implementation](../../ros/src/wojtek_nav/wojtek_nav/goto.py), [regression tests](../../ros/src/wojtek_nav/test/test_goto.py) for immediate stop/late goals and corrected [navigation semantics](../../ros/src/wojtek_nav/README.md). Physical stopping behavior remains a commissioning check.

  Remove the duplicate `GotoController.cancel` definition and reconcile its
  signature with the ROS callback. Exercise cancellation while moving and a
  goal arriving immediately after cancellation.
  **Done when:** regression checks verify immediate cancellation and the late-goal
  hold-off, and the STOP description matches the verified behavior.
  Sources: [controller](../../ros/src/wojtek_nav/wojtek_nav/goto.py),
  [callback](../../ros/src/wojtek_nav/wojtek_nav/goto_node.py),
  [navigation guide](../../ros/src/wojtek_nav/README.md).

- [ ] **D02 Define operating states and stopping behavior.** **Docs + Hardware**
  **Status: Partial.** [Operations](../operations.md) has the state table, transitions and torque/ramp distinctions. Confirm the installed physical isolation method and stop procedure: H04.

  Add one state table and transition diagram covering policy enable, arm/disarm,
  powered hold, pose/trick ramps, dry-run, drive disable and physical power
  removal. Correct the claim that arming is when torque first reaches the motors.
  **Done when:** every operator guide explains that disarm does not remove torque
  or cancel an active ramp, and the installed robot's physical stop procedure has
  been confirmed. Keep that confirmation open until inspected.
  Sources: [ROS guide](../../ros/README.md),
  [I/O adapter](../../ros/src/wojtek_bringup/wojtek_bringup/real_io_node.py).

- [ ] **D03 Reconcile current hardware and control settings.** **Docs + Hardware**
  **Status: Partial.** [Compatibility](../compatibility.md) and [architecture](../ros-architecture.md) distinguish configured rates/defaults from measurements. Record installed drives, board, image and measured rates: H05–H06.

  Replace current-looking 400 Hz/8M/disabled-IMU descriptions with the settings
  supported by the active configuration. Distinguish Pi 3 and Pi 5 deployments,
  configured rates from measured rates, and historical choices from defaults.
  **Done when:** architecture, bench guides and quickstarts agree on control and
  sensor rates, CAN transport/baud, IMU integration and machine roles; the
  installed hardware identity is confirmed rather than inferred from comments.
  Sources: [architecture](../ros-architecture.md),
  [controller settings](../../ros/src/wojtek_bringup/config/real_controllers.yaml),
  [launch defaults](../../ros/src/wojtek_bringup/wojtek_bringup/launch_common.py),
  [bench tests](../../ros/hw_tests/README.md).

- [x] **D04 Reconcile deployment and service instructions.** **Docs**
  **Status: Done.** [Deck setup](../../ros/deploy/deck/README.md), [Pi image](../../ros/deploy/rpi/IMAGE.md), [operations](../operations.md) and [sim contract](../sim-test-contract.md) match repository scripts/configuration. Per-unit effective-service inspection is required by commissioning.

  Correct Deck package installation claims, policy-store paths, autostart and
  restart behavior, and the effective service drop-in. Explain which invocation
  enables bags, telemetry, camera and Foxglove. Remove resolved dependency
  failures from current simulation instructions.
  **Done when:** documented defaults match the installed service configuration;
  a reader can inspect the effective unit and verify which data is recorded.
  Sources: [Deck runbook](../../ros/deploy/deck/README.md),
  [service override](../../ros/deploy/rpi/wojtek-robot-local.conf),
  [deployment script](../../ros/deploy.sh), [sim contract](../sim-test-contract.md).

- [x] **D05 Make required artifacts and access explicit.** **Docs**
  **Status: Done.** [Simulation](../simulation.md) states keeper/local-policy prerequisites and container paths; [training](../../training/README.md) and [policy lifecycle](../../training/docs/policy-lifecycle.md) provide the local train/export route. Its new GPU acceptance run is tracked by D14.

  Explain private keeper access, Hugging Face configuration, local policy
  directories, container-visible paths and offline caching. Choose a reproducible
  starter route for readers without private keeper access: an accessible sample
  artifact or a fully documented local training/export path.
  **Done when:** every quickstart states its artifact prerequisites before launch,
  and missing access produces an explained recovery path.
  Sources: [environment template](../../.env.example),
  [policy resolution](../../ros/src/wojtek_policy/wojtek_policy/policy_source.py).

## P1 Complete the main reader workflows

- [x] **D06 Create a documentation home and improve the root README.** **Docs**
  **Status: Done.** [Root README](../../README.md) and [documentation home](../README.md) route builders, operators, integrators and RL developers to current guides and separate historical evidence.

  Explain what Wojtek is, show the current robot, summarize its capabilities,
  and link routes for operators, builders, software integrators and RL developers.
  Separate tutorials, operational procedures, references and historical reports.
  **Done when:** each question in the opening paragraph has a clear starting
  link; no newcomer has to begin with the agent guide.

- [ ] **D07 Publish a hardware and software compatibility matrix.** **Docs + Hardware**
  **Status: Partial.** [Compatibility](../compatibility.md) maps physical names, training/ROS selectors, models, policies and support limits. Confirm each physical assembly's revision correspondence: H01.

  Map stock legs, CAD v6.27, physical v6.37a, training `legs_v627` and ROS
  `wojtek_v2`. Include model, joint map, policy artifact, observation set, task,
  simulator and deployment support. Explain `WOJTEK_ROBOT`, training `robot=`
  and ROS `robot:=`, including where each setting takes effect.
  **Done when:** supported, experimental, unsupported and unvalidated combinations
  are explicit; trainable policies are distinguished from exportable/deployable
  ones. Confirm the physical revision correspondence with the mechanical team.
  Sources: [variants](../../training/docs/robots.md),
  [ROS profiles](../../ros/src/wojtek_policy/wojtek_policy/robots.py).

- [ ] **D08 Write a simulation chooser and clean-install quickstart.** **Docs**
  **Status: Partial.** [Simulation](../simulation.md) covers mock, MuJoCo, variants, setup, control and restart; other simulators are classified. Extend the [recorded checks](../validation.md) to the remaining supported host/profile paths and a fresh-reader run.

  Cover model viewing, ROS mock, ROS MuJoCo, MJX training, room demos and Isaac;
  identify which software stack each exercises. Give tested Linux/macOS setup,
  resource requirements, policy acquisition, headless options and expected output.
  Include stock and new-leg launches, control, reset/restart and shutdown.
  **Done when:** a reader starting from a clean supported environment can launch
  a matching policy and complete a stand/drive/stop exercise; unsupported paths
  are labeled beside the commands. Depends on D05 and D07. Ship the ROS path
  first; only maintained room/Isaac recipes depend on D27.

- [ ] **D09 Write one cold-start-to-shutdown operator runbook.** **Docs + Hardware**
  **Status: Partial.** [Operations](../operations.md) covers states, startup/shutdown, pads, Deck mappings and recovery. Complete physical pairing, disconnect, handover and full operator acceptance on the actual devices.

  Include physical boot pose, profile/policy selection, network connection,
  zeroing, stand, arm, drive, disarm, lie and shutdown. Cover browser pad,
  ROS `joy`, robot-paired Xbox and Steam Deck setup, button mappings, speed,
  height, tricks, Steam controller capture and reconnect behavior.
  **Done when:** another operator completes the sequence and common recovery
  cases without reading implementation files. All stop instructions use D02;
  control handover follows D13.
  Sources: [ROS guide](../../ros/README.md),
  [Deck runbook](../../ros/deploy/deck/README.md),
  [panel guide](../../ros/src/wojtek_deck/README.md).

- [x] **D10 Consolidate troubleshooting, telemetry and incident recovery.** **Docs**
  **Status: Done.** [Troubleshooting](../troubleshooting.md) identifies failing layers, expected observations, recovery, recording checks and the diagnostic bundle.

  Cover missing policy, DDS discovery, wrong profile/zero pose, refused arm,
  stale sensors, competing command sources, absent camera, low control rate,
  motor power cycles and missing recordings. Give checks, expected observations,
  recovery steps, bag locations and a useful diagnostic bundle.
  **Done when:** an operator can distinguish a command, motor, sensor, network
  or compute problem, collect relevant evidence and follow a documented recovery.

- [x] **D11 Refresh the ROS architecture and interface reference.** **Docs**
  **Status: Done.** [Architecture](../ros-architecture.md) documents machine roles, graph, coordinates, topics/services, timing and sim substitution; [integration](../integration.md) supplies client-facing semantics.

  Show processes on the robot, PC/GPU host and handheld, plus the sim hardware
  substitution. Document topic/service types, units, frames, joint order,
  boot-relative versus absolute angles, timestamps, QoS, rates and timeouts.
  Separate command freshness from sensor freshness and motor behavior.
  **Done when:** diagrams match the supported launch graph, obsolete paths are
  clearly historical, and an integrator can locate each interface contract.
  Source: [architecture](../ros-architecture.md). Depends on D03 and D07.

- [x] **D12 Add a complete external integration tutorial.** **Docs + Code**
  **Status: Done.** [Tutorial](../integration.md), [waypoint client](../../ros/src/wojtek_nav/wojtek_nav/goal_client.py) and [tests](../../ros/src/wojtek_nav/test/test_goal_client.py) cover readiness, completion, timeout, disconnect and cancellation. Task-correlation limits are explicit; runtime scope is in [validation](../validation.md).

  Provide a small runnable client that checks readiness, subscribes to status,
  sends a supported command/task, observes completion, cancels and handles a
  disconnect. Document available task correlation, status JSON, errors and
  versioning; identify any guarantees the current interfaces cannot provide.
  **Done when:** the example runs against ROS MuJoCo with documented QoS and
  frame/time semantics, and its cancellation path passes D01. Include routes
  for velocity, metric goal, visual target and natural-language integration.

- [ ] **D13 Define and enforce command ownership.** **Code + Docs**
  **Status: Partial.** [Opt-in mux](../command-ownership.md) and its tests define priorities, leases, manual takeover and explicit autonomy enable. ROS graph and simulated handover checks pass; complete physical-device handover. The default direct path still has no ownership.

  Specify ownership and handover between pad, Deck, consoles and autonomy,
  including what happens to a brain task on disarm or manual takeover.
  Document the current last-writer-wins limitation immediately, then implement
  the chosen arbitration behavior as a separate engineering change.
  **Done when:** simultaneous sources and reconnects have tested outcomes,
  manual takeover cannot silently resume an old task, and operator/client
  examples follow the same contract.

- [ ] **D14 Add a training README and explain the full RL pipeline.** **Docs**
  **Status: Partial.** [Training quickstart](../../training/README.md) covers locked setup, observations, PPO, evaluation and export; configurations resolve and model-free tests run. Complete a clean bounded GPU train/checkpoint/evaluate/export run and retain its artifacts.

  Trace model → environment → actor/critic observations → PPO → checkpoint →
  fixed evaluation → schema-2 export → NumPy runtime → ROS. Document installation
  from the lockfile, optional extras, GPU requirements, configuration resolution,
  bounded GPU validation, training, evaluation and export. Resolve contradictory
  CPU-smoke advice; distinguish CPU-capable inspection/evaluation from training.
  **Done when:** a new developer follows one supported run through export, knows
  which observations exist on hardware, and can find expected files and failure
  recovery. Keep the detailed reference and link it from the tutorial.
  Sources: [configuration](../../training/docs/configuration.md),
  [dependencies](../../training/pyproject.toml),
  [learning course](../../learning/wojtek_rl_course/README.md).

- [ ] **D15 Write a VLM navigation tutorial and evaluation guide.** **Docs**
  **Status: Partial.** [Navigation guide](../../ros/src/wojtek_nav/README.md) covers server setup, task contracts, limits and capture/scoring tools. Build a human-reviewed fixture and record end-to-end success, failures and latency with named model/host versions.

  Cover model/server versions, measured host requirements, setup/tunneling,
  container addresses, camera/depth requirements, task submission, status and
  stopping. Explain target verification, pixel resolution, local costmap and
  straight-line navigation limits. Include model substitution and a repeatable
  evaluation recipe with success, failure and latency results.
  **Done when:** a reader reproduces a supported simulated task; robot, remote
  relay and simulation validation status are distinct. Do not claim persistent
  mapping, general route planning or negative-obstacle handling where absent.
  Source: [navigation guide](../../ros/src/wojtek_nav/README.md). Depends on D01.

- [x] **D16 Document network access and remote deployment boundaries.** **Docs + Code**
  **Status: Done.** [Network guide](../network-access.md) and [observer profile](../../ros/src/wojtek_pc/config/observer.yaml) define ports, trust and restrictions. [Live acceptance](../validation.md) covers allowed subscriptions and denied control/asset requests. This applies to the observer bridge only; DDS and other gateways retain their documented trust limits.

  List required network paths, ports, DDS domain, offline behavior and where
  cameras/model requests travel. Document which bridge/gateway clients can
  publish commands, call services or change parameters. Provide a tested access
  restriction configuration and state remaining trusted-network assumptions.
  **Done when:** a deployer can expose only the intended capabilities and test
  both authorized operation and refused access. Keep private infrastructure
  identities and credentials out of examples.

- [ ] **D17 Define policy promotion, release provenance and rollback.** **Docs**
  **Status: Partial.** [Policy lifecycle](../../training/docs/policy-lifecycle.md) defines artifacts, comparison criteria, local simulation, pins and rollback. Archive one completed candidate's provenance/evaluation bundle and rehearse restoring its exact reference.

  Explain checkpoint versus deployable artifact, baseline comparisons, agreed
  acceptance criteria, seeds/resolved config, code/model versions, export parity,
  pinned references, deployment and rollback. Include firmware/calibration
  compatibility in the release record.
  **Done when:** one worked example identifies everything needed to reproduce
  and restore a supported robot/policy combination. Depends on D07 and D14;
  complete this before presenting the deployment workflow as ready to follow.

## P2 Make hardware and digital twins reproducible

- [ ] **D18 Assemble the current mechanical build and upgrade pack.** **Docs + Hardware**
  **Status: Partial.** [Hardware](../hardware.md) separates simulation assets from manufacturing CAD and identifies new-leg limits. Obtain revisioned CAD/BOM, drawings and illustrated assembly: H01–H03.

  Obtain versioned source CAD, drawings, BOM, fabrication/printing requirements,
  exploded assembly, fasteners, bearings, tolerances and cable routing. Include
  illustrated stock-to-new-leg changes and links to the authoritative mechanical
  source, whether maintained here or elsewhere.
  **Done when:** a builder can identify and obtain every part and assemble a
  specified revision without treating visualization meshes as manufacturing CAD.
  Label the thesis as the historical 2025 design reference.

- [ ] **D19 Assemble the electrical build and power reference.** **Docs + Hardware**
  **Status: Partial.** [Hardware](../hardware.md) supplies known communication topology, CAN IDs and IMU header wiring. Obtain the rated power/harness schematic, connector details and isolation sequence: H04.

  Obtain an annotated battery/DC-DC/fuse/switch/power-distribution diagram,
  CAN/I2C/USB topology, connector pinouts, wire specifications, grounding,
  termination locations and CAN-ID-to-joint table. Identify physical isolation
  points and the power-on/off sequence.
  **Done when:** another builder can reproduce and inspect the harness; all
  electrical ratings come from confirmed components/design data rather than
  inferred values.

- [ ] **D20 Document drive provisioning, replacement and recovery.** **Docs + Hardware**
  **Status: Partial.** [MD80 guide](../../ros/src/md80_hardware_interface/README.md) retires the incomplete flashing recipe and describes replacement prerequisites. Obtain identities, firmware and restorable configuration backups: H05.

  Reconcile historical motor names with installed motors. Record firmware,
  known-good configuration, CAN IDs/baud and backup/restore procedures. Supply
  the missing motor configuration referenced by flashing scripts, or retire
  the obsolete recipe.
  **Done when:** replacement-drive setup is reproducible for a named hardware
  revision, and every referenced configuration artifact is available.

- [ ] **D21 Create a commissioning and calibration runbook.** **Docs + Hardware**
  **Status: Partial.** [Commissioning](../commissioning.md) orders bus, signs, zeros, IMU, camera and powered checks, with records and expected observations. Set remaining physical acceptance limits and execute the procedure on an identified unit.

  Order checks for inventory, wiring/bus health, selected profile, boot pose,
  encoder zeros, manual joint signs, IMU axes/bias/magnetometer, camera extrinsics,
  dry-run and initial powered movement. Reuse existing soak/sign-check tools and
  record calibration values with the physical unit and revision.
  **Done when:** each check has required tools, expected output and an agreed
  pass condition; placeholder calibration is visibly distinguished from measured
  calibration. Depends on D02, D07, D19 and D20.
  Sources: [bench tests](../../ros/hw_tests/README.md),
  [sign checks](../../ros/hw_tests/signcheck/README.md),
  [perception](../../ros/src/wojtek_perception_bringup/README.md).

- [ ] **D22 Document maintenance and measured operating limits.** **Docs + Hardware**
  **Status: Partial.** [Maintenance](../maintenance.md) covers inspection, change-triggered checks and recovery. Obtain component charging/storage instructions and measured operating limits: H10.

  Cover charging/storage, battery stop thresholds, thermal limits, runtime,
  payload, speed and terrain envelope, fastener/bearing/harness inspection and
  recalibration after replacement. Record unknown limits as measurement tasks.
  **Done when:** operators have a revision-specific maintenance schedule and
  evidence-backed limits; undocumented values are not presented as safe limits.

- [x] **D23 Resolve or explicitly bound new-leg model fidelity.** **Code + Hardware + Docs**
  **Status: Done — bounded.** [Variants](../../training/docs/robots.md), [hardware](../hardware.md) and [twin guide](../../training/docs/digital-twin.md) identify the folded-range/collision mismatches, affected workflows and closure evidence. Geometry correction and physical validation remain H07.

  Measure the real folded hip stop against the documented 2.903 rad target and
  exported 2.695 rad range. Reconcile lying-pose collision shapes with physical
  geometry and commit a reproducible validation procedure. Track stock-sized
  camera/planner/benchmark assumptions separately from unsupported tasks.
  **Done when:** each mismatch has evidence, affected workflows and a closure
  criterion; folded/stand/trick fidelity is claimed only for validated cases.
  Source: [variant limitations](../../training/docs/robots.md).

- [x] **D24 Make system identification select the correct robot and joint map.** **Code + Docs**
  **Status: Done.** [CLI](../../training/wojtek_rl/sysid/__main__.py), [input validation](../../training/wojtek_rl/sysid/inputs.py), [tests](../../training/tests/unit/test_sysid_inputs.py) and [guide](../../training/docs/sysid.md) cover profile/map selection, coordinate conversion, provenance and held-out fitting. Unlabelled bags do not establish physical identity.

  Expose and validate variant/joint-map selection in the sysid CLI; the audit
  found that changing XML still used the stock map when reading a bag. Update
  instructions for excitation, data checks, fitting, held-out validation and
  applying results. Explain effective fitted gains versus drive settings.
  **Done when:** coordinate conversion is tested for both profiles, profile/map
  selection and provenance are explicit, and detectable metadata mismatches are
  refused. Do not imply that an unlabelled bag reveals its physical revision.
  Sources: [sysid guide](../../training/docs/sysid.md),
  [CLI](../../training/wojtek_rl/sysid/__main__.py).

- [x] **D25 Publish a digital-twin definition and validation workflow.** **Docs**
  **Status: Done.** [Twin guide](../../training/docs/digital-twin.md) separates implemented simulation/sysid from calibrated-unit evidence, walking-bag comparison and live synchronization. The physical demonstration is D26.

  Distinguish a simulated robot, a calibrated model of a specific physical unit,
  and live/replay synchronization. Connect measurements, calibration, actuator
  identification, telemetry timing, held-out replay and divergence measurements.
  Version model, calibration, policy and physical revision together.
  **Done when:** the guide identifies implemented stages, prerequisites, expected
  evidence and missing capabilities. Mark live synchronization or replay
  unavailable until implemented. Publish this scope guide now; its physical
  demonstration is D26 and depends on D21–D24.

- [ ] **D26 Demonstrate a calibrated twin of one physical unit.** **Hardware + Code + Docs**
  **Status: Open.** Obtain H07–H09 measurements and independent held-out data, implement the needed comparison/replay tools, and publish a reproducible error/uncertainty report using the [twin bundle requirements](../../training/docs/digital-twin.md#release-a-validation-bundle).

  Collect versioned calibration and held-out measurements for one supported
  revision. Implement any missing comparison/replay tooling required by the
  chosen validation scope, then compare modeled and measured behavior.
  **Done when:** another developer can reproduce a comparison report with agreed
  tolerances, measured error, uncertainty and exclusions. Depends on D21–D25;
  record what remains unvalidated instead of claiming full twin fidelity.

- [ ] **D27 Make room and Isaac demo setup reproducible or archive it.** **Docs + Code**
  **Status: Partial.** [Room guide](../../training/demo/README.md) covers extras, assets/license, scenes and policy selection; [Isaac](../../training/demo/isaac/README.md) and the legacy app are explicitly historical. Complete the room recipe from a clean checkout and record host/rendering results.

  Decide the support status of each demo. For maintained paths, document optional
  dependencies, asset source/license, acquisition, scene construction, policy
  selection, camera placement and robot footprint. Repair removed runtime
  references and machine-local paths. Separate kinematic navigation evaluation
  from physics rollout and ROS integration testing.
  **Done when:** a supported demo launches from a clean checkout with documented
  assets; archived demos cannot be mistaken for a current twin quickstart.
  Sources: [demo guide](../../training/demo/README.md),
  [Isaac guide](../../training/demo/isaac/README.md),
  [room application](../../training/wojtek_rl/room_app.py).

- [ ] **D28 Make perception and autonomy evidence repeatable.** **Docs + Hardware + Code**
  **Status: Partial.** [Perception guide](../../ros/src/wojtek_perception_bringup/README.md) distinguishes design extrinsics and historical noise results; [navigation capture/scoring](../../ros/src/wojtek_nav/README.md) provides a new fixture route. Supply the original noise-measurement tools, measure H09 and record acceptance for each deployment route.

  Publish camera extrinsics/noise measurement procedures and required scripts.
  Reconcile stale SLAM references with the current rolling costmap. Record
  navigation validation separately for core and experimental deployment routes,
  including whether obstacle veto is active and where compute runs.
  **Done when:** replacing a camera or changing a deployment can be followed by
  the same acceptance checks; documented successes include their actual setup
  and do not imply capabilities of another route.
  Sources: [camera noise](../perception/d435-noise.md),
  [perception guide](../../ros/src/wojtek_perception_bringup/README.md).

## P3 Keep the documentation useful

- [x] **D29 Audit all commands and classify historical material.** **Docs**
  **Status: Done.** [Configuration reference](../../training/docs/configuration.md) inventories commands; [demo status](../../training/demo/README.md), [MJWarp report](mjwarp-phase0-report.md) and [documentation home](../README.md) separate current workflows from historical results. Execution coverage is limited to [validation](../validation.md).

  Inventory active commands, including sysid, distillation, room construction
  and navigation evaluation. Label references, experiments and archived reports;
  keep current conclusions linked to their evidence. Improve experiment
  discovery while preserving dependency isolation and repository hygiene rules.
  **Done when:** supported commands are discoverable and dated experimental
  conclusions cannot be mistaken for current installation instructions.

- [x] **D30 Set documentation ownership and language conventions.** **Docs**
  **Status: Done.** [Contributing](../../CONTRIBUTING.md) assigns subsystem review roles and conventions for English entry points, Polish references, evidence, prerequisites and command locations.

  Assign a maintaining role to each guide; agree the primary newcomer-guide
  language and treatment of Polish technical notes. Standardize revision names,
  glossary terms, working-directory notation, prerequisites and expected output.
  **Done when:** readers can distinguish current facts, assumptions and measured
  results; contributors know which guides must change with their code/hardware.

- [x] **D31 Add documentation validation to contributor checks.** **Code + Docs**
  **Status: Done.** [Checker](../../scripts/check_docs.py), [CI workflow](../../.github/workflows/docs.yml) and [contributor checks](../../CONTRIBUTING.md) cover local links/anchors, important defaults and whitespace. [Validation](../validation.md) keeps simulator, GPU and physical acceptance separate.

  Check relative links and examples, and validate important documented defaults
  against configuration where practical. Keep fast/model-free checks separate
  from GPU, simulator and hardware validation. Record last-verified revision and
  environment for executable quickstarts.
  **Done when:** contributor instructions give one clear docs-validation route;
  broken links/default drift are caught without launching hardware or training.

- [ ] **D32 Review usability with people unfamiliar with the project.** **Docs**
  **Status: Open.** Recruit independent readers for the journeys below; record missing prerequisites, assistance and fixes. An agent's source review or test run is not a fresh human walkthrough.

  Have independent readers try the operator, integration, training and builder
  workflows. Record every missing prerequisite, unexplained term and need for
  maintainer assistance; fix the guide where the reader got stuck.
  **Done when:** the acceptance journeys below pass for their stated support
  level, and remaining blockers have explicit owners and follow-up tasks.

## Acceptance journeys — all human walkthroughs pending

Each maintained tutorial must state prerequisites, supported revisions, exact
commands and working directories, expected outputs, completion criteria,
shutdown/recovery and its verification environment. Unknown physical facts stay
open until obtained; writing a plausible value is not completion. Existing
agent-run checks do not complete a fresh human walkthrough. Record the reader's
failures, needed assistance, fixes and follow-up owner in the
[validation record](../validation.md).

| Reader | Evidence that the documentation works |
|---|---|
| New contributor | Finds the right guide, acquires a compatible policy and drives the supported robot in ROS simulation. |
| Operator | Connects a pad or Steam Deck, performs the full startup/shutdown sequence, and recovers from a documented fault. |
| Integrator | Runs the example client, observes completion/cancellation and handles a disconnect without guessing interface semantics. |
| RL developer | Installs the environment, runs a bounded supported experiment, evaluates it and validates an exported policy in ROS simulation. |
| Builder | Obtains the correct parts/design revision, assembles mechanics/electronics and records commissioning results. |
| Twin developer | Reproduces a physical-unit comparison with versioned inputs, measured error and explicit fidelity limits. |
