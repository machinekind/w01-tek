# Evaluate, promote and restore a policy

A training checkpoint and a deployment artifact serve different purposes.
Checkpoints resume training and need the original network/configuration.
The deployment pair contains the NumPy weights and the robot contract:
`policy.npz` and `policy_meta.json`. Keep the pair together.

## Record a candidate

For each candidate, retain the code revision, source/generated model revision,
`uv.lock`, resolved configuration, seed, `run.json`, checkpoint and evaluation
results. For a physical run, also record robot revision, calibration, firmware,
effective service configuration and any gain/torque overrides. Generated runs
and credentials stay out of Git.

```bash
# Repository root; paths passed to run.sh are relative to training/.
./training/run.sh report --run runs/wojtek_candidate
./training/run.sh courses --run runs/wojtek_candidate
./training/run.sh eval --run runs/wojtek_candidate --x-vel 0.3
./training/run.sh export --run runs/wojtek_candidate
```

Compare against a named baseline with the same robot, scenarios, seeds and
evaluation settings. Review falls, velocity/height tracking, torque saturation,
impact proxies, power and turning/stopping behavior. Choose acceptance limits
before comparing candidates; the repository has no single release threshold
valid for every task and hardware revision. Preserve rejected results too.

## Test the exported pair in ROS simulation

Export into the store mounted by the ROS container:

```bash
./training/run.sh export --run runs/wojtek_candidate \
  --out ../ros/policies/local/wojtek_candidate
./ros/sim.sh --foxglove robot:=wojtek \
  policy:=/ros2_ws/policies/local/wojtek_candidate
```

For a `legs_v627` run, use `robot:=wojtek_v2`. The launch rejects a policy for
the wrong leg family. Follow the [simulation procedure](../../docs/simulation.md)
and [operator runbook](../../docs/operations.md): verify startup, zeroing,
stand/lie ramps, walking, command expiry and cancellation. A short pipeline
check may export successfully and still fall immediately; keep it in simulation.

The exported contract carries gains, torque cap, limits, observation layout,
control timing and robot identity. The launch reads these values. A manual
override changes the plant the policy sees, so record and evaluate it explicitly.
Do not edit weights or metadata to bypass a compatibility rejection.

## Publish a keeper

Publish only after the candidate meets the chosen evaluation criteria. An
artifact repository should contain the deployment pair, source checkpoint,
`run.json`, a model card describing hardware/task support, evaluation results
and enough provenance to reproduce them. Keep the policy reference pinned to
an immutable revision for operation.

```bash
# Run from the repository root, after creating the intended artifact repository.
hf upload <organization>/<policy-name> \
  training/runs/wojtek_candidate/deploy . --include 'policy*'
```

Upload checkpoint and report files separately as required by that repository's
layout. The command above publishes only the deployment pair. Private keeper
access needs `HF_TOKEN` or `hf auth login`; a local exported directory works
without either.

The default robot pins live in
[`wojtek_policy/robots.py`](../../ros/src/wojtek_policy/wojtek_policy/robots.py).
Changing a pin is a reviewable configuration change. The stock training demo
default and ROS profile pin are separate settings; pass an explicit reference
when comparing them.

## Deploy and roll back

Before physical deployment, complete the revision-specific commissioning checks
and arrange the correct folded pose. Record the previous policy reference and
effective robot/service configuration. From the repository root:

```bash
./ros/deploy.sh --policy <organization>/<policy-name>@<revision>
```

This fetches and syncs the policy and writes the robot's override. The deployment
script restarts the robot service only when it is enabled; inspect the effective
service rather than assuming a restart occurred. The robot can then load its
cached pinned artifact without internet access.

To restore a previous override, deploy its exact reference. A plain
`./ros/deploy.sh` clears the override and returns to the current profile pin;
it does not necessarily restore the last policy you ran. Restore the previous
code/configuration and calibration separately if those also changed. Follow
the operator runbook for shutdown and startup; disarm does not remove torque.

`WOJTEK_POLICY` affects desk/simulation sessions only. `WOJTEK_ROBOT` is read
by deployment; provisioning writes the selected profile into the robot service.
Changing that variable alone does not change an already installed service.
