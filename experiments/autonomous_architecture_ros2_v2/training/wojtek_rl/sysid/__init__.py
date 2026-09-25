"""Engine-parameter identification from rosbag signals.

Replays a recorded command stream (/wojtek/joint_targets) through batched
MJX rollouts under candidate physics parameters and fits the parameters so
the simulated joint trajectories match the recorded /joint_states. The
search runs CMA-ES; each generation is one vmapped rollout of the whole
population, reusing the batched-model machinery from randomize.py.

Pipeline: bag.read_bag -> dataset.build_dataset -> space.ParamSpace +
rollout.make_evaluator -> the CMA-ES loop in __main__. See
training/docs/sysid.md for the workflow, including how to record
ground-truth bags from the ROS MuJoCo sim.
"""
