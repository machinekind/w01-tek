"""Robot variants: the numbers that belong to one set of legs.

Every variant has the same skeleton, so the env finds bodies, joints and
sensors by the same names on all of them: a free `root`, four legs in
paths.LEGS order, three driven joints per leg and a four-bar loop closed by
an equality constraint. What differs is geometry and strength, and that
lives here. build_model reads a variant to build its MJX model; the envs
read it for the constants they used to hold as literals.

A run picks its variant with `task.env.robot` (the `robot=` config group).
run.json records it, so eval, report and export rebuild the same robot.

This module stays free of mujoco and jax so the unit tests can import it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Robot:
    name: str
    # Which body of a leg carries the ground contact, as a suffix after
    # "{leg}_". The contact sphere sits at the "{leg}_foot" site, which must
    # be a site of this body.
    foot_body: str
    # Centre of the contact sphere relative to that site, in the foot body's
    # frame, for a site that marks the pad's axis rather than its middle.
    foot_offset: tuple
    foot_radius: float
    total_mass: float
    kp: float
    kd: float
    forcerange: float
    # link suffix -> floor-only primitive, see build_model step 2b.
    leg_collision: dict
    # Driven-joint targets (first, second, third) per leg, paths.LEGS order,
    # that the home keyframe is settled from.
    stand_pose: tuple
    # Height the base is dropped from when the home keyframe is settled.
    settle_drop_height: float
    # Make the source model's first keyframe the built model's rest pose.
    # MuJoCo opens every model at its rest pose, which is all joints at zero
    # unless the model says otherwise. With this on, a viewer or a bare
    # mj_resetData shows the robot standing with its loops closed. Joint
    # values keep their meaning. Off for the stock robot, whose generated
    # files stay as they are.
    rest_pose_from_source_key: bool
    timestep: float
    iterations: int
    ls_iterations: int
    # Dry friction (N*m) that replaces the source model's, by joint suffix
    # after "{leg}_". For a number the mechanical team corrected after the
    # export was made; a later export that carries it makes the entry moot.
    joint_frictionloss: dict
    # (d0, dwidth) of the loop-closure constraints' solimp, or None to keep
    # the source model's. Closer to 1 is a harder constraint.
    closure_solimp: tuple | None
    # Third-joint angle past which the four-bar is on its far branch. Tasks
    # that use it (getup, jump, biped) penalize knee targets beyond it.
    knee_singularity: float | None
    # Whether the env's four-bar toggle angle (fall.max_toggle_deg, the
    # toggle_flat reward) is defined for this linkage. It is measured
    # between sixth_link and foot_link, which only means something when the
    # loop closes at the foot.
    has_toggle_angle: bool
    # Standing height against a uniform leg extension: at each rung every
    # second joint moves by `dsecond` rad and every third by `dthird`, away
    # from the home pose, and the foot stays under the hip. Both are written
    # for a leg whose joints turn the positive way; `leg_sign` flips them
    # for the legs mounted the other way round, paths.LEGS order.
    height_table: tuple
    dsecond_table: tuple
    dthird_table: tuple
    leg_sign: tuple


# The robot as built. Comments on the individual numbers:
#   foot_radius: the real foot is a half-disc rubber pad bolted to the
#     sixth-link tip, its arc centered on the fifth/sixth closure pivot, so a
#     pivot-centered sphere of the pad's radius is the right contact model.
#     Radius measured from the pad disc baked into sixth_link.stl (mesh
#     extends 0.046 m past the pivot).
#   kd: raised 0.5 -> 1.0 after fbb_v2: the policy buzzed at ~7 Hz, right at
#     the underdamped PD resonance (sqrt(kp/I)/2pi with reflected inertia
#     ~0.01).
#   total_mass: real robot measured at 14 kg on a scale (2026-07-16);
#     supersedes the ~16 kg owner estimate from 2026-07-05 and the original
#     10.0 placeholder.
#   forcerange: set at the 16 kg estimate to keep the torque-to-weight ratio
#     that trained well at 10 kg / 6 Nm; kept at 9 for the measured 14 kg,
#     which lands a bit above that ratio (0.64 vs 0.6 Nm/kg). MD80 drives
#     allow 15 Nm; the jump task overrides this per-env (12 Nm) for the
#     launch.
#   leg_collision: sized from the link mesh AABBs. The bar capsules stop at
#     x=0.14 so they can never reach the foot sphere at the 0.21 pivot.
#   stand_pose: chosen via pose_explorer. Settled standing height 0.109 m,
#     the design height per the controller's gait code (foot z0 = -0.15
#     below the hip frame).
#   iterations: 1 clears the GPU step-rate gate but diverges in training:
#     under a flailing exploration policy the single Newton iteration goes
#     NaN and poisons the whole batch (fbb_v1, dead by 23M steps). Solver
#     robustness beats the gate heuristic; 2 iterations trains stably.
#   knee_singularity: the four-bar snaps through its singular branch past
#     this third-joint angle (the foot flips above the trunk). Crossing it
#     under load can break the linkage.
#   height_table: settled standing height measured with a 2 s PD-hold
#     settle (kp=20/kd=1). On these legs the third joint moves twice as far
#     as the second, at every height.
WOJTEK = Robot(
    name="wojtek",
    foot_body="foot_link",
    foot_offset=(0.0, 0.0, 0.0),
    foot_radius=0.046,
    total_mass=14.0,
    kp=20.0,
    kd=1.0,
    forcerange=9.0,
    leg_collision={
        "second_link": dict(type="sphere", size=[0.035, 0, 0], pos=[0.055, 0, 0]),
        "third_link": dict(
            type="capsule", size=[0.028, 0, 0], fromto=[-0.12, 0, 0.03, 0.04, 0, 0.03]
        ),
        "fifth_link": dict(
            type="capsule", size=[0.012, 0, 0], fromto=[0.02, 0, 0, 0.14, 0, 0]
        ),
        "sixth_link": dict(
            type="capsule", size=[0.012, 0, 0], fromto=[0.02, 0, 0, 0.14, 0, 0]
        ),
    },
    stand_pose=((0.0, -0.2, 3.1),) * 4,
    settle_drop_height=0.25,
    rest_pose_from_source_key=False,
    timestep=0.004,
    iterations=2,
    ls_iterations=5,
    joint_frictionloss={},
    closure_solimp=None,
    knee_singularity=3.2,
    has_toggle_angle=True,
    height_table=(0.084, 0.094, 0.106, 0.121, 0.139, 0.160, 0.182),
    dsecond_table=(-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45),
    dthird_table=(-0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90),
    leg_sign=(1.0, 1.0, 1.0, 1.0),
)

# The v6.27 four-bar legs (CAD archive sim_robot_v627, imported by
# import_robot.py). Same body, same joint names; the legs are about 2.3
# times longer and the loop closes 42 mm below the knee instead of at the
# foot. Every number below was measured on the imported model, not carried
# over from the stock robot:
#   foot_body / foot_offset / foot_radius: the pad is a 20 mm disc, 27 mm
#     wide, at the tip of the shin (sixth_link). The "{leg}_foot" site marks
#     its axis in the link plane; the disc's middle is 25.5 mm out along the
#     link's z. foot_link is only the closure point here and touches nothing.
#   total_mass: the CAD's own total (7.54 kg body + 8.81 kg of legs).
#   forcerange: the motor limit on all twelve joints. The export caps the
#     abduction motors at 2 N*m in its MJCF and at 9 N*m in its URDF; the
#     mechanical team confirmed the full 22 N*m for them (2026-09-20).
#   joint_frictionloss: the export has 1.26 N*m on the rod's joint and
#     1.48 N*m on the knee. The mechanical team corrected them to 0.48 and 0
#     (2026-09-20).
#   kp / kd: a starting point, not a tuned value. Standing takes about
#     8 N*m at the knee crank, so kp 60 leaves 0.13 rad of sag there.
#     task.env.pd_kp / pd_kd override them per run without a rebuild.
#   leg_collision: from the visual mesh AABBs. The shin capsule is thinner
#     than the pad and stops 15 % short of it, so the pad always lands first.
#   stand_pose: the export's `stanie` keyframe (second 2.25954745, third
#     2.00631532), with the rear legs a tenth of a height rung longer and
#     the front legs a tenth shorter, about 3 mm each. The centre of mass is
#     18 mm behind the middle, so with the keyframe's equal targets the rear
#     legs sag more and the body stands 1.1 degrees nose-up. The first
#     policy carried that lean into every gait. With the offset the body
#     settles level at 0.345 m. Front and rear legs are mounted mirrored
#     (knees point at each other), so their joint values have opposite
#     signs; see leg_sign.
#   rest_pose_from_source_key: at all joints zero the CAD model has each
#     shin folded back along its thigh, outside every joint range, and a
#     simulation started there closes the loops on the wrong side.
#   timestep / iterations / closure_solimp: the crank that drives the knee
#     is 33 mm long, so a closure that opens by millimetres is a different
#     leg. With the stock settings (4 ms, 2 iterations, default solimp) the
#     loop stood 24 mm open and the stance sagged 6 cm. These values hold it
#     to 0.3 mm standing and 0.5 mm under random 50 Hz targets; the table
#     is in training/docs/robots.md. They cost about four times the solver
#     work per control step, and the env must run at the same step
#     (task.env.sim_dt, set by the robot=legs_v627 preset).
#   knee_singularity: none. The export's joint ranges already stop the
#     linkage 13 degrees short of its dead points on both sides, and the
#     tasks that read this number do not support this variant yet.
#   height_table: the foot moves straight up and down under the hip, in
#     30 mm steps of hip-to-foot distance, home being the sixth rung. The
#     offsets were solved on the exact loop kinematics (the stock rule,
#     third = 2 x second, swings the foot 17 cm fore and aft on these legs).
#     The heights are the settled ones from the built model, 2 s PD hold at
#     this kp / kd; with rigid joints they would be 0.205-0.389 m.
LEGS_V627 = Robot(
    name="legs_v627",
    foot_body="sixth_link",
    foot_offset=(0.0, 0.0, 0.0255),
    foot_radius=0.02,
    total_mass=16.35,
    kp=60.0,
    kd=2.0,
    forcerange=22.0,
    leg_collision={
        "second_link": dict(type="sphere", size=[0.035, 0, 0], pos=[0.055, 0, 0]),
        "third_link": dict(
            type="capsule", size=[0.03, 0, 0], fromto=[0.0, 0, 0.03, 0.288, 0, 0.03]
        ),
        "fifth_link": dict(
            type="capsule", size=[0.012, 0, 0], fromto=[0.02, 0, 0.0255, 0.29, 0, 0.0255]
        ),
        "sixth_link": dict(
            type="capsule",
            size=[0.015, 0, 0],
            fromto=[0.0, 0, 0.0255, -0.202, 0.087, 0.0255],
        ),
    },
    stand_pose=(
        (0.0, -2.2515898, -2.03678985),
        (0.0, -2.2515898, -2.03678985),
        (0.0, 2.26669567, 1.98215855),
        (0.0, 2.26669567, 1.98215855),
    ),
    settle_drop_height=0.40,
    rest_pose_from_source_key=True,
    timestep=0.002,
    iterations=4,
    ls_iterations=8,
    joint_frictionloss={"fourth_joint": 0.48, "fifth_joint": 0.0},
    closure_solimp=(0.99, 0.999),
    knee_singularity=None,
    has_toggle_angle=False,
    height_table=(0.172, 0.204, 0.238, 0.273, 0.309, 0.345, 0.380),
    dsecond_table=(0.2844, 0.2381, 0.1865, 0.1299, 0.0680, 0.0, -0.0757),
    dthird_table=(-0.9088, -0.7601, -0.6007, -0.4262, -0.2298, 0.0, 0.2899),
    leg_sign=(-1.0, -1.0, 1.0, 1.0),
)
ROBOTS = {r.name: r for r in (WOJTEK, LEGS_V627)}
NAMES = tuple(ROBOTS)


def get(name: str) -> Robot:
    if name not in ROBOTS:
        raise ValueError(f"robot must be one of {NAMES}, got {name!r}")
    return ROBOTS[name]
