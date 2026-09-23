"""Robot profiles: the facts that change when the legs change.

There are two sets of legs for the Wojtek body. The stack has to agree with
itself about which one it is driving. The URDF has to show those legs. The
joint map has to match their joint convention. The default policy has to be
one trained on them. The knee clamp only makes sense on legs that have a
knee singularity. A profile keeps these four facts together, one profile per
robot, so a launch picks a robot once with robot:= and the rest follows.

This is the ROS twin of training/wojtek_rl/robots.py. A profile names the
training variant it runs, and a policy's policy_meta.json names the variant
it was trained on. The two must match before a policy reaches the drives.

Standard library only. deploy.sh imports this package on the operator PC,
where there is no ROS and no yaml.
"""

import sys
from dataclasses import dataclass

DEFAULT_ROBOT = "wojtek"


@dataclass(frozen=True)
class RobotProfile:
    name: str
    # The `legs` argument of the wojtek_body xacro macro in
    # wojtek_description/urdf/body.urdf.xacro.
    legs: str
    # The variant name in training/wojtek_rl/robots.py. A policy's
    # policy_meta.json carries it as "robot".
    training_robot: str
    # File name under wojtek_policy/config/ that maps MuJoCo joint angles to
    # URDF ones. It is a file for every robot, the identity map included.
    # Three readers take the map as a yaml path: policy_node, real_io_node
    # and the C++ MuJoCo hardware plugin. A file keeps all three unchanged.
    joint_map: str
    # The pinned default policy, "name@commit" without the organization, or
    # None when the robot has no deployable policy yet. The organization
    # comes from HF_ORGANIZATION at run time (see policy_source.py).
    default_policy: str | None
    # Whether policy_node clamps knee targets at the policy's knee
    # singularity angle.
    clamp_knee: bool
    # The scene a simulated plant loads when model_xml:= is not given, as a
    # package name and a path inside that package's share directory.
    sim_scene: tuple


# The robot as built.
#   default_policy: a bringup runs it when no policy:= is given and no
#     override file is present. The pin is a commit on purpose. An unpinned
#     repo id follows main, so tomorrow's launch could silently run a
#     different policy. Only a pinned commit resolves from the store with no
#     network. The pin lives in this module rather than in the launch files
#     because deploy.sh has to resolve it too.
#   clamp_knee: the four-bar snaps through its singular branch past the
#     knee angle in the policy's contract. Crossing it under load can break
#     the linkage, so the clamp is on for this robot in every bringup.
#   sim_scene: the training scene plus the props the simulated camera has
#     something to see in (wojtek_pc/config/scene_sim.xml).
WOJTEK = RobotProfile(
    name="wojtek",
    legs="stock",
    training_robot="wojtek",
    joint_map="joint_map.yaml",
    default_policy=("wojtek-quiet-locomotion"
                    "@553795b13001cc1f519a4abc0235f275095129f8"),
    clamp_knee=True,
    sim_scene=("wojtek_pc", "config/scene_sim.xml"),
)

# The v6.27 four-bar legs on the same body. Training calls them legs_v627.
#   joint_map: the v6.27 URDF uses MuJoCo's joint angles directly. It has no
#     gear-correction offsets, so the map is the identity.
#   default_policy: none. No exportable policy exists yet. The trained ones
#     observe the gait clock, which this runtime does not implement.
#   clamp_knee: off. These legs have no knee singularity angle. The export's
#     joint ranges stop the linkage short of its dead points on both sides,
#     so their policies carry knee_singularity null and the runtime refuses
#     the clamp for them.
#   sim_scene: the generated training scene for these legs. It has no props.
WOJTEK_V2 = RobotProfile(
    name="wojtek_v2",
    legs="legs_v627",
    training_robot="legs_v627",
    joint_map="joint_map_identity.yaml",
    default_policy=None,
    clamp_knee=False,
    sim_scene=("wojtek_description", "mujoco/legs_v627/scene_mjx.xml"),
)

PROFILES = {p.name: p for p in (WOJTEK, WOJTEK_V2)}
NAMES = tuple(PROFILES)


def get(name: str) -> RobotProfile:
    if name not in PROFILES:
        raise ValueError(f"robot must be one of {NAMES}, got {name!r}")
    return PROFILES[name]


def policy_robot(meta: dict) -> str:
    """The training variant a policy_meta.json was exported for.

    Policies exported before the "robot" field existed were all trained on
    the stock legs. Training reads a run without the field the same way.
    """
    return meta.get("robot") or "wojtek"


def check_policy(meta: dict, robot: str, where: str = "the policy") -> None:
    """Refuse a policy that was trained for other legs than this robot's.

    A policy for the wrong legs must never reach the drives. Its joint
    targets would be angles of another linkage.
    """
    profile = get(robot)
    trained_on = policy_robot(meta)
    if trained_on != profile.training_robot:
        raise ValueError(
            f"{where} was trained for robot {trained_on!r}, but this is "
            f"robot:={profile.name}, which runs {profile.training_robot!r} "
            f"policies. Pick a policy for these legs, or the robot:= that "
            f"matches the policy."
        )


def main(argv=None) -> int:
    """Print a profile's pinned policy, or an empty line when it has none.

    deploy.sh uses this to decide whether there is a default to resolve:
        python3 -m wojtek_policy.robots <robot>
    """
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(f"usage: python3 -m wojtek_policy.robots <{'|'.join(NAMES)}>",
              file=sys.stderr)
        return 2
    try:
        profile = get(args[0])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(profile.default_policy or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
