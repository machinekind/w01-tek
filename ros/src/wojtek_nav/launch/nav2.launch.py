"""Nav2 without a map: the planner and the follower on the rolling perception.

    ros2 launch wojtek_nav nav2.launch.py
    ros2 launch wojtek_nav nav2.launch.py --show-args
    ros2 launch wojtek_pc sim.launch.py model_xml:=scene_nav.xml leg_odom:=true nav2:=true

Runs standalone against a running point cloud (/wojtek/nav/points, from
costmap.launch.py) and odometry (odom->base_link), and is meant to be
included by the robot/sim bringup (nav2:=true), which also turns the
standalone costmap and goto_node OFF: Nav2 keeps its own costmaps, and two
drivers on one /cmd_vel would fight.

    /wojtek/nav/points ──► local_costmap (6x6, odom)  ──► controller_server (RPP) ──► /cmd_vel
                       └─► global_costmap (10x10, odom) ──► planner_server (NavFn) ──► /plan
    /goal_pose (RViz), NavigateToPose action ──► bt_navigator ──► the two above + behavior_server

Four lifecycle nodes and their manager, all stock nav2, configured from
config/nav2.yaml (what every number follows from is argued there). No
map_server, no AMCL, no smoother, no velocity smoother, no collision
monitor: the file says why.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "wojtek_nav"

# The lifecycle nodes, in the order the manager brings them up. The
# controller and planner own the costmaps, so they come first.
LIFECYCLE_NODES = [
    "controller_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
]

_SERVERS = {
    "controller_server": ("nav2_controller", "controller_server"),
    "planner_server": ("nav2_planner", "planner_server"),
    "behavior_server": ("nav2_behaviors", "behavior_server"),
    "bt_navigator": ("nav2_bt_navigator", "bt_navigator"),
}


def _setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    # Same treatment as the perception nodes: off the isolated RT cores when
    # the bringup asks for it.
    cpus = arg("cpus")
    prefix = [f"taskset -c {cpus}"] if cpus else None
    params = arg("nav2_params_file")
    log_level = arg("log_level")

    actions = [
        Node(
            package=pkg,
            executable=exe,
            name=name,
            parameters=[params],
            arguments=["--ros-args", "--log-level", log_level],
            prefix=prefix,
            output="screen",
        )
        for name, (pkg, exe) in _SERVERS.items()
    ]
    actions.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            parameters=[{
                "autostart": arg("autostart").lower() in ("true", "1"),
                "node_names": LIFECYCLE_NODES,
                # A node that stops answering is brought down and back up
                # by the manager, not left half-active.
                "bond_timeout": 4.0,
            }],
            prefix=prefix,
            output="screen",
        )
    )
    return actions


def generate_launch_description():
    share = get_package_share_directory(PKG)
    return LaunchDescription(
        [
            # Not `params_file`: an included launch shares its parent's
            # configurations, and costmap.launch.py, included alongside,
            # declares that name first -- the nav2 servers would silently
            # read costmap.yaml. (Seen for real: DWB defaults, 20 Hz.)
            DeclareLaunchArgument(
                "nav2_params_file", default_value=f"{share}/config/nav2.yaml",
                description="Nav2 settings; see the file's commentary.",
            ),
            DeclareLaunchArgument(
                "autostart", default_value="true",
                description="Configure and activate the lifecycle nodes at "
                            "start; false leaves them unconfigured for "
                            "`ros2 lifecycle`.",
            ),
            DeclareLaunchArgument(
                "log_level", default_value="info",
            ),
            DeclareLaunchArgument(
                "cpus", default_value="",
                description="CPU affinity for all nav2 nodes (comma list); "
                            "empty = inherit.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
