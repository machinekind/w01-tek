"""Pure helpers of the step-1 smoke check. Model-free (no ROS import)."""

from wojtek_rai import limits
from wojtek_rai.topics import format_graph, missing


def test_missing_lists_required_names_absent_from_graph():
    graph = [(limits.CMD_VEL_TOPIC, ["geometry_msgs/msg/Twist"])]
    assert missing(graph, limits.REQUIRED_TOPICS) == [
        limits.NAV_COMMAND_TOPIC,
        limits.COLOR_IMAGE_TOPIC,
    ]


def test_missing_is_empty_when_everything_is_present():
    graph = [(n, ["x"]) for n in limits.REQUIRED_TOPICS]
    assert missing(graph, limits.REQUIRED_TOPICS) == []


def test_format_graph_sorts_and_hides_parameter_services():
    out = format_graph(
        [("/b", ["B"]), ("/a", ["A"])],
        [("/n/get_parameters", ["P"]), ("/wojtek/stand_up", ["std_srvs/srv/Trigger"])],
    )
    assert out.splitlines() == [
        "topics:",
        "  /a  A",
        "  /b  B",
        "services:",
        "  /wojtek/stand_up  std_srvs/srv/Trigger",
    ]
