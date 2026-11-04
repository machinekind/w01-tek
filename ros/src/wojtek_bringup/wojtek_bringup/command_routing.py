"""Optional source topics; the default launch keeps the existing command path."""

from launch.substitutions import LaunchConfiguration, PythonExpression


def command_topic(source):
    if source not in {"pad", "deck", "console", "external", "text", "nav", "brain"}:
        raise ValueError(f"unknown command source {source!r}")
    return PythonExpression([
        f"'/wojtek/cmd_vel/{source}' if '",
        LaunchConfiguration("command_mux", default="false"),
        "'.lower() in ('true', '1') else '/cmd_vel'",
    ])
