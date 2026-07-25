from glob import glob

from setuptools import setup

package_name = "wojtek_viz"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
        (f"share/{package_name}/web", glob("web/*.html")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakub Chmielewski",
    maintainer_email="kchmielewski707@gmail.com",
    description="PC-side viz/debug/sim (RViz, PlotJuggler, teleop, MuJoCo) for wojtek.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = wojtek_viz.mujoco_sim_node:main",
            "console = wojtek_viz.operator_console:main",
            "web_console = wojtek_viz.web_console:main",
            "gamepad_teleop = wojtek_viz.gamepad_teleop:main",
            "sysid_excitation = wojtek_viz.sysid_excitation_node:main",
        ],
    },
)
