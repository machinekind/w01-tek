from glob import glob

from setuptools import setup

package_name = "fbb_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michal Pogoda",
    maintainer_email="michalpogoda@surferseo.com",
    description="RL locomotion policy for four_bar_bot (numpy runtime + ROS node).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = fbb_policy.policy_node:main",
            "xbox_teleop_node = fbb_policy.xbox_teleop_node:main",
        ],
    },
)
