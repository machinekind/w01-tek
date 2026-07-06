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
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michal Pogoda",
    maintainer_email="michalpogoda@surferseo.com",
    description="RL locomotion policy runner for four_bar_bot (sim + real).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = fbb_policy.policy_node:main",
            "mujoco_sim_node = fbb_policy.mujoco_sim_node:main",
            "real_io_node = fbb_policy.real_io_node:main",
        ],
    },
)
