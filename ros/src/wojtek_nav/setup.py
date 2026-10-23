from glob import glob

from setuptools import setup

package_name = "wojtek_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakub Chmielewski",
    maintainer_email="kchmielewski707@gmail.com",
    description="Navigation: the rolling costmap, the goto setpoint driver and the pixel-goal resolver.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "goto_node = wojtek_nav.goto_node:main",
            "pixel_goal_node = wojtek_nav.pixel_goal_node:main",
            "vlm_brain_node = wojtek_nav.vlm_brain_node:main",
        ],
    },
)
