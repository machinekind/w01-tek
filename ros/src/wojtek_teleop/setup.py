from glob import glob

from setuptools import setup

package_name = "wojtek_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakub Chmielewski",
    maintainer_email="kchmielewski707@gmail.com",
    description="Robot-side teleop input (bluetooth Xbox pad -> /cmd_vel) for wojtek.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gamepad_teleop = wojtek_teleop.gamepad_teleop:main",
        ],
    },
)
