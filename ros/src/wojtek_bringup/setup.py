from glob import glob

from setuptools import setup

package_name = "wojtek_bringup"

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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakub Chmielewski",
    maintainer_email="kchmielewski707@gmail.com",
    description="Robot-side bringup (MD80 + IMU via ros2_control) for wojtek / wojtek_policy.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "real_io_node = wojtek_bringup.real_io_node:main",
            # One-command robot bring-up from the PC dev container:
            #   ros2 run wojtek_bringup robot
            "robot = wojtek_bringup.robot:main",
        ],
    },
)
