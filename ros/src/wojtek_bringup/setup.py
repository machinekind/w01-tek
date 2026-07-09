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
    description="Bringup (real + MuJoCo sim) for four_bar_bot / wojtek_policy.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = wojtek_bringup.mujoco_sim_node:main",
            "real_io_node = wojtek_bringup.real_io_node:main",
        ],
    },
)
