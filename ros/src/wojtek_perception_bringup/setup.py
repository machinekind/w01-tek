from glob import glob

from setuptools import setup

package_name = "wojtek_perception_bringup"

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
    description="Perception pipeline bringup (RealSense D435 + depth reduction) for wojtek.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cloud_reduce_node = wojtek_perception_bringup.cloud_reduce_node:main",
        ],
    },
)
