from setuptools import setup

package_name = "wojtek_futurenav_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Greg",
    maintainer_email="piotrowskigrzegorz2000@gmail.com",
    description="FutureNav action server to /cmd_vel bridge for sim E2E testing.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "futurenav_bridge = wojtek_futurenav_bridge.bridge_node:main",
        ],
    },
)
