from setuptools import setup

package_name = "wojtek_odometry"

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
    maintainer="Grzegorz Piotrowski",
    maintainer_email="piotrowskigrzegorz2000@gmail.com",
    description="Leg-kinematics + IMU odometry for Wojtek.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "leg_odometry_node = wojtek_odometry.leg_odometry_node:main",
            "odom_vs_ground_truth = wojtek_odometry.odom_vs_ground_truth:main",
            "odom_trace = wojtek_odometry.odom_trace:main",
        ],
    },
)
