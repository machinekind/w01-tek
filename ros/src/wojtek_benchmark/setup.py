from glob import glob

from setuptools import setup

package_name = "wojtek_benchmark"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/tags", glob("tags/*.pdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michał Pogoda",
    maintainer_email="michal.pogoda@bards.ai",
    description="AprilTag ground-truth benchmark instrumentation for wojtek.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "generate_tags = wojtek_benchmark.generate_tags:main",
        ],
    },
)
