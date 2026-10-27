from setuptools import setup

package_name = "wojtek_sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/world.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maciej Gruszczynski",
    maintainer_email="maciejgruszczynski@surferseo.com",
    description="World-side sim + websocket bridge for the agent stack.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_bridge = wojtek_sim_bridge.bridge_node:main",
        ],
    },
)
