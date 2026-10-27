from setuptools import setup

package_name = "wojtek_agent_perf"

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
    maintainer="Maciej Gruszczynski",
    maintainer_email="maciejgruszczynski@surferseo.com",
    description="Passive latency probe for the agent pipeline.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "probe = wojtek_agent_perf.probe_node:main",
        ],
    },
)
