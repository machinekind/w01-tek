import glob

from setuptools import setup

package_name = "wojtek_agent_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob.glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maciej Gruszczynski",
    maintainer_email="maciejgruszczynski@surferseo.com",
    description="Launch files for the agentic voice/brain stack.",
    license="Apache-2.0",
)
