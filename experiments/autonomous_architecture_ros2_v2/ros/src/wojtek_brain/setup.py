from setuptools import setup

package_name = "wojtek_brain"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["prompts/bielik/*.txt",
                                 "prompts/bielik/phrases/*.txt"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maciej Gruszczynski",
    maintainer_email="maciejgruszczynski@surferseo.com",
    description="Brain nodes: intent router, Bielik conversational node, VLM agent.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "router = wojtek_brain.router_node:main",
            "bielik = wojtek_brain.bielik_node:main",
            "vlm_agent = wojtek_brain.vlm_agent_node:main",
        ],
    },
)
