from glob import glob

from setuptools import setup

package_name = "wojtek_pc"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # glob("config/*") would list the props directory itself, which
        # data_files cannot copy; the two lines below name files only.
        (f"share/{package_name}/config", glob("config/*.*")),
        (f"share/{package_name}/config/props", glob("config/props/*")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
        (f"share/{package_name}/web", glob("web/*.html")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakub Chmielewski",
    maintainer_email="kchmielewski707@gmail.com",
    description="PC-side tooling for wojtek: MuJoCo sim, RViz/PlotJuggler, teleop, consoles.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = wojtek_pc.mujoco_sim_node:main",
            "sim_camera_node = wojtek_pc.sim_camera_node:main",
            "console = wojtek_pc.operator_console:main",
            "web_console = wojtek_pc.web_console:main",
            "sysid_excitation = wojtek_pc.sysid_excitation_node:main",
        ],
    },
)
