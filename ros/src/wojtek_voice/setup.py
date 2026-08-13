from setuptools import setup

package_name = "wojtek_voice"

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
    description="Voice pipeline nodes: audio bridge, VAD, ASR.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "audio_bridge = wojtek_voice.audio_bridge_node:main",
            "vad = wojtek_voice.vad_node:main",
            "asr = wojtek_voice.asr_node:main",
        ],
    },
)
