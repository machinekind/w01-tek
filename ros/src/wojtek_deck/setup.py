from glob import glob

from setuptools import setup

package_name = "wojtek_deck"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # The whole page: html, css and the JS modules it imports.
        (f"share/{package_name}/web", glob("web/*.html") + glob("web/*.css")
         + glob("web/*.js")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Marcin Wysocki",
    maintainer_email="wysocki@undef.pl",
    description="Deck panel: browser cockpit for a handheld on the robot's wifi, "
                "and its robot-side command gateway.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "deck_gateway = wojtek_deck.deck_gateway:main",
        ],
    },
)
