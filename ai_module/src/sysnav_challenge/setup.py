from glob import glob
import os

from setuptools import find_packages, setup


PACKAGE_NAME = "sysnav_challenge"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
        (os.path.join("share", PACKAGE_NAME, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TMAH",
    maintainer_email="tmah@example.com",
    description="CMU VLN Challenge adapter for SysNav",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "challenge_adapter = sysnav_challenge.challenge_adapter:main",
        ],
    },
)
