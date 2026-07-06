from setuptools import setup
import os
from glob import glob

package_name = "tmah_vlm"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch 파일 설치
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TMAH",
    maintainer_email="you@example.com",
    description="TMAH team VLM module for CMU VLN Challenge 2026",
    license="BSD",
    entry_points={
        "console_scripts": [
            # `ros2 run tmah_vlm tmah_vlm` 로 실행됨
            "tmah_vlm = tmah_vlm.vlm_node:main",
        ],
    },
)
