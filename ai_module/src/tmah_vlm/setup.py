from setuptools import setup, find_packages
import os
from glob import glob

package_name = "tmah_vlm"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TMAH",
    maintainer_email="you@example.com",
    description="TMAH team VLM module for CMU VLN Challenge 2026",
    license="BSD",
    entry_points={
        "console_scripts": [
            "tmah_vlm = tmah_vlm.vlm_node:main",
            "scene_graph_json_markers = tmah_vlm.graph.json_marker_node:main",
        ],
    },
)
