from glob import glob

from setuptools import find_packages, setup

package_name = "vln_orchestrator"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", ["config/orchestrator.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anurag",
    maintainer_email="anuragshetye@gmail.com",
    description="CMU VLN Challenge 2026 AI module orchestrator (replaces dummy_vlm).",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orchestrator = vln_orchestrator.orchestrator_node:main",
        ],
    },
)
