from glob import glob
from setuptools import find_packages, setup

package_name = "sysnav"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=("tests",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="kante",
    maintainer_email="gangtae9050@gmail.com",
    description="Single-room SysNav ROS2 MVP with coverage-based structured scene graph",
    license="Apache-2.0",
    entry_points={"console_scripts": ["sysnav = sysnav.main:main"]},
)
