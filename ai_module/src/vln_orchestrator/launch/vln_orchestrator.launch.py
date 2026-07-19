"""Launch the VLN orchestrator node (replaces dummy_vlm.launch).

In the system startup script, launch this instead of dummy_vlm:
    ros2 launch vln_orchestrator vln_orchestrator.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(
        get_package_share_directory("vln_orchestrator"),
        "config",
        "orchestrator.yaml",
    )
    return LaunchDescription([
        Node(
            package="vln_orchestrator",
            executable="orchestrator",
            name="vln_orchestrator",
            output="screen",
            parameters=[config],
        ),
    ])
