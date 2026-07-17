import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_legacy = LaunchConfiguration("use_legacy_solvers")
    use_vlm = LaunchConfiguration("use_sysnav_vlm")
    auto_start = LaunchConfiguration("auto_start")
    single_room = LaunchConfiguration("single_room")
    use_rviz = LaunchConfiguration("use_rviz")
    tare_share = get_package_share_directory("tare_planner")
    semantic_share = get_package_share_directory("semantic_mapping")
    challenge_share = get_package_share_directory("sysnav_challenge")
    tare_config = os.path.join(tare_share, "matterport_sim.yaml")
    semantic_config = os.path.join(semantic_share, "mapping_mecanum_sim.yaml")
    object_file = os.path.join(challenge_share, "config", "challenge_objects.yaml")

    common_tare_overrides = {
        # /state_estimation is explicitly allowed by the challenge API.
        "sub_state_estimation_topic_": "/state_estimation",
        "sub_registered_scan_topic_": "/registered_scan",
        "sub_terrain_map_topic_": "/terrain_map",
        "sub_terrain_map_ext_topic_": "/terrain_map_ext",
        "pub_waypoint_topic_": "/way_point",
        "kAutoStart": ParameterValue(auto_start, value_type=bool),
        "kSingleRoomMode": ParameterValue(single_room, value_type=bool),
        # The released SysNav boundary does not match held-out challenge scenes.
        "kUseCoverageBoundaryOnFrontier": False,
    }

    return LaunchDescription([
        DeclareLaunchArgument("use_legacy_solvers", default_value="true"),
        DeclareLaunchArgument("use_sysnav_vlm", default_value="true"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("single_room", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        Node(
            package="sysnav_challenge",
            executable="challenge_adapter",
            name="sysnav_challenge_adapter",
            output="screen",
        ),
        Node(
            package="tare_planner",
            executable="tare_planner_node",
            name="tare_planner_node",
            output="screen",
            parameters=[tare_config, common_tare_overrides],
        ),
        Node(
            package="tare_planner",
            executable="room_segmentation",
            name="room_segmentation",
            output="screen",
            parameters=[tare_config, common_tare_overrides],
        ),
        Node(
            package="semantic_mapping",
            executable="detection_node",
            name="sysnav_detection_node",
            output="screen",
            parameters=[semantic_config, {
                "platform": "mecanum_sim",
                "object_file": object_file,
            }],
        ),
        Node(
            package="semantic_mapping",
            executable="semantic_mapping_node",
            name="sysnav_semantic_mapping_node",
            output="screen",
            parameters=[semantic_config, {
                "platform": "mecanum_sim",
                "object_file": object_file,
            }],
        ),
        Node(
            package="vlm_node",
            executable="vlm_reasoning_node",
            name="sysnav_vlm_node",
            output="screen",
            parameters=[{
                "platform": "mecanum",
                "object_file": object_file,
            }],
            condition=IfCondition(use_vlm),
        ),
        # Keep the existing numerical solver, but hide the public challenge
        # topic from it. The adapter sends only numerical questions here.
        Node(
            package="tmah_vlm",
            executable="tmah_vlm",
            name="legacy_numerical_solver",
            output="screen",
            remappings=[
                ("/challenge_question", "/sysnav_challenge/legacy_question"),
                # SysNav alone owns navigation while this launch is active.
                ("/way_point_with_heading", "/sysnav_challenge/unused_waypoint"),
                ("/selected_object_marker", "/sysnav_challenge/legacy_marker"),
            ],
            condition=IfCondition(use_legacy),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="sysnav_map_rviz",
            arguments=["-d", os.path.join(challenge_share, "rviz", "single_room_map.rviz")],
            output="screen",
            condition=IfCondition(use_rviz),
        ),
    ])
