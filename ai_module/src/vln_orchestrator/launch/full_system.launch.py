"""Coordinated SysNav semantic-exploration bring-up (all ai_module nodes).

Runs against the challenge's PLAIN system (`system_simulation.sh` — sim + base
autonomy that navigates /way_point). Exploration lives here, in ai_module.

Order matters: perception (detection + semantic_mapping) must be live before TARE
explores, or the map misses the run. semantic_mapping loads SAM2 (~60-90 s), so
TARE + the vlm coordinator are started after `explore_delay_s` to let it warm up.

Nodes:
  detection_node            open-vocab detector  -> /detection_result
  semantic_mapping_node     SAM2 + lidar fusion  -> /object_nodes_list
  vlm_node                  exploration coordinator (rooms / target)
  tare_planner + room_seg   semantic exploration -> /way_point   (after delay)
  vln_orchestrator          challenge answerer (external_exploration=true)

All nodes use sim time so they share one clock (avoids detection/odom desync).

Args: scenario (default indoor), explore_delay_s (default 75),
      object_file (detector/mapper vocabulary).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description() -> LaunchDescription:
    scenario = LaunchConfiguration("scenario")
    delay = LaunchConfiguration("explore_delay_s")
    object_file = LaunchConfiguration("object_file")
    boundary_file = LaunchConfiguration("boundary_file")

    # the detector/mapper read their vocab from the SOURCE tree (config/ is not
    # installed to share; nodes resolve it via --symlink-install CONFIG_DIR).
    default_object_file = (
        "/home/docker/ai_module/src/semantic_mapping/semantic_mapping/config/objects.yaml"
    )
    tare_share = get_package_share_directory("tare_planner")
    vlm_share = get_package_share_directory("vlm_node")

    # perception (start immediately; semantic_mapping warms up SAM2)
    detection = Node(
        package="semantic_mapping", executable="detection_node", output="screen",
        parameters=[{"annotate_image": False, "object_file": object_file}],
    )
    mapping = Node(
        package="semantic_mapping", executable="semantic_mapping_node", output="screen",
        parameters=[{"object_file": object_file}],
    )

    # our answerer (waits for the question; does not drive while TARE explores)
    orchestrator = Node(
        package="vln_orchestrator", executable="orchestrator", name="vln_orchestrator",
        output="screen", parameters=[{"external_exploration": True}],
    )

    # exploration brain, started after perception is warm
    vlm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(vlm_share, "vlm_node_sim.launch")),
        launch_arguments={"use_sim_time": "false"}.items(),
    )
    tare = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tare_share, "explore_world_sim.launch")),
        launch_arguments={"scenario": scenario, "boundary_file": boundary_file}.items(),
    )
    room_seg = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tare_share, "room_segmentation.launch")),
        launch_arguments={"scenario": scenario}.items(),
    )
    explore_group = TimerAction(period=delay, actions=[vlm, tare, room_seg])

    return LaunchDescription([
        # matterport_sim carries the room-segmentation params (room_resolution,
        # ...) SysNav's room-aware TARE needs; indoor.yaml is the base config.
        DeclareLaunchArgument("scenario", default_value="matterport_sim"),
        DeclareLaunchArgument("explore_delay_s", default_value="75.0"),
        DeclareLaunchArgument("object_file", default_value=default_object_file),
        # exploration boundary — MUST match the running scene. Default = shipped
        # boundary_sim.ply; pass a scene-matched one (derived from the scene's
        # traversable_area.ply) e.g. data/boundary_sim_livingroom_3.ply.
        DeclareLaunchArgument(
            "boundary_file",
            default_value=os.path.join(tare_share, "boundary_sim.ply")),
        # The CMU sim publishes no /clock, so use_sim_time:=true makes nodes see
        # time 0 (TARE: "Start time is zero"). Use the wall clock; desync is avoided
        # by one coordinated launch (never restart the sim under live perception).
        SetParameter(name="use_sim_time", value=False),
        detection,
        mapping,
        orchestrator,
        explore_group,
    ])
