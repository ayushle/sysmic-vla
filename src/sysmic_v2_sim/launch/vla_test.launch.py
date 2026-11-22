"""SYSMIC 2.0 in a RoboCasa kitchen set up for testing the VLA perception pipeline.

    ros2 launch sysmic_v2_sim vla_test.launch.py
    ros2 launch sysmic_v2_sim vla_test.launch.py layout:=2 seed:=3 rviz:=false
    ros2 launch sysmic_v2_sim vla_test.launch.py pose:=none lift:=0.3   # no saved pose

Same as sim.launch.py (simulation server, ROS 2 bridge, robot_state_publisher, RViz),
but the kitchen is the SysmicTableware task from sysmic_v2_robocasa/kitchen_objects.py:
plates, cups, mugs, bowls and cutlery on the counter in front of the robot, plus
AprilTags 36h11 (0-3 flat on the counter, 4 and 5 on 5 cm / 12 cm blocks) for
vla/ (repo root) vla_ros2_node.py.

The robot starts in the pose saved in sysmic_v2_robocasa/vla_test_pose.txt (base offset,
lift, arms, grippers; the left gripper camera looks at the counter). The pose's joints
override `lift`. With a fixed `seed` the kitchen and the object placement are the same on
every launch. The whole kitchen is lowered by `kitchen_drop` (default 0.2 m: counter top
at 0.72 m instead of 0.92 m), so the head RealSense at full lift sees the whole counter.

The helpers (conda lookup, URDF, RViz environment) are taken from sim.launch.py, which
this file does not change.
"""
import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

TASK = "SysmicTableware"


def load_sim_launch():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sim.launch.py")
    spec = importlib.util.spec_from_file_location("sysmic_sim_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_setup(context):
    sim = load_sim_launch()
    share = get_package_share_directory("sysmic_v2_sim")
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    env_name = arg("conda_env")
    python = arg("conda_python") or sim.find_conda_python(env_name)
    if not python or not os.path.isfile(python):
        return [
            LogInfo(msg=(
                f"\n[sysmic_v2_sim] Could not find the Python of conda env '{env_name}'.\n"
                "  Install it with:  bash install_sim.sh   (in the workspace folder)\n"
                "  or pass:          conda_python:=/path/to/envs/robocasa/bin/python\n")),
            EmitEvent(event=Shutdown(reason="conda env not found")),
        ]

    sensors_module = sim.load_sensors_module(share)
    sensors_path = os.path.abspath(os.path.expanduser(
        arg("sensors") or os.path.join(share, "sysmic_v2_robocasa", "sensors.yaml")))
    sensors_cfg = sensors_module.load(sensors_path)

    pose = arg("pose") or os.path.join(share, "sysmic_v2_robocasa", "vla_test_pose.txt")
    extra = []
    if pose.lower() not in ("none", "false", ""):
        extra += ["--pose", os.path.abspath(os.path.expanduser(pose))]
    if float(arg("kitchen_drop") or 0):
        extra += ["--kitchen-drop", arg("kitchen_drop")]
    if arg("seed").lower() not in ("none", "random", ""):
        extra += ["--seed", arg("seed")]
    simulation = ExecuteProcess(
        cmd=[python, os.path.join(share, "sim", "sysmic_sim_server.py"),
             "--task", TASK, "--lift", arg("lift"),
             "--layout", arg("layout"), "--style", arg("style"), "--viewer", arg("viewer")]
            + extra,
        additional_env={"PYTHONPATH": "", "PYTHONUNBUFFERED": "1",
                        "SYSMIC_URDF": os.path.realpath(sim.urdf_path(share)),
                        "SYSMIC_SENSORS": sensors_path},
        output="screen",
    )
    sim_time = {"use_sim_time": True}
    actions = [
        LogInfo(msg=f"[sysmic_v2_sim] VLA test: task {TASK}, layout {arg('layout')}, "
                    f"style {arg('style')}, seed {arg('seed')}, kitchen lowered {arg('kitchen_drop')} m, "
                    f"pose {pose}"),
        LogInfo(msg=f"[sysmic_v2_sim] simulation python: {python}"),
        LogInfo(msg=f"[sysmic_v2_sim] sensors: {sensors_path}"),
        simulation,
        Node(package="sysmic_v2_sim", executable="ros2_bridge.py", output="screen"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": ParameterValue(
                 sim.robot_description(share, sensors_cfg, sensors_module), value_type=str)},
                 sim_time],
             output="screen"),
        RegisterEventHandler(OnProcessExit(
            target_action=simulation,
            on_exit=[EmitEvent(event=Shutdown(reason="simulation closed"))])),
    ]
    if arg("rviz").lower() in ("true", "1", "yes"):
        actions.append(Node(package="rviz2", executable="rviz2",
                            arguments=["-d", os.path.join(share, "rviz", "sysmic.rviz")],
                            parameters=[sim_time], env=sim.desktop_env(), output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("viewer", default_value="true",
                              description="Show the MuJoCo viewer window"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        DeclareLaunchArgument("layout", default_value="1",
                              description="RoboCasa kitchen layout id"),
        DeclareLaunchArgument("style", default_value="1",
                              description="RoboCasa kitchen style id"),
        DeclareLaunchArgument("lift", default_value="0.45",
                              description="lift height (m) held until /sysmic/joint_command "
                                          "sets another one; 0 ... 0.447"),
        DeclareLaunchArgument("pose", default_value="",
                              description="saved robot pose to start in (default: the package's "
                                          "sysmic_v2_robocasa/vla_test_pose.txt; 'none' = no pose)"),
        DeclareLaunchArgument("kitchen_drop", default_value="0.2",
                              description="lower the whole kitchen by this many metres (floor and "
                                          "robot stay), so the head RealSense at full lift sees "
                                          "the whole counter; 0 = RoboCasa's heights"),
        DeclareLaunchArgument("seed", default_value="0",
                              description="RoboCasa seed: same kitchen + objects every launch "
                                          "('random' = different each time)"),
        DeclareLaunchArgument("sensors", default_value="",
                              description="sensors.yaml to use (default: the package's "
                                          "sysmic_v2_robocasa/sensors.yaml)"),
        DeclareLaunchArgument("conda_env", default_value="robocasa",
                              description="Name of the conda env with RoboCasa"),
        DeclareLaunchArgument("conda_python", default_value="",
                              description="Full path to that env's python (auto-detected if empty)"),
        OpaqueFunction(function=launch_setup),
    ])
