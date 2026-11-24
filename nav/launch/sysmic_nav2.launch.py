"""Nav2 for the SYSMIC 2.0 robot. Navigation ONLY: no localization of any kind.

This launch does not start AMCL, RTAB-Map or any scan matching. It trusts whatever
localization you already run and simply follows TF:

    map -> icp_odom          from YOUR RTAB-Map localization (started before this)
    icp_odom -> base_link    from YOUR icp_odometry (lidar only, no wheel odometry)

Nav2 uses ICP odometry as its odometry: /icp_odom for velocity, frame icp_odom for the
local costmap, recoveries and collision monitor.

    # 1. simulation
    ros2 launch sysmic_v2_sim sim.launch.py
    # 2. your localization; wait until the robot is localized
    ros2 launch ~/sysmic-sim/slam/src/launch/sysmic_localization.launch.py
    # 3. navigation
    ros2 launch ~/sysmic-sim/nav/launch/sysmic_nav2.launch.py

    # options
    ... sysmic_nav2.launch.py depth_obstacles:=false              # lidars only
    ... sysmic_nav2.launch.py collision_monitor:=false
    ... sysmic_nav2.launch.py rviz:=false

What runs
    depth obstacle cloud    /head_camera/depth/image_raw
                            -> rtabmap_util PointCloudXYZ (1/4 resolution, 5 cm voxels; just
                               depth image -> points, no SLAM / matching)
                            -> pcl_ros CropBox: removes the robot itself (tray, arms, lift)
                            -> /sysmic/head_camera/points      (base_link)
    Nav2                    planner (Smac 2D), controller (RotationShim + MPPI),
                            smoother, behaviors, bt_navigator, waypoint_follower,
                            velocity_smoother, collision_monitor
    cmd_vel chain           controller -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel_smoothed
                            -> collision_monitor -> /cmd_vel -> simulation bridge
                            (collision_monitor:=false: velocity_smoother -> /cmd_vel)

All Nav2 parameters: ~/sysmic-sim/nav/config/sysmic_nav2_params.yaml
(odom_frame / odom_topic below override the odometry entries in it)

Needs (apt): ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-pcl-ros ros-jazzy-rtabmap-ros
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

HERE = os.path.dirname(os.path.realpath(__file__))
NAV_DIR = os.path.dirname(HERE)

DEPTH_IMAGE = '/head_camera/depth/image_raw'
DEPTH_INFO = '/head_camera/depth/camera_info'
DEPTH_CLOUD_RAW = '/sysmic/head_camera/points_raw'
DEPTH_CLOUD = '/sysmic/head_camera/points'      # used by both costmaps and the collision monitor

# Box around the whole robot in base_link (m), lift fully up, arms tucked, + 3 cm.
# Depth points inside it are the robot itself (tray, things on the tray, grippers).
ROBOT_BOX = dict(min_x=-0.31, max_x=0.29, min_y=-0.29, max_y=0.29, min_z=-0.05, max_z=1.50)


def is_true(value):
    return value.lower() in ('true', '1', 'yes')


def prepare_params(params_file, odom_frame, odom_topic, use_depth):
    """Copy of the params file with the odometry frame/topic set everywhere Nav2 uses
    them and, for depth_obstacles:=false, the 'depth' observation source removed."""
    with open(params_file) as f:
        data = yaml.safe_load(f)

    def section(name):
        return data[name]['ros__parameters']

    # odometry topic: robot velocity for the controller / BT / smoother
    for node in ('bt_navigator', 'controller_server', 'velocity_smoother'):
        section(node)['odom_topic'] = odom_topic
    # odometry frame: continuous (never jumps), used for local planning
    data['local_costmap']['local_costmap']['ros__parameters']['global_frame'] = odom_frame
    section('behavior_server')['local_frame'] = odom_frame
    section('collision_monitor')['odom_frame_id'] = odom_frame

    if not use_depth:
        def strip(node):
            if isinstance(node, dict):
                src = node.get('observation_sources')
                if isinstance(src, str):
                    node['observation_sources'] = ' '.join(s for s in src.split() if s != 'depth')
                    node.pop('depth', None)
                elif isinstance(src, list):
                    node['observation_sources'] = [s for s in src if s != 'depth']
                    node.pop('depth', None)
                for value in node.values():
                    strip(value)
        strip(data)

    out = tempfile.NamedTemporaryFile('w', prefix='sysmic_nav2_', suffix='.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    return out.name


def launch_setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    use_sim_time = is_true(arg('use_sim_time'))
    sim_time = {'use_sim_time': use_sim_time}
    log_level = arg('log_level')
    ros_args = ['--ros-args', '--log-level', log_level]
    use_cm = is_true(arg('collision_monitor'))
    use_depth = is_true(arg('depth_obstacles'))
    odom_frame, odom_topic = arg('odom_frame'), arg('odom_topic')
    source = os.path.expanduser(arg('params_file'))
    params = prepare_params(source, odom_frame, odom_topic, use_depth)

    actions = [
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        LogInfo(msg=f'[sysmic_nav2] params: {source}'),
        LogInfo(msg=f'[sysmic_nav2] odometry: topic {odom_topic}, frame {odom_frame}. '
                    'No localization is started here: map -> ' + odom_frame +
                    ' must come from your RTAB-Map.'),
    ]

    # ---------------- depth camera -> obstacle cloud without the robot ----------------
    if use_depth:
        actions.append(ComposableNodeContainer(
            name='sysmic_depth_obstacles',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            arguments=ros_args,
            output='screen',
            composable_node_descriptions=[
                ComposableNode(
                    package='rtabmap_util',
                    plugin='rtabmap_util::PointCloudXYZ',
                    name='depth_to_cloud',
                    parameters=[sim_time, {
                        'decimation': 4,          # 1280x720 -> 320x180
                        'voxel_size': 0.05,       # m, same as the costmap cells
                        'min_depth': 0.3,         # D435i minimum range
                        'max_depth': 4.0,         # far depth is noisy and not needed locally
                        'approx_sync': False,     # image and info have the same stamp
                    }],
                    remappings=[('depth/image', DEPTH_IMAGE),
                                ('depth/camera_info', DEPTH_INFO),
                                ('cloud', DEPTH_CLOUD_RAW)],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::CropBox',
                    name='robot_self_filter',
                    parameters=[sim_time, ROBOT_BOX, {
                        'negative': True,         # keep what is OUTSIDE the box
                        'keep_organized': False,
                        'input_frame': 'base_link',
                        'output_frame': 'base_link',
                    }],
                    remappings=[('input', DEPTH_CLOUD_RAW), ('output', DEPTH_CLOUD)],
                    extra_arguments=[{'use_intra_process_comms': True}]),
            ]))

    # ---------------- Nav2 ----------------
    tf_remaps = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    def nav2_node(package, executable, name, remaps=()):
        # use_sim_time passed explicitly: the YAML has no use_sim_time keys, and without
        # it the collision monitor compares sim-time scans with wall time and drops them
        return Node(package=package, executable=executable, name=name, output='screen',
                    parameters=[params, sim_time], arguments=ros_args,
                    remappings=tf_remaps + list(remaps))

    actions += [
        nav2_node('nav2_controller', 'controller_server', 'controller_server',
                  [('cmd_vel', 'cmd_vel_nav')]),
        nav2_node('nav2_smoother', 'smoother_server', 'smoother_server'),
        nav2_node('nav2_planner', 'planner_server', 'planner_server'),
        nav2_node('nav2_behaviors', 'behavior_server', 'behavior_server',
                  [('cmd_vel', 'cmd_vel_nav')]),
        nav2_node('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
        nav2_node('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower'),
        # velocity_smoother: in cmd_vel_nav, out cmd_vel_smoothed (or /cmd_vel directly)
        nav2_node('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother',
                  [('cmd_vel', 'cmd_vel_nav')]
                  + ([] if use_cm else [('cmd_vel_smoothed', 'cmd_vel')])),
    ]
    lifecycle_nodes = ['controller_server', 'smoother_server', 'planner_server',
                       'behavior_server', 'velocity_smoother', 'bt_navigator',
                       'waypoint_follower']
    if use_cm:
        actions.append(nav2_node('nav2_collision_monitor', 'collision_monitor',
                                 'collision_monitor'))
        lifecycle_nodes.append('collision_monitor')

    actions.append(Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen', arguments=ros_args,
        parameters=[sim_time, {'autostart': is_true(arg('autostart')),
                               'node_names': lifecycle_nodes,
                               'bond_timeout': 10.0}]))

    # ---------------- RViz ----------------
    if is_true(arg('rviz')):
        rviz_cfg = os.path.join(get_package_share_directory('nav2_bringup'),
                                'rviz', 'nav2_default_view.rviz')
        actions.append(Node(package='rviz2', executable='rviz2', name='rviz2_nav',
                            arguments=['-d', rviz_cfg], parameters=[sim_time],
                            output='log'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(NAV_DIR, 'config', 'sysmic_nav2_params.yaml'),
            description='Nav2 parameters'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('odom_frame', default_value='icp_odom',
                              description='Odometry frame Nav2 plans locally in '
                                          '(icp_odom = ICP odometry)'),
        DeclareLaunchArgument('odom_topic', default_value='/icp_odom',
                              description='Odometry topic Nav2 reads velocity from'),
        DeclareLaunchArgument('depth_obstacles', default_value='true',
                              description='D435i depth -> obstacle cloud for costmaps and '
                                          'collision monitor (false = lidars only; the '
                                          '"depth" sources are dropped from the params)'),
        DeclareLaunchArgument('collision_monitor', default_value='true',
                              description='Safety stop/slowdown between Nav2 and /cmd_vel'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('log_level', default_value='info'),
        OpaqueFunction(function=launch_setup),
    ])
