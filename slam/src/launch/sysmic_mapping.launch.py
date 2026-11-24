"""RTAB-Map SLAM (odometry + mapping) for the SYSMIC 2.0 robot.

    # 1. simulation (separate terminal)
    ros2 launch sysmic_v2_sim sim.launch.py      # publish_odom must stay false (default):
                                                 # icp_odometry owns the odometry TF
    # 2. mapping
    ros2 launch rtabmap_examples sysmic_mapping.launch.py
    ros2 launch rtabmap_examples sysmic_mapping.launch.py new_map:=true      # erase old db
    ros2 launch rtabmap_examples sysmic_mapping.launch.py rtabmap_viz:=true

Drive around (teleop /cmd_vel); the map is saved to `database_path` on exit (Ctrl+C).
Then localize against it with sysmic_localization.launch.py.

Sensors used
    /scan_top, /scan_base           2x 2D lidar  -> merged 2D scan -> ICP odometry + SLAM + /map
    /head_camera/color + depth      D435i RGB-D  -> visual loop closure detection
    /imu                            base IMU     -> odometry init / gravity
The tray and gripper cameras are not used: they look at the robot itself (tray, gripper
fingers), which would create features that move with the robot and corrupt loop closures.

TF:  map -> icp_odom -> base_link   (lidar ICP odometry only, no wheel odometry)

Extra packages needed (apt):
    ros-$ROS_DISTRO-depth-image-proc  ros-$ROS_DISTRO-pointcloud-to-laserscan

All tunable parameters are in sysmic_mapping_params.yaml (same directory).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

HERE = os.path.dirname(os.path.realpath(__file__))

# ---- robot topics (from sysmic_v2_sim/sysmic_v2_robocasa/sensors.yaml) ----
SCAN_TOP = '/scan_top'
SCAN_BASE = '/scan_base'
IMU = '/imu'
COLOR_IMAGE = '/head_camera/color/image_raw'
COLOR_INFO = '/head_camera/color/camera_info'
DEPTH_IMAGE = '/head_camera/depth/image_raw'
DEPTH_INFO = '/head_camera/depth/camera_info'

# ---- topics created by this launch ----
SCAN_TOP_CLOUD = '/sysmic/scan_top_cloud'
SCAN_BASE_CLOUD = '/sysmic/scan_base_cloud'
SCAN_MERGED = '/sysmic/scan_merged'
DEPTH_REGISTERED = '/head_camera/depth_registered/image_raw'
DEPTH_REGISTERED_INFO = '/head_camera/depth_registered/camera_info'
RGBD_IMAGE = '/sysmic/rgbd_image'
ICP_ODOM = '/icp_odom'
ICP_ODOM_INFO = '/icp_odom_info'


def is_true(value):
    return value.lower() in ('true', '1', 'yes')


def launch_setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    params_file = os.path.expanduser(arg('params_file'))
    database_path = os.path.expanduser(arg('database_path'))
    use_sim_time = is_true(arg('use_sim_time'))
    register_depth = is_true(arg('register_depth'))
    log_level = arg('log_level')
    sim_time = {'use_sim_time': use_sim_time}
    ros_args = ['--ros-args', '--log-level', log_level]

    depth_topic = DEPTH_REGISTERED if register_depth else DEPTH_IMAGE

    # ---------------- preprocessing (one process, intra-process comms) ----------------
    preprocessing = [
        ComposableNode(
            package='pointcloud_to_laserscan',
            plugin='pointcloud_to_laserscan::LaserScanToPointCloudNode',
            name='scan_top_to_cloud',
            parameters=[params_file, sim_time],
            remappings=[('scan_in', SCAN_TOP), ('cloud', SCAN_TOP_CLOUD)]),
        ComposableNode(
            package='pointcloud_to_laserscan',
            plugin='pointcloud_to_laserscan::LaserScanToPointCloudNode',
            name='scan_base_to_cloud',
            parameters=[params_file, sim_time],
            remappings=[('scan_in', SCAN_BASE), ('cloud', SCAN_BASE_CLOUD)]),
        ComposableNode(
            package='rtabmap_util',
            plugin='rtabmap_util::PointCloudAggregator',
            name='scan_merger',
            parameters=[params_file, sim_time],
            remappings=[('cloud1', SCAN_TOP_CLOUD), ('cloud2', SCAN_BASE_CLOUD),
                        ('combined_cloud', SCAN_MERGED)]),
    ]
    if register_depth:
        # D435i colour (69 deg) and depth (87 deg) are separate imagers: reproject the
        # depth image into the colour camera so the pixels line up (needed by RTAB-Map).
        preprocessing.append(ComposableNode(
            package='depth_image_proc',
            plugin='depth_image_proc::RegisterNode',
            name='register_depth',
            parameters=[params_file, sim_time],
            remappings=[('depth/image_rect', DEPTH_IMAGE),
                        ('depth/camera_info', DEPTH_INFO),
                        ('rgb/camera_info', COLOR_INFO),
                        ('depth_registered/image_rect', DEPTH_REGISTERED),
                        ('depth_registered/camera_info', DEPTH_REGISTERED_INFO)]))

    actions = [
        LogInfo(msg=f'[sysmic_mapping] params:   {params_file}'),
        LogInfo(msg=f'[sysmic_mapping] database: {database_path}'
                    + ('  (will be ERASED)' if is_true(arg('new_map')) else '')),
        ComposableNodeContainer(
            name='sysmic_preprocessing',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=preprocessing,
            parameters=[sim_time],
            arguments=ros_args,
            output='screen'),

        Node(
            package='rtabmap_sync', executable='rgbd_sync', name='rgbd_sync',
            output='screen',
            parameters=[params_file, sim_time],
            remappings=[('rgb/image', COLOR_IMAGE),
                        ('rgb/camera_info', COLOR_INFO),
                        ('depth/image', depth_topic),
                        ('rgbd_image', RGBD_IMAGE)],
            arguments=ros_args),
    ]

    actions.append(Node(
        package='rtabmap_odom', executable='icp_odometry', name='icp_odometry',
        output='screen',
        parameters=[params_file, sim_time],
        remappings=[('scan_cloud', SCAN_MERGED),
                    ('imu', IMU),
                    ('odom', ICP_ODOM),
                    ('odom_info', ICP_ODOM_INFO)],
        arguments=ros_args))

    actions.append(Node(
        package='rtabmap_slam', executable='rtabmap', name='rtabmap',
        output='screen',
        parameters=[params_file, sim_time, {
            'database_path': database_path,
            'delete_db_on_start': is_true(arg('new_map')),
            'subscribe_odom_info': True,
        }],
        remappings=[('rgbd_image', RGBD_IMAGE),
                    ('scan_cloud', SCAN_MERGED),
                    ('imu', IMU),
                    ('odom', ICP_ODOM),
                    ('odom_info', ICP_ODOM_INFO)],
        arguments=ros_args))

    if is_true(arg('rtabmap_viz')):
        actions.append(Node(
            package='rtabmap_viz', executable='rtabmap_viz', name='rtabmap_viz',
            output='screen',
            parameters=[sim_time, {
                'frame_id': 'base_link',
                'subscribe_rgbd': True,
                'subscribe_scan_cloud': True,
                'subscribe_odom_info': True,
                'approx_sync': True,
                'sync_queue_size': 30,
            }],
            remappings=[('rgbd_image', RGBD_IMAGE),
                        ('scan_cloud', SCAN_MERGED),
                        ('odom', ICP_ODOM),
                        ('odom_info', ICP_ODOM_INFO)]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(HERE, 'sysmic_mapping_params.yaml'),
            description='YAML with all node / RTAB-Map parameters'),
        DeclareLaunchArgument(
            'database_path', default_value='~/.ros/sysmic_rtabmap.db',
            description='RTAB-Map database the map is written to'),
        DeclareLaunchArgument(
            'new_map', default_value='false',
            description='true = delete the database and start a new map; '
                        'false = continue the existing map (multi-session)'),
        DeclareLaunchArgument(
            'register_depth', default_value='true',
            description='Register D435i depth into the colour frame. Set false only if '
                        'depth is already aligned to colour (same intrinsics and pose)'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rtabmap_viz', default_value='true',
                              description='Start the RTAB-Map GUI'),
        DeclareLaunchArgument('log_level', default_value='info'),
        OpaqueFunction(function=launch_setup),
    ])
