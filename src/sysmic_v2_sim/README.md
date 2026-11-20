# sysmic_v2_sim

SYSMIC 2.0 in a RoboCasa kitchen with a ROS 2 Humble bridge, with cameras, lidars
and an IMU. The package is self-contained: the robot description is in `urdf/`
(`sysmic.urdf` + meshes, the Fusion export from
[143dp/sysmicV2-urdf](https://github.com/143dp/sysmicV2-urdf)), used as it is and
never edited by the scripts.

| Folder | What it is |
|---|---|
| `urdf/` | SYSMIC 2.0 URDF, meshes and images |
| `sysmic_v2_robocasa/` | **Standalone RoboCasa robot**: MuJoCo XML / MJCF models + meshes in `assets/`, Python module, sensors config, control panel, demo. Copy it to use SYSMIC 2.0 in any robosuite / RoboCasa setup without ROS ([README](sysmic_v2_robocasa/README.md)) |
| `sim/` | builds the MuJoCo / RoboCasa model from the URDF, simulation server, teleop |
| `scripts/ros2_bridge.py` | ROS 2 node (topics below) |
| `launch/sim.launch.py`, `rviz/` | one-command start, RViz config |

The MuJoCo files ready to use without building anything, all in
`sysmic_v2_robocasa/assets/` (they share `assets/meshes/`):

| File | What it is |
|---|---|
| `robot.xml`, `base.xml`, `left_gripper.xml`, `right_gripper.xml`, `controller.json` | robosuite / RoboCasa robot (loaded by `sysmic_robocasa.py`) |
| `sysmic_v2_scene.xml` | physical MuJoCo model: floor, 15 motors, real wheels, casters and suspension. `python -m mujoco.viewer --mjcf=sysmic_v2_scene.xml` |
| `sysmic_v2.xml` | plain URDF to MJCF conversion, no motors or floor |

Regenerate them after changing the URDF: `cd sim && python sysmic_to_mjcf.py --export`.

## Run

```bash
source /opt/ros/humble/setup.bash
cd ~/sysmic_ws && colcon build --symlink-install
source install/setup.bash
ros2 launch sysmic_v2_sim sim.launch.py            # layout:=5 style:=12 rviz:=false viewer:=false
                                                   # sensors:=/path/to/my_sensors.yaml
```

Do not activate conda in that terminal; the launch file finds the `robocasa` env's
Python by itself. Arguments: `layout`, `style`, `rviz`, `viewer`, `sensors`,
`conda_env`, `conda_python`.

Without ROS, in the conda env:

```bash
conda activate robocasa
cd ~/sysmic_ws/src/sysmic_v2_sim/sim
python sysmic_robosuite.py --no-viewer   # self-test
python sysmic_robosuite.py               # self-test, then kitchen viewer
python sysmic_teleop.py                  # control panel
python sysmic_to_mjcf.py --view          # physical model on an empty floor
```

## ROS 2 interface

Joint names are the URDF names:

| Topic | Type | Notes |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | the 23 movable URDF joints, 30 Hz |
| `/odom`, `/tf` | | `odom -> base_link` (ground truth) |
| `/clock` | `rosgraph_msgs/Clock` | use `use_sim_time:=true` |
| `/cmd_vel` | `geometry_msgs/Twist` | `linear.x` ≤ 0.5 m/s, `angular.z` ≤ 1.0 rad/s; stops after 0.5 s silence |
| `/sysmic/joint_command` | `sensor_msgs/JointState` | `name[]` + `position[]`, joints hold their last target |

Joints for `/sysmic/joint_command`: `lift_joint` (0 to 0.4477 m) and
`arm_{left,right}_{pan,shoulder,elbow,wrist_flex,wrist_roll,gripper}_joint` (rad,
limits from the URDF). The old names (`left_shoulder_lift`, `right_gripper`, ...)
are still accepted.

```bash
ros2 topic pub --once /sysmic/joint_command sensor_msgs/msg/JointState \
  "{name: [lift_joint, arm_left_shoulder_joint, arm_right_gripper_joint], position: [0.35, 1.4, 1.2]}"
```

### Sensors

Configured in `sysmic_v2_robocasa/sensors.yaml` (on/off and every parameter; use
another file with `sensors:=/path/to/file.yaml`). Defaults:

| Topic | Type | Sensor |
|---|---|---|
| `/head_camera/color/image_raw`, `/head_camera/color/camera_info` | `Image` rgb8, `CameraInfo` | RealSense D435i colour, 640x360, 30 Hz |
| `/head_camera/depth/image_raw`, `/head_camera/depth/camera_info` | `Image` 16UC1 (mm), `CameraInfo` | D435i depth, 640x360, 0.3 to 10 m, 30 Hz |
| `/tray_camera/image_raw`, `.../camera_info` | `Image`, `CameraInfo` | webcam under the "SYSMIC" plate, 640x480, 30 Hz |
| `/left_gripper_camera/image_raw`, `/right_gripper_camera/image_raw` (+ `camera_info`) | `Image`, `CameraInfo` | palm webcams, 640x480, 30 Hz |
| `/scan_top`, `/scan_base` | `LaserScan` | YDLIDARs, 720 beams, 10 m, 10 Hz |
| `/imu` | `Imu` | base IMU, 30 Hz (the D435i's IMU, `/head_camera/imu`, is off by default) |

Each sensor has a TF frame (`head_camera_link`, `head_camera_color_optical_frame`,
`lidar_top_link`, `base_imu_link`, ...), added to the robot description from the
same config. A camera is only rendered while its topic has a subscriber: with no
camera watched the simulation keeps real time, each watched camera costs about
10 ms per frame (all five at 30 Hz: about 0.4x real time). RViz shows both scans
and the head camera; tick the other camera displays to see them.

The simulation and the bridge talk over UDP ports 9880 (state) and 9881 (commands),
and TCP 9882 for camera frames, so they do not collide with the original `sysmic_sim`
(9870/9871).

## What the simulation adds to the URDF

Done in memory by `sim/sysmic_to_mjcf.py`; generated files go to
`~/.cache/sysmic_v2_sim` (safe to delete, rebuilt when the URDF changes).

| Addition | Reason |
|---|---|
| Lighter copies of meshes above 60k triangles (vertex clustering) | MuJoCo refuses STL files above 200k triangles; the arm meshes (with screws) have up to 353k |
| Colours | The URDF has no materials |
| Contact cylinders on the 2 drive wheels and 4 casters, sized from the meshes | A mesh hull rolls like a polygon |
| Sprung slide joint (34000 N/m, 12 mm travel) above each drive wheel | Drive wheels (r = 0.085 m, axle at 0.081 m) reach 4 mm below the casters; on the robot that is spring preload |
| Motors: lift, 12 arm/gripper joints (2.9 N m), 2 drive wheels | A URDF has no actuators |

In RoboCasa the base is driven by planar joints (forward, sideways, yaw) like
RoboCasa's own robots; wheels and casters are passive and roll on a private ground
plane. Arms, lift and grippers use robosuite's joint-position controllers.

## Checked

Self-test in RoboCasa kitchen layout 1 / style 1 (MuJoCo 3.3.1):

```
  jitter at rest (rad)            : 0.0019
  hold error (rad)                : 0.0007
  arm tracking error (rad)        : 0.0002
  lift error (m)                  : 0.0000
  closed jaw angle (rad)          : 1.7453
  reversed in 2 s (m)             : 0.9026
  heading error after turn (rad)  : 0.0016
```

Through ROS: `/cmd_vel` -0.25 m/s and 0.4 rad/s gave odometry -0.246 m/s and 0.391 rad/s.

## Notes

- Masses come from the CAD export and are much higher than in the old URDF (base
  117 kg, lift carriage 30 kg, wrist 0.63 kg). The arms still hold their pose within
  the 2.9 N m servo limit, but check these numbers against the real robot.
- At the start pose RoboCasa puts the robot right in front of a counter; driving
  forward hits the cabinet handles. Reverse or turn first.
- `kdl_parser` warns that `base_link` has an inertia. This is harmless for
  `robot_state_publisher`.
