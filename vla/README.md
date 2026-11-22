# VLA perception pipeline

Finds one object in a camera image and publishes its 3D point cloud on ROS 2. Depth comes
from a single colour camera: Depth Anything V2 gives relative depth, AprilTags on the table
make it metric. Grounding DINO + SAM find the object from its name.

## Files

```
vla/
├── vla_ros2_node.py     ROS 2 node: camera topic in, object point cloud out (start here)
├── vla-pipeline-v2.py   the pipeline the node loads (camera model, tags, depth, segmentation)
├── requirements.txt     Python packages (tested versions)
├── setup_vla.sh         one-time setup: venv, Depth Anything V2 code + weights
├── third_party/         Depth-Anything-V2 checkout   (made by setup_vla.sh, not in git)
└── models/              depth_anything_v2_vitb.pth   (made by setup_vla.sh, not in git)
```

## Needs

- Ubuntu 24.04 with ROS 2 Jazzy (`rclpy`, `sensor_msgs`, `std_msgs`, `std_srvs`)
- Python 3.12, an NVIDIA GPU with about 8 GB (CPU works, but very slowly)
- Internet on the first run: the Depth Anything weights (setup script) and Grounding DINO
  (`IDEA-Research/grounding-dino-base`) + SAM (`facebook/sam-vit-base`) from Hugging Face

## Setup (once)

```bash
cd ~/sysmic-sim/sysmic_ws/vla
bash setup_vla.sh            # makes ~/envs/cvenev; pass another path to put the venv elsewhere
```

Model locations can be overridden with `VLA_DEPTH_WEIGHTS=/path/to/depth_anything_v2_vitb.pth`
and `VLA_DEPTH_REPO=/path/to/Depth-Anything-V2`. Without them the pipeline uses `models/` and
`third_party/` here, else `~/cnn/depth/`.

## Run

```bash
source /opt/ros/jazzy/setup.bash
~/envs/cvenev/bin/python vla_ros2_node.py mug --ros-args \
  -p image_topic:=/head_camera/color/image_raw \
  -p camera_info_topic:=/head_camera/color/camera_info \
  -p image_is_rectified:=true -p use_sim_time:=true
```

The arguments and outputs are explained in [../VLA_TEST.md](../VLA_TEST.md) (simulation) and
at the top of `vla_ros2_node.py`. For a real camera, drop `image_is_rectified` and
`use_sim_time`, and set the calibration (`CAMERA_MATRIX`, `DIST_COEFFS`), the tag size
(`TAG_SIZE_METERS`) and tag ids at the top of `vla-pipeline-v2.py`, or pass the camera's
`camera_info_topic`.
