# VLA perception test in simulation

The SYSMIC robot stands at a kitchen counter with plates, bowls, mugs, cups and cutlery and
AprilTags on it. The VLA pipeline (`vla/vla_ros2_node.py`) uses the head RealSense
camera to find one object and publishes its point cloud.

## 0. Install everything (new machine, once)

**Needs:** Ubuntu 24.04, an NVIDIA GPU with about 8 GB and its driver installed
(`nvidia-smi` works), about 30 GB of free disk, internet. Run the blocks in order.
All paths below assume the workspace is at `~/sysmic-sim/sysmic_ws`.

```bash
# 1. ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), then:
sudo apt install -y ros-jazzy-desktop ros-jazzy-sensor-msgs-py python3-colcon-common-extensions \
    python3-venv git curl
# optional, to move the robot by hand:
sudo apt install -y ros-jazzy-joint-state-publisher-gui ros-jazzy-teleop-twist-keyboard

# 2. Miniconda (skip if `conda --version` works)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init && source ~/.bashrc

# 3. Simulation env: robosuite + RoboCasa (Python 3.11, versions this was tested with)
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa
mkdir -p ~/robocasa_ws && cd ~/robocasa_ws
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && git checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c && pip install -e . && cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa && git checkout 7bfc60c9d26c02506d4265cbe0e86b35e6e25d42 && pip install -e .
pip install pyyaml opencv-python          # used by the tableware kitchen (AprilTags)
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets     # about 10 GB
conda deactivate

# 4. This workspace (new terminal, no conda env active)
source /opt/ros/jazzy/setup.bash
git clone -b v2 https://github.com/rahulrajak6942/sysmic_ws.git ~/sysmic-sim/sysmic_ws
cd ~/sysmic-sim/sysmic_ws && colcon build --symlink-install

# 5. VLA pipeline: Python 3.12 venv at ~/envs/cvenev with torch, transformers, OpenCV,
#    plus Depth Anything V2 code and weights (372 MB). See vla/README.md.
bash ~/sysmic-sim/sysmic_ws/vla/setup_vla.sh
```

The first pipeline run also downloads Grounding DINO and SAM from Hugging Face (about 1.5 GB).
Tested with: Ubuntu 24.04, ROS 2 Jazzy, robosuite 1.5.2, RoboCasa 1.0.1, MuJoCo 3.3.1,
torch 2.15 (CUDA 13.2), transformers 4.51.3, on an RTX 5060 laptop GPU (8 GB).

## 0b. After changing code in `src/`

```bash
cd ~/sysmic-sim/sysmic_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select sysmic_v2_sim
```

## 1. Start the simulation (terminal 1)

```bash
cd ~/sysmic-sim/sysmic_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sysmic_v2_sim vla_test.launch.py
```

Wait for `SYSMIC simulation server running`.

| Argument | Default | What it does |
|---|---|---|
| `seed` | `0` | Same kitchen and object placement every launch. `random` gives a new one each time. |
| `layout`, `style` | `1`, `1` | RoboCasa kitchen layout and look. |
| `kitchen_drop` | `0.2` | Lowers the whole kitchen by this many metres so the RealSense sees the whole counter. `0` = normal heights. |
| `pose` | `vla_test_pose.txt` | Robot pose to start in (lift, arms, base). `none` = RoboCasa's start pose. |
| `lift` | `0.45` | Lift height when `pose:=none` (a pose overrides it). |
| `viewer`, `rviz` | `true` | Show the MuJoCo window / RViz. |

## 2. Start the VLA pipeline (terminal 2)

```bash
cd ~/sysmic-sim/sysmic_ws/vla
source /opt/ros/jazzy/setup.bash
~/envs/cvenev/bin/python vla_ros2_node.py mug --ros-args \
  -p image_topic:=/head_camera/color/image_raw \
  -p camera_info_topic:=/head_camera/color/camera_info \
  -p image_is_rectified:=true \
  -p use_sim_time:=true
```

It runs in a loop: each new camera frame is processed (about 1–2 s each) and the object's
cloud is republished. **To change the object, stop it (Ctrl+C) and start it again with
another name.** Names with spaces go in quotes: `"red block"`.

| Argument | Meaning |
|---|---|
| `mug` (first word) | The object to find, in plain words. Overrides `object_name`. |
| `image_topic` | Camera image to use. `/head_camera/color/image_raw` = RealSense colour (1920×1080). |
| `camera_info_topic` | That camera's intrinsics. Always pair it with the image topic. |
| `image_is_rectified` | `true` in simulation (the sim's cameras have no lens distortion). |
| `use_sim_time` | `true` in simulation. |
| `mode` | `continuous` (default) = loop forever. `trigger` = only when you call `ros2 service call /vla_perception/capture std_srvs/srv/Trigger`. |
| `scene_stride` | Pixel step for the whole-scene cloud (default 4; 1 = every pixel, much bigger). |
| `low_vram` | `true` unloads models between uses: slower, but fits small GPUs. |

What it publishes (all under `/vla_perception/`):

| Topic | Contents |
|---|---|
| `object_cloud` | Point cloud of the object (x y z rgb, camera frame). **This is the main output.** |
| `scene_cloud` | Point cloud of the whole view. |
| `depth`, `mask` | Metric depth image (m) and the object mask. |
| `status_image` | Camera image with tags, mask and calibration drawn on it. Good for checking. |
| `calib` | JSON: which tags were seen and how good the depth calibration is. |

To see it: in RViz add a PointCloud2 display on `/vla_perception/object_cloud`.

## 3. Other things you may need

Commands below are run from `~/sysmic-sim/sysmic_ws`.

- **Only one pipeline at a time.** The models need most of the 8 GB GPU.
- **Only one simulation at a time.** A second one fails with `Address already in use`.
- **Calibration needs the tags.** The pipeline needs at least 3 of AprilTags 0–3 in view; with
  4 or more it uses the more accurate "dense table plane" fit. Check `status_image` if numbers
  look wrong. The tag size (35 mm) and ids are set in `vla/vla-pipeline-v2.py`.
- **Save a new start pose.** Move the robot (e.g. with `joint_state_publisher_gui` on
  `/sysmic/joint_command` and `teleop_twist_keyboard`), then, with the simulation running:
  ```bash
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 src/sysmic_v2_sim/sysmic_v2_robocasa/save_pose.py \
      src/sysmic_v2_sim/sysmic_v2_robocasa/vla_test_pose.txt
  ```
  The next launch starts in it.
- **Measure accuracy against the simulation's ground truth.** Start the pipeline with
  `-p mode:=trigger` (and no object name), then:
  ```bash
  ~/envs/cvenev/bin/python src/sysmic_v2_sim/sysmic_v2_robocasa/vla_eval.py --camera head_camera_color --runs 3
  ```
  It writes `report.md` (mask IoU, depth MSE / RMSE, centroid error per object) to
  `~/sysmic-sim/vla_eval/<date>/`.
- **Camera settings** (resolution, tilt) are in `src/sysmic_v2_sim/sysmic_v2_robocasa/sensors.yaml`
  under `head_camera`. Restart the simulation after editing it.
