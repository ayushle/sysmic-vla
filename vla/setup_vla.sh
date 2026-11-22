#!/usr/bin/env bash
# One-time setup for the VLA pipeline (vla_ros2_node.py). Safe to run again.
#
#   bash setup_vla.sh                 # venv at ~/envs/cvenev
#   bash setup_vla.sh /path/to/venv   # somewhere else
#
# 1. Python venv with requirements.txt (it can see the ROS 2 Jazzy packages once
#    /opt/ros/jazzy/setup.bash is sourced).
# 2. Depth Anything V2 code -> third_party/Depth-Anything-V2 (the tested commit).
# 3. Depth Anything V2 Base weights (372 MB) -> models/depth_anything_v2_vitb.pth.
# Grounding DINO and SAM download themselves from Hugging Face on the first run (~1.5 GB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-$HOME/envs/cvenev}"
DA_COMMIT=a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
WEIGHTS_URL=https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating venv $VENV"
    python3 -m venv "$VENV"
fi
echo "== installing Python packages"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

if [ ! -d "$HERE/third_party/Depth-Anything-V2/.git" ]; then
    echo "== cloning Depth Anything V2"
    mkdir -p "$HERE/third_party"
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$HERE/third_party/Depth-Anything-V2"
fi
git -C "$HERE/third_party/Depth-Anything-V2" checkout -q "$DA_COMMIT"

if [ ! -f "$HERE/models/depth_anything_v2_vitb.pth" ]; then
    echo "== downloading Depth Anything V2 Base weights (372 MB)"
    mkdir -p "$HERE/models"
    curl -L --fail -o "$HERE/models/depth_anything_v2_vitb.pth.part" "$WEIGHTS_URL"
    mv "$HERE/models/depth_anything_v2_vitb.pth.part" "$HERE/models/depth_anything_v2_vitb.pth"
fi
echo "== done. Run the node with: $VENV/bin/python $HERE/vla_ros2_node.py <object> --ros-args ..."
