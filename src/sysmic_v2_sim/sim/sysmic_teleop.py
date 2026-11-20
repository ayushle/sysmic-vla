#!/usr/bin/env python3
"""Control panel for SYSMIC 2.0 in a RoboCasa kitchen (workspace version).

    conda activate robocasa
    python sysmic_teleop.py                      # kitchen layout 1, style 1
    python sysmic_teleop.py --layout 5 --style 12

Same panel as ../sysmic_v2_robocasa/control_panel.py, but with the robot model
rebuilt from the URDF first (sysmic_robosuite does that on import).
"""
import sysmic_robosuite  # noqa: F401  (builds the model from the URDF, puts the folder on sys.path)
from control_panel import main

if __name__ == "__main__":
    main()
