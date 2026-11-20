#!/usr/bin/env python3
"""Move every SYSMIC 2.0 joint in a RoboCasa kitchen.

    python demo.py                        # kitchen viewer, robot goes through a few poses
    python demo.py --layout 5 --style 12  # another kitchen
    python demo.py --no-viewer            # same motions without a window, prints joints

Needs robosuite and RoboCasa (with kitchen assets) installed; nothing else.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysmic_robocasa as S  # noqa: E402  (registers the robot)

# (seconds, joint targets by URDF name, base (forward, turn))
READY = {"lift_joint": 0.2, **{f"arm_{s}_{j}": v for s in ("left", "right")
                               for j, v in zip(S.URDF_ARM_JOINTS, S.ARM_READY)}}
SEQUENCE = [
    (2.0, READY, (0.0, 0.0)),
    (2.5, {}, (-0.5, 0.0)),  # RoboCasa starts the robot at a counter: back away first
    (2.0, {}, (0.0, 0.8)),   # turn (arms tucked in)
    (3.0, {"lift_joint": 0.40}, (0.0, 0.0)),
    # With the CAD masses (3.5 kg per arm) the 2.9 N m servos only just hold the arm;
    # poses that stretch it further out sag. This one tracks well.
    (3.0, {"arm_left_pan_joint": -0.3, "arm_left_shoulder_joint": 1.4,
           "arm_left_elbow_joint": -1.0, "arm_left_wrist_flex_joint": 0.2,
           "arm_right_pan_joint": 0.3, "arm_right_shoulder_joint": 1.4,
           "arm_right_elbow_joint": -1.0, "arm_right_wrist_flex_joint": 0.2}, (0.0, 0.0)),
    (2.0, {"arm_left_wrist_roll_joint": -1.0, "arm_right_wrist_roll_joint": 1.0}, (0.0, 0.0)),
    (2.0, {"arm_left_gripper_joint": 1.7, "arm_right_gripper_joint": 1.7}, (0.0, 0.0)),
    (2.0, {"arm_left_gripper_joint": -0.17, "arm_right_gripper_joint": -0.17}, (0.0, 0.0)),
    (3.0, READY, (0.0, 0.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Kitchen")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    kitchen = dict(layout_ids=args.layout, style_ids=args.style) if args.task == "Kitchen" else {}
    view = not args.no_viewer
    env = S.make_env(args.task, has_renderer=view,
                     **(dict(renderer="mjviewer", render_camera=None) if view else {}), **kitchen)
    env.reset()
    print("Joints (URDF names):", ", ".join(S.JOINT_NAMES))

    targets = {}
    try:
        for seconds, step_targets, base in SEQUENCE:
            targets.update(step_targets)
            for _ in range(int(seconds * env.control_freq)):
                env.step(S.joint_action(env, targets, base=base))
                if view:
                    env.render()
                    if S.viewer_closed(env):
                        return
            now = S.get_joints(env)
            print("  " + ("  ".join(f"{k}={now[k]:+.2f}" for k in step_targets) or "(base move)"))
        if view:
            print("Done. The robot holds its pose; close the viewer to quit.")
            while not S.viewer_closed(env):
                env.step(S.joint_action(env, targets))
                env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
