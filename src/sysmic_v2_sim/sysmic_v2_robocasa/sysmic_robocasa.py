#!/usr/bin/env python3
"""SYSMIC 2.0 for robosuite / RoboCasa: self-contained, needs only this folder.

    import sys; sys.path.insert(0, "/path/to/sysmic_v2_robocasa")
    import sysmic_robocasa as S                 # registers the robot "Sysmic"

    env = S.make_env("Kitchen", layout_ids=1, style_ids=1)   # any robosuite/RoboCasa env
    env.reset()
    print(S.get_joints(env))                    # {"lift_joint": 0.2, "arm_left_pan_joint": 0.0, ...}
    env.step(S.joint_action(env, {"lift_joint": 0.35, "arm_right_shoulder_joint": 1.4,
                                  "arm_right_gripper_joint": 1.2}, base=(0.0, 0.0)))

Joint names are the URDF names (sysmic.urdf). Arms/lift take absolute targets in rad / m;
joints left out of the dict hold their current position. Grippers take the jaw angle
in rad (-0.17 open ... 1.75 closed). base = (forward, turn), each -1..1 of top speed
(0.5 m/s, 1.0 rad/s).

Model files are in assets/ next to this file. Set SYSMIC_ROBOCASA_ASSETS to use
another copy (the workspace's sysmic_robosuite.py does this for freshly built files).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysmic_sensors  # noqa: E402  (cameras, lidars, IMUs from sensors.yaml)
from sysmic_sensors import Sensors  # noqa: E402,F401

__all__ = [
    "Sysmic", "SysmicMobileBase", "SysmicLeftGripper", "SysmicRightGripper",
    "make_env", "controller_config", "make_action", "joint_action", "get_joints",
    "gripper_action", "viewer_closed", "self_test",
    "JOINT_NAMES", "JOINT_LIMITS", "ARM_JOINTS", "URDF_ARM_JOINTS", "ARM_READY",
    "LIFT_RANGE", "GRIPPER_OPEN", "GRIPPER_CLOSED", "ASSET_DIR",
    "Sensors", "sysmic_sensors",
]

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.environ.get("SYSMIC_ROBOCASA_ASSETS") or os.path.join(HERE, "assets")
ROBOT_XML = os.path.join(ASSET_DIR, "robot.xml")
BASE_XML = os.path.join(ASSET_DIR, "base.xml")
GRIPPER_XML = {s: os.path.join(ASSET_DIR, f"{s}_gripper.xml") for s in ("left", "right")}
CONTROLLER_JSON = os.path.join(ASSET_DIR, "controller.json")

# Internal joint names in the robosuite model: robot0_{side}_<ARM_JOINTS>,
# robot0_torso_lift_joint, gripper0_{side}_jaw_joint. URDF names map onto them.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF_ARM_JOINTS = ["pan_joint", "shoulder_joint", "elbow_joint", "wrist_flex_joint",
                   "wrist_roll_joint"]
LIFT_RANGE = (0.0, 0.447677)  # m
GRIPPER_OPEN, GRIPPER_CLOSED = -0.174533, 1.74533  # jaw angle (rad)
# Arm pose at the start of an episode: [pan, shoulder, elbow, wrist_flex, wrist_roll]
ARM_READY = [0.0, 1.0, -1.2, 0.4, 0.0]
WHEEL_GROUND = "wheel_ground"

# URDF joint names that can be commanded, in a fixed order
JOINT_NAMES = (["lift_joint"]
               + [f"arm_{s}_{j}" for s in ("left", "right") for j in URDF_ARM_JOINTS]
               + [f"arm_{s}_gripper_joint" for s in ("left", "right")])
JOINT_LIMITS = {"lift_joint": LIFT_RANGE}
for _s in ("left", "right"):
    JOINT_LIMITS.update({
        f"arm_{_s}_pan_joint": (-1.919862, 1.919862),
        f"arm_{_s}_shoulder_joint": (0.0, 3.490659),
        f"arm_{_s}_elbow_joint": (-3.385939, 0.0),
        f"arm_{_s}_wrist_flex_joint": (-1.658063, 1.658063),
        f"arm_{_s}_wrist_roll_joint": (-2.740167, 2.844887),
        f"arm_{_s}_gripper_joint": (GRIPPER_OPEN, GRIPPER_CLOSED)})

try:
    import robosuite  # noqa: E402,F401
except ModuleNotFoundError:
    import sys as _sys

    raise SystemExit(
        f"robosuite is not installed for this Python ({_sys.executable}).\n"
        "Use the Python that has robosuite and RoboCasa, e.g.:\n"
        "    conda activate robocasa && python demo.py\n"
        "or  ~/miniconda3/envs/robocasa/bin/python demo.py") from None

from robosuite.models.bases import BASE_MAPPING  # noqa: E402
from robosuite.models.bases.mobile_base_model import MobileBaseModel  # noqa: E402
from robosuite.models.grippers import register_gripper  # noqa: E402
from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel  # noqa: E402
from robosuite.robots import register_robot_class  # noqa: E402


class SysmicMobileBase(MobileBaseModel):
    """Planar (forward / sideways / yaw) base under SYSMIC 2.0."""

    def __init__(self, idn=0):
        super().__init__(BASE_XML, idn=idn)
        # robosuite treats every joint of a base model as a driven base joint.
        # The wheel / caster joints are passive, so keep them out of that list.
        self._joints = [j for j in self._joints if "passive" not in j]

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.36


BASE_MAPPING["SysmicMobileBase"] = SysmicMobileBase


class _SysmicGripper(GripperModel):
    side = None

    def __init__(self, idn=0):
        super().__init__(GRIPPER_XML[self.side], idn=idn)
        sysmic_sensors.mount(self, f"gripper_{self.side}")  # gripper cameras

    def format_action(self, action):
        """Jaw target: -1 fully open, +1 fully closed, values in between are in between.

        The jaw's position motor then moves it there at servo speed.
        """
        assert len(action) == self.dof
        self.current_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.current_action

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        return np.array([0.3])

    @property
    def _important_geoms(self):
        return {"left_finger": ["jaw_collision"], "right_finger": ["fixed_finger_collision"],
                "left_fingerpad": ["jaw_collision"], "right_fingerpad": ["fixed_finger_collision"]}


@register_gripper
class SysmicLeftGripper(_SysmicGripper):
    side = "left"


@register_gripper
class SysmicRightGripper(_SysmicGripper):
    side = "right"


@register_robot_class("WheeledRobot")
class Sysmic(ManipulatorModel):
    """SYSMIC 2.0: mobile base, lift and two 5-joint arms with single-jaw grippers."""

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(ROBOT_XML, idn=idn)
        sysmic_sensors.mount(self, "robot")  # head/tray cameras, IMUs

    def set_base_xpos(self, pos):
        """Place the robot; the wheels' private ground follows the base height."""
        super().set_base_xpos(pos)
        for geom in self.worldbody.findall("geom"):
            if geom.get("name", "").endswith(WHEEL_GROUND):
                geom.set("pos", f"0 0 {float(pos[2]) - float(self.bottom_offset[2])}")

    @property
    def default_base(self):
        return "SysmicMobileBase"

    @property
    def default_gripper(self):
        return {"right": "SysmicRightGripper", "left": "SysmicLeftGripper"}

    @property
    def default_controller_config(self):
        return {"right": "joint_position", "left": "joint_position", "torso": "joint_position"}

    @property
    def init_qpos(self):
        return np.array([0.2] + ARM_READY + ARM_READY)  # [lift, right arm, left arm]

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {"bins": (-0.6, -0.1, 0), "empty": (-0.6, 0, 0),
                "table": lambda table_length: (-0.45 - table_length / 2, 0, 0)}

    @property
    def top_offset(self):
        return np.array((0, 0, 1.2))

    @property
    def _horizontal_radius(self):
        return 0.36

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_hand", "left": "left_hand"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def controller_config():
    from robosuite.controllers import load_composite_controller_config

    return load_composite_controller_config(controller=CONTROLLER_JSON)


def make_env(env_name="Kitchen", sensors=None, **kwargs):
    """robosuite.make(...) with SYSMIC and its controller config filled in.

    sensors   path of a sensors.yaml (default: $SYSMIC_SENSORS or the one next to this
              file). It decides which sensors are mounted and the control rate.
    """
    import robosuite

    cfg = sysmic_sensors.set_active(sensors)

    if env_name not in robosuite.ALL_ENVIRONMENTS:
        import robocasa  # noqa: F401  (registers the kitchen environments)
    opts = dict(env_name=env_name, robots="Sysmic", controller_configs=controller_config(),
                has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                ignore_done=True, control_freq=cfg["sim"]["control_freq"])
    opts.update(kwargs)
    return robosuite.make(**opts)


def gripper_action(angle):
    """Jaw angle in rad (URDF convention, about -0.17 open ... 1.75 closed) -> action."""
    return float(np.clip(2 * (angle - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN) - 1, -1, 1))


def viewer_closed(env):
    """True once the user has closed the MuJoCo viewer window."""
    handle = getattr(getattr(env, "viewer", None), "viewer", None)
    return handle is not None and not handle.is_running()


def _base_action(env, forward, turn):
    """Base action that drives along the robot's own heading.

    robosuite's base controller rotates the (forward, sideways) command by minus the
    angle the robot has turned since reset before sending it to the slide joints,
    which lie along the frame the robot started in. Left alone, "forward" stops
    pointing where the robot faces as soon as it has turned (the stock PandaOmron
    does the same). Rotating the command the other way first cancels that out.
    """
    if forward == 0.0:
        return [0.0, 0.0, turn]
    import robosuite.utils.transform_utils as T

    robot, sim = env.robots[0], env.sim
    yaw_joint = f"{robot.robot_model.base.naming_prefix}joint_mobile_yaw"
    yaw = float(sim.data.qpos[sim.model.get_joint_qpos_addr(yaw_joint)])
    ctrl = robot.composite_controller.part_controllers["base"]
    theta = 0.0
    if getattr(ctrl, "init_ori", None) is not None:
        theta = T.mat2euler(ctrl.get_base_pose()[1])[2] - T.mat2euler(ctrl.init_ori)[2]
    angle = theta + yaw
    return [forward * np.cos(angle), forward * np.sin(angle), turn]


def make_action(env, base=(0.0, 0.0), lift=None, right_arm=None, left_arm=None,
                right_gripper=-1.0, left_gripper=-1.0):
    """Build one action vector.

    base          (forward, turn) in [-1, 1]; sideways is always 0 (diff-drive)
    lift          lift height in metres (None = hold current height)
    right_arm     5 joint angles in rad  (None = hold current angles)
    left_arm      5 joint angles in rad  (None = hold current angles)
    *_gripper     -1 fully open ... +1 fully closed (use gripper_action() for an angle)
    """
    robot = env.robots[0]
    sim = env.sim

    def current(joints):
        return np.array([sim.data.qpos[sim.model.get_joint_qpos_addr(j)] for j in joints])

    prefix = robot.robot_model.naming_prefix
    parts = {
        "right": current([f"{prefix}right_{j}" for j in ARM_JOINTS]) if right_arm is None else right_arm,
        "left": current([f"{prefix}left_{j}" for j in ARM_JOINTS]) if left_arm is None else left_arm,
        "torso": current([f"{prefix}torso_lift_joint"]) if lift is None else [lift],
        "base": _base_action(env, base[0], base[1]),
        "right_gripper": [right_gripper],
        "left_gripper": [left_gripper],
    }
    return robot.create_action_vector({k: np.asarray(v, dtype=float) for k, v in parts.items()})


def self_test(env, verbose=True):
    """Hold still, move the arms and lift, close a gripper, then drive."""
    env.reset()
    steps = lambda seconds: int(round(seconds * env.control_freq))  # noqa: E731
    robot = env.robots[0]
    prefix = robot.robot_model.naming_prefix
    q = lambda j: float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(j)])  # noqa: E731
    arm_q = lambda s: np.array([q(f"{prefix}{s}_{j}") for j in ARM_JOINTS])  # noqa: E731
    base_xy = lambda: env.sim.data.body_xpos[  # noqa: E731
        env.sim.model.body_name2id(f"{prefix}base_link")][:2].copy()
    results = {}

    hold = dict(right_arm=np.array(ARM_READY), left_arm=np.array(ARM_READY), lift=0.2)
    trace = []
    for i in range(steps(3.0)):
        env.step(make_action(env, **hold))
        if i >= steps(1.5):
            trace.append(np.concatenate([arm_q("right"), arm_q("left")]))
    results["jitter at rest (rad)"] = float(np.ptp(np.array(trace), axis=0).max())
    results["hold error (rad)"] = float(np.abs(arm_q("right") - ARM_READY).max())

    target = np.array([0.3, 1.4, -1.0, 0.2, 0.5])
    for _ in range(steps(4.0)):
        env.step(make_action(env, right_arm=target, left_arm=target * [-1, 1, 1, 1, -1], lift=0.35))
    results["arm tracking error (rad)"] = float(np.abs(arm_q("right") - target).max())
    results["lift error (m)"] = abs(q(f"{prefix}torso_lift_joint") - 0.35)

    jaw = f"{robot.gripper['right'].naming_prefix}jaw_joint"
    for _ in range(steps(2.0)):
        env.step(make_action(env, right_arm=target, left_arm=target * [-1, 1, 1, 1, -1],
                             lift=0.35, right_gripper=1.0))
    results["closed jaw angle (rad)"] = q(jaw)

    start = base_xy()
    for _ in range(steps(2.0)):
        env.step(make_action(env, base=(-1.0, 0.0)))
    results["reversed in 2 s (m)"] = float(np.linalg.norm(base_xy() - start))
    # after a quarter turn, "forward" must still move the robot the way it faces
    def heading():
        mat = env.sim.data.body_xmat[env.sim.model.body_name2id(f"{prefix}base_link")].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    for _ in range(steps(1.6)):
        env.step(make_action(env, base=(0.0, 1.0)))
    for _ in range(steps(0.5)):
        env.step(make_action(env))
    start, face = base_xy(), heading()
    for _ in range(steps(1.0)):
        env.step(make_action(env, base=(-0.6, 0.0)))
    move = start - base_xy()  # reversing, so flip it to compare with the heading
    off = (np.arctan2(move[1], move[0]) - face + np.pi) % (2 * np.pi) - np.pi
    results["heading error after turn (rad)"] = abs(float(off))
    if verbose:
        for k, v in results.items():
            print(f"  {k:32s}: {v:.4f}")
    return results


# --------------------------------------------------------------------------- #
# Joints by URDF name
# --------------------------------------------------------------------------- #
def _sim_joint(env, name):
    """URDF joint name (or internal name like 'left_elbow_flex') -> MuJoCo joint name."""
    robot = env.robots[0]
    prefix = robot.robot_model.naming_prefix
    if name == "lift_joint":
        return f"{prefix}torso_lift_joint"
    for side in ("left", "right"):
        if name in (f"arm_{side}_gripper_joint", f"{side}_gripper"):
            return f"{robot.gripper[side].naming_prefix}jaw_joint"
        for u, j in zip(URDF_ARM_JOINTS, ARM_JOINTS):
            if name in (f"arm_{side}_{u}", f"{side}_{j}"):
                return f"{prefix}{side}_{j}"
    raise KeyError(f"unknown SYSMIC joint '{name}'. Known: {', '.join(JOINT_NAMES)}")


def get_joints(env):
    """Current positions of all commandable joints, by URDF name (m / rad)."""
    sim = env.sim
    return {n: float(sim.data.qpos[sim.model.get_joint_qpos_addr(_sim_joint(env, n))])
            for n in JOINT_NAMES}


def joint_action(env, targets=None, base=(0.0, 0.0)):
    """Action vector from joint targets by URDF name.

    targets   {joint: position}; m for lift_joint, rad otherwise. Joints that are
              left out hold their current position. Values are clipped to the limits.
    base      (forward, turn), each -1..1 of top speed.
    """
    current = get_joints(env)
    want = dict(current)
    for name, value in (targets or {}).items():
        sim_name = _sim_joint(env, name)  # also checks the name
        urdf = next(n for n in JOINT_NAMES if _sim_joint(env, n) == sim_name)
        lo, hi = JOINT_LIMITS[urdf]
        want[urdf] = float(np.clip(value, lo, hi))
    arm = lambda s: [want[f"arm_{s}_{u}"] for u in URDF_ARM_JOINTS]  # noqa: E731
    return make_action(env, base=base, lift=want["lift_joint"],
                       right_arm=arm("right"), left_arm=arm("left"),
                       right_gripper=gripper_action(want["arm_right_gripper_joint"]),
                       left_gripper=gripper_action(want["arm_left_gripper_joint"]))
