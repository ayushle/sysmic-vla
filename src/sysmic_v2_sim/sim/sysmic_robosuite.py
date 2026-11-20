#!/usr/bin/env python3
"""SYSMIC 2.0 (urdf/sysmic.urdf) as a registered robosuite / RoboCasa robot.

Builds the robosuite model files from the URDF (into ~/.cache/sysmic_v2_sim) and
then uses the standalone module ../sysmic_v2_robocasa/sysmic_robocasa.py for the
robot classes and helpers. To share the robot, use that folder (see its README).

    import sysmic_robosuite as S                  # registers the robot
    env = S.make_env("Kitchen", layout_ids=1, style_ids=1)

or run the self-test / viewer:

    python sysmic_robosuite.py                    # self-test, then a kitchen viewer
    python sysmic_robosuite.py --no-viewer        # self-test only

What is modelled differently from the URDF
-------------------------------------------------
* The base is moved by three planar joints (forward, sideways, yaw), exactly like
  RoboCasa's own PandaOmron; send 0 for the sideways action to stay diff-drive.
  The drive wheels, casters and suspension are real passive joints, so they roll
  and swivel, but they do not propel the robot. They roll on an invisible ground
  plane of their own at floor height, not on the kitchen floor itself.
* The base collision is two boxes (platform + tower) instead of the mesh.
* Each gripper finger collides as its own convex shape, so objects fit between the jaws.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sysmic_paths import DESCRIPTION, ROBOCASA_DIR, ROBOSUITE_DIR as ASSET_DIR, \
    ROBOT_XML as SRC_XML

# The robot classes and helpers live in the standalone folder sysmic_v2_robocasa/
# (sysmic_robocasa.py). Point it at the files this script builds from the URDF.
os.environ["SYSMIC_ROBOCASA_ASSETS"] = ASSET_DIR
if ROBOCASA_DIR not in sys.path:
    sys.path.insert(0, ROBOCASA_DIR)
from sysmic_robocasa import (ARM_JOINTS, ARM_READY, GRIPPER_CLOSED, GRIPPER_OPEN,  # noqa: E402
                             LIFT_RANGE, URDF_ARM_JOINTS, WHEEL_GROUND)

ROBOT_XML = os.path.join(ASSET_DIR, "robot.xml")
BASE_XML = os.path.join(ASSET_DIR, "base.xml")
GRIPPER_XML = {s: os.path.join(ASSET_DIR, f"{s}_gripper.xml") for s in ("left", "right")}
CONTROLLER_JSON = os.path.join(ASSET_DIR, "controller.json")

# URDF link names. Inside the robosuite model the arm bodies are called
# {side}_link0..5 and the joints {side}_<ARM_JOINTS>.
URDF_ARM_LINKS = ["base_link", "shoulder_pan_link", "upper_arm_link", "forearm_link",
                  "wrist_link", "wrist_roll_link"]


def urdf_link(side, i):
    """URDF link name of arm link i (0 = mount ... 5 = wrist roll, 6 = jaw)."""
    return f"arm_{side}_jaw_link" if i == 6 else f"arm_{side}_{URDF_ARM_LINKS[i]}"

SERVO_TORQUE = 2.9  # N m, STS3215
ARM_ARMATURE = "0.01"  # reflected gearbox inertia (estimate)
ARM_DAMPING = "0.2"
WHEEL_LAYER = "4"  # collision layer shared only by the wheels and their private ground


# --------------------------------------------------------------------------- #
# Asset generation
# --------------------------------------------------------------------------- #
def _fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _mesh_verts(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
    a, n = model.mesh_vertadr[i], model.mesh_vertnum[i]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.mesh_quat[i])
    return model.mesh_vert[a:a + n] @ R.reshape(3, 3).T + model.mesh_pos[i]


def _box_from_verts(verts, R):
    """Oriented bounding box of verts in the frame whose columns are R."""
    local = verts @ R
    lo, hi = local.min(0), local.max(0)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return R @ ((lo + hi) / 2), (hi - lo) / 2, quat


def _mesh_asset(src_root, names):
    asset = ET.Element("asset")
    for mesh in src_root.find("asset").findall("mesh"):
        if mesh.get("name") in names:
            # robosuite resolves mesh files relative to the XML's own folder
            path = os.path.join(DESCRIPTION, mesh.get("file"))
            el = ET.SubElement(asset, "mesh", name=mesh.get("name"),
                               file=os.path.relpath(path, ASSET_DIR))
            if mesh.get("inertia"):  # decals are thin surfaces: shell inertia
                el.set("inertia", mesh.get("inertia"))
    return asset


def _visual(parent, name, mesh, rgba, pos=None):
    g = ET.SubElement(parent, "geom", name=name, type="mesh", mesh=mesh, rgba=rgba,
                      contype="0", conaffinity="0", group="1")
    if pos is not None:
        g.set("pos", _fmt(pos))
    return g


def _collision_mesh(parent, name, mesh):
    return ET.SubElement(parent, "geom", name=name, type="mesh", mesh=mesh,
                         group="0", rgba="0.5 0.5 0.5 1")


def _collision_box(parent, name, pos, size, quat=None, friction=None):
    g = ET.SubElement(parent, "geom", name=name, type="box", pos=_fmt(pos),
                      size=_fmt(size), group="0", rgba="0.5 0.5 0.5 1")
    if quat is not None:
        g.set("quat", _fmt(quat))
    if friction:
        g.set("friction", friction)
        g.set("condim", "4")
    return g


def _collision_hull(asset, parent, name, verts, friction=None):
    """Collision geom shaped like the convex hull of verts (stored inline in the XML)."""
    from scipy.spatial import ConvexHull

    hull = verts[ConvexHull(verts).vertices]
    ET.SubElement(asset, "mesh", name=f"{name}_mesh", vertex=_fmt(hull.flatten()))
    g = ET.SubElement(parent, "geom", name=name, type="mesh", mesh=f"{name}_mesh",
                      group="0", rgba="0.5 0.5 0.5 1")
    if friction:
        g.set("friction", friction)
        g.set("condim", "4")
    return g


def _write(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")


def build_assets(force=False):
    """Generate robosuite-style XML files from the converted SYSMIC model."""
    import sysmic_to_mjcf

    if sysmic_to_mjcf.is_stale():  # URDF changed, or first run on this machine
        sysmic_to_mjcf.convert(verbose=False)
    targets = [ROBOT_XML, BASE_XML, CONTROLLER_JSON, *GRIPPER_XML.values()]
    if not force and all(os.path.exists(t) and os.path.getmtime(t) >= os.path.getmtime(SRC_XML)
                         and os.path.getmtime(t) >= os.path.getmtime(__file__) for t in targets):
        return
    os.makedirs(ASSET_DIR, exist_ok=True)

    src_model = mujoco.MjModel.from_xml_path(SRC_XML)
    src = ET.parse(SRC_XML).getroot()
    bodies = {b.get("name"): b for b in src.iter("body")}
    rgba = {g.get("mesh"): g.get("rgba") for g in src.iter("geom") if g.get("mesh")}

    def mesh(body):
        """Mesh asset name of a URDF link's collision (the full original mesh)."""
        geoms = [g for g in bodies[body].findall("geom") if g.get("mesh")]
        return next((g.get("mesh") for g in geoms if g.get("contype") != "0"),
                    geoms[0].get("mesh"))

    def visuals(body):
        """[(mesh, rgba)] of a URDF link's visuals (more than one if it has a decal)."""
        return [(g.get("mesh"), g.get("rgba")) for g in bodies[body].findall("geom")
                if g.get("mesh") and g.get("contype") == "0"]

    def vec(el, key, default="0 0 0"):
        return np.array([float(x) for x in (el.get(key) or default).split()])

    def inertial(dst, name):
        dst.append(ET.fromstring(ET.tostring(bodies[name].find("inertial"))))

    # ---------------- robot.xml : base body, lift, two arms ---------------- #
    robot = ET.Element("mujoco", model="sysmic")
    ET.SubElement(robot, "compiler", angle="radian")
    used = {mesh("base_link"), mesh("lift_carriage_link")} | {
        mesh(urdf_link(s, i)) for s in ("left", "right") for i in range(6)} | {
        m for b in ("base_link", "lift_carriage_link") for m, _ in visuals(b)}
    robot.append(_mesh_asset(src, used))
    actuator = ET.SubElement(robot, "actuator")
    contact = ET.SubElement(robot, "contact")
    wb = ET.SubElement(robot, "worldbody")
    root = ET.SubElement(wb, "body", name="base", pos="0 0 0")
    # Invisible ground that only the wheels and casters touch. RoboCasa rejects any
    # robot start position where a robot part touches something that is not the
    # robot - the kitchen floor included - so the wheels roll on this plane instead.
    # It is fixed in the world and belongs to the robot model.
    ET.SubElement(wb, "geom", name=WHEEL_GROUND, type="plane", pos="0 0 0", size="0 0 1",
                  group="0", rgba="0 0 0 0", friction="1 0.005 0.0001",
                  contype=WHEEL_LAYER, conaffinity=WHEEL_LAYER)
    base = ET.SubElement(root, "body", name="base_link", pos="0 0 0")
    inertial(base, "base_link")
    ET.SubElement(base, "site", name="center", pos="0 0 0", size="0.01", rgba="0 0 0 0", group="2")
    for k, (m, c) in enumerate(visuals("base_link")):
        _visual(base, f"base_link_vis{k or ''}", m, c)

    # Wheels, casters and the platform collision box live in the mobile base model
    # (base.xml) so their passive joints stay out of robosuite's joint bookkeeping.
    bv = _mesh_verts(src_model, mesh("base_link"))
    plat = bv[bv[:, 2] < 0.6]
    tower = bv[bv[:, 2] > 0.6]
    p_lo, p_hi = plat.min(0), plat.max(0)
    p_lo[2] = 0.02  # reach down to just above the floor
    t_lo, t_hi = tower.min(0), tower.max(0)
    t_lo[2] = p_hi[2]
    _collision_box(base, "base_tower_col", (t_lo + t_hi) / 2, (t_hi - t_lo) / 2)

    # lift (robosuite "torso")
    lift_src = bodies["lift_carriage_link"]
    torso = ET.SubElement(base, "body", name="torso_lift_link", pos=lift_src.get("pos"))
    inertial(torso, "lift_carriage_link")
    ET.SubElement(torso, "joint", name="torso_lift_joint", type="slide", axis="0 0 1",
                  limited="true", range=_fmt(LIFT_RANGE), damping="50", armature="1")
    for k, (m, c) in enumerate(visuals("lift_carriage_link")):  # body + "SYSMIC" lettering
        _visual(torso, f"torso_vis{k or ''}", m, c)
    _collision_mesh(torso, "torso_col", mesh("lift_carriage_link"))
    ET.SubElement(actuator, "motor", name="torq_torso_lift", joint="torso_lift_joint",
                  ctrllimited="true", ctrlrange="-1000 1000")
    ET.SubElement(contact, "exclude", body1="base_link", body2="torso_lift_link")

    grip_geo = {}
    for side in ("right", "left"):  # robosuite expects the right arm first
        parent = torso
        for i in range(6):
            name, src_name = f"{side}_link{i}", urdf_link(side, i)
            b = bodies[src_name]
            body = ET.SubElement(parent, "body", name=name, pos=b.get("pos") or "0 0 0")
            inertial(body, src_name)
            if i == 0:  # reference frame robosuite's controllers use for this arm
                ET.SubElement(body, "site", name=f"{side}_center", pos="0 0 0", size="0.01",
                              rgba="0 0 0 0", group="2")
            if i > 0:
                j = b.find("joint")
                ET.SubElement(body, "joint", name=f"{side}_{ARM_JOINTS[i - 1]}", type="hinge",
                              axis=j.get("axis"), limited="true", range=j.get("range"),
                              damping=ARM_DAMPING, armature=ARM_ARMATURE)
                ET.SubElement(actuator, "motor", name=f"torq_{side}_j{i}",
                              joint=f"{side}_{ARM_JOINTS[i - 1]}", ctrllimited="true",
                              ctrlrange=f"{-SERVO_TORQUE} {SERVO_TORQUE}")
            _visual(body, f"{name}_vis", mesh(src_name), rgba[mesh(src_name)])
            if i < 5:
                _collision_mesh(body, f"{name}_col", mesh(src_name))
            parent = body
        ET.SubElement(contact, "exclude", body1="torso_lift_link", body2=f"{side}_link1")
        ET.SubElement(contact, "exclude", body1=f"{side}_link1", body2=f"{side}_link3")
        # the mount bracket's convex hull is much bigger than the bracket and blocks
        # the upper arm once the arm pans (e.g. pan 0.6, shoulder 1.6)
        ET.SubElement(contact, "exclude", body1=f"{side}_link0", body2=f"{side}_link2")

        # wrist link: servo body as a box; the fixed finger goes into the gripper model
        jaw_src = bodies[urdf_link(side, 6)]
        jaw_pos = vec(jaw_src, "pos")
        axis = vec(jaw_src.find("joint"), "axis")
        axis /= np.linalg.norm(axis)
        x = np.array([1.0, 0, 0])
        d = np.cross(x, axis)  # direction the open jaw points; also the closing normal
        R = np.stack([x, d, np.cross(x, d)], axis=1)
        v5 = _mesh_verts(src_model, mesh(urdf_link(side, 5)))
        split = jaw_pos[0] + 0.02
        pos, size, quat = _box_from_verts(v5[v5[:, 0] < split], R)
        _collision_box(parent, f"{side}_link5_col", pos, size, quat)
        ET.SubElement(parent, "body", name=f"{side}_hand", pos="0 0 0")
        grip_geo[side] = dict(jaw_pos=jaw_pos, axis=axis, d=d, R=R, split=split, v5=v5,
                              jaw_range=jaw_src.find("joint").get("range"),
                              vj=_mesh_verts(src_model, mesh(urdf_link(side, 6))),
                              jaw_mesh=mesh(urdf_link(side, 6)))
    _write(robot, ROBOT_XML)

    # ---------------- grippers ---------------- #
    for side, g in grip_geo.items():
        R, d, x = g["R"], g["d"], np.array([1.0, 0, 0])
        gr = ET.Element("mujoco", model=f"sysmic_{side}_gripper")
        g_asset = _mesh_asset(src, {g["jaw_mesh"]})
        gr.append(g_asset)
        act = ET.SubElement(gr, "actuator")
        ET.SubElement(act, "position", name="gripper_jaw", joint="jaw_joint", kp="20",
                      ctrllimited="true", ctrlrange=f"{GRIPPER_OPEN} {GRIPPER_CLOSED}",
                      forcelimited="true", forcerange=f"{-SERVO_TORQUE} {SERVO_TORQUE}")
        gwb = ET.SubElement(gr, "worldbody")
        groot = ET.SubElement(gwb, "body", name="gripper_base", pos="0 0 0")
        ET.SubElement(groot, "site", name="ft_frame", pos="0 0 0", size="0.01", type="sphere",
                      rgba="1 0 0 0", group="1")
        ET.SubElement(groot, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")

        # fixed finger: convex hull of the blade part of the wrist link
        blade = g["v5"][g["v5"][:, 0] >= g["split"]]
        f_pos, f_size, f_quat = _box_from_verts(blade, R)
        _collision_hull(g_asset, groot, "fixed_finger_collision", blade,
                        friction="2 0.05 0.0001")

        # grasp point: above the middle of the fixed finger, towards the moving jaw
        pad = f_pos + d * (f_size[1] + 0.012) + x * 0.25 * f_size[0]
        eef_R = np.stack([np.cross(d, x), d, x], axis=1)  # z = approach, y = closing
        eef_q = np.zeros(4)
        mujoco.mju_mat2Quat(eef_q, eef_R.flatten())
        eef = ET.SubElement(groot, "body", name="eef", pos=_fmt(pad), quat=_fmt(eef_q))
        ET.SubElement(eef, "site", name="grip_site", pos="0 0 0", size="0.008", type="sphere",
                      rgba="1 0 0 0.5", group="1")
        for n, p, q, c in (("ee_x", "0.05 0 0", "0.707105 0 0.707108 0", "1 0 0 0"),
                           ("ee_y", "0 0.05 0", "0.707105 0.707108 0 0", "0 1 0 0"),
                           ("ee_z", "0 0 0.05", "1 0 0 0", "0 0 1 0")):
            ET.SubElement(eef, "site", name=n, pos=p, quat=q, size="0.003 0.05",
                          type="cylinder", rgba=c, group="1")
        ET.SubElement(eef, "site", name="grip_site_cylinder", pos="0 0 0", size="0.003 10",
                      type="cylinder", rgba="0 1 0 0", group="1")

        # moving jaw
        jaw = ET.SubElement(groot, "body", name="jaw", pos=_fmt(g["jaw_pos"]))
        jaw.append(ET.fromstring(ET.tostring(bodies[urdf_link(side, 6)].find("inertial"))))
        ET.SubElement(jaw, "joint", name="jaw_joint", type="hinge", axis=_fmt(g["axis"]),
                      limited="true", range=g["jaw_range"], damping=ARM_DAMPING,
                      armature=ARM_ARMATURE)
        _visual(jaw, "jaw_vis", g["jaw_mesh"], rgba[g["jaw_mesh"]])
        # jaw blade (the part beyond the hinge block), as a convex hull
        vj = g["vj"]
        _collision_hull(g_asset, jaw, "jaw_collision", vj[vj @ d > 0.02],
                        friction="2 0.05 0.0001")

        sensor = ET.SubElement(gr, "sensor")
        ET.SubElement(sensor, "force", name="force_ee", site="ft_frame")
        ET.SubElement(sensor, "torque", name="torque_ee", site="ft_frame")
        _write(gr, GRIPPER_XML[side])

    # ---------------- mobile base ---------------- #
    # Three driven planar joints (like RoboCasa's own robots) move the base. The drive
    # wheels and casters hang underneath on PASSIVE joints and are turned by floor
    # friction, so they roll, swivel and ride their springs as the base moves.
    mb = ET.Element("mujoco", model="sysmic_mobile_base")
    wheel_meshes = {mesh(n) for n in bodies if ("caster" in n or "wheel" in n) and "susp" not in n}
    mb.append(_mesh_asset(src, wheel_meshes))
    act = ET.SubElement(mb, "actuator")
    mcontact = ET.SubElement(mb, "contact")
    #            joint      type     axis     top speed  kv    armature friction
    planar = (("forward", "slide", "1 0 0", 0.5, 4000, 0, 10),
              ("side", "slide", "0 1 0", 0.5, 4000, 0, 10),
              # yaw: the robot's turning inertia is small, so the motor gain needs
              # extra rotor inertia (armature) to stay stable at a 2 ms time step
              ("yaw", "hinge", "0 0 1", 1.0, 1000, 5, 10))
    for jn, _, _, lim, kv, _, _ in planar:
        ET.SubElement(act, "velocity", name=f"actuator_mobile_{jn}", joint=f"joint_mobile_{jn}",
                      kv=str(kv), ctrllimited="true", ctrlrange=f"{-lim} {lim}",
                      forcelimited="true", forcerange="-600 600")
    support = ET.SubElement(ET.SubElement(mb, "worldbody"), "body", name="support", pos="0 0 0")
    ET.SubElement(support, "site", name="center", type="sphere", pos="0 0 0", size="0.01",
                  group="1", rgba="0 0 0 0")
    for jn, tp, ax, _, _, arm, fl in planar:
        ET.SubElement(support, "joint", name=f"joint_mobile_{jn}", pos="0 0 0", axis=ax, type=tp,
                      limited="false", damping="0", armature=str(arm), frictionloss=str(fl))
    _collision_box(support, "platform_col", (p_lo + p_hi) / 2, (p_hi - p_lo) / 2)

    def wheel_parts(dst, src_body, tag):
        """Copy a wheel's visual mesh and its contact cylinder."""
        for i, g in enumerate(src_body.findall("geom")):
            if g.get("mesh") and g.get("contype") == "0":  # the visual copy
                _visual(dst, f"{tag}_vis{i}", g.get("mesh"), g.get("rgba"))
            elif g.get("type") == "cylinder":
                ET.SubElement(dst, "geom", name=f"{tag}_col", type="cylinder", size=g.get("size"),
                              pos=g.get("pos"), quat=g.get("quat"), group="0",
                              rgba="0.5 0.5 0.5 1", friction="1 0.005 0.0001",
                              contype=WHEEL_LAYER, conaffinity=WHEEL_LAYER)

    def passive_body(parent, name, src_name, pos=None):
        b = ET.SubElement(parent, "body", name=name, pos=pos or bodies[src_name].get("pos"))
        b.append(ET.fromstring(ET.tostring(bodies[src_name].find("inertial"))))
        return b

    for c in ("FL", "FR", "RL", "RR"):
        swivel = passive_body(support, f"caster_{c}", f"caster_{c}_swivel_link")
        # small sprung travel keeps the caster on the floor (about 50 N) even if the
        # floor is a millimetre or two higher or lower than expected
        ET.SubElement(swivel, "joint", name=f"passive_caster_{c}_susp", type="slide", axis="0 0 1",
                      limited="true", range="-0.004 0.004", stiffness="20000",
                      springref="-0.0025", damping="200", armature="1")
        ET.SubElement(swivel, "joint", name=f"passive_caster_{c}_swivel", type="hinge",
                      axis="0 0 1", damping="0.005", armature="0.0005")
        wheel_parts(swivel, bodies[f"caster_{c}_swivel_link"], f"caster_{c}")
        wheel = passive_body(swivel, f"caster_{c}_wheel", f"caster_{c}_wheel_link")
        ET.SubElement(wheel, "joint", name=f"passive_caster_{c}_wheel", type="hinge",
                      axis="0 1 0", damping="0.005", armature="0.0005")
        wheel_parts(wheel, bodies[f"caster_{c}_wheel_link"], f"caster_{c}_wheel")
        ET.SubElement(mcontact, "exclude", body1="support", body2=f"caster_{c}_wheel")

    for side in ("left", "right"):
        susp = passive_body(support, f"wheel_{side}_susp", f"wheel_{side}_susp_link")
        ET.SubElement(susp, "joint", name=f"passive_wheel_{side}_susp", type="slide", axis="0 0 1",
                      limited="true", range="0 0.012", stiffness="34000", springref="0",
                      damping="400", armature="0.5")
        wheel = passive_body(susp, f"wheel_{side}", f"wheel_{side}_link", pos="0 0 0")
        ET.SubElement(wheel, "joint", name=f"passive_wheel_{side}_spin", type="hinge",
                      axis="0 1 0", damping="0.1", armature="0.001")
        wheel_parts(wheel, bodies[f"wheel_{side}_link"], f"wheel_{side}")
        ET.SubElement(mcontact, "exclude", body1="support", body2=f"wheel_{side}")
    _write(mb, BASE_XML)

    # ---------------- controller config ---------------- #
    def arm_cfg():
        return {"type": "JOINT_POSITION", "input_max": 1, "input_min": -1,
                "input_type": "absolute", "output_max": 0.5, "output_min": -0.5,
                "kp": 1500, "damping_ratio": 1, "impedance_mode": "fixed",
                "kp_limits": [0, 5000], "damping_ratio_limits": [0, 10],
                "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"}}

    cfg = {"type": "BASIC",
           "body_parts": {
               "arms": {"right": arm_cfg(), "left": arm_cfg()},
               "torso": {"type": "JOINT_POSITION", "input_max": 1, "input_min": -1,
                         "input_type": "absolute", "output_max": 0.5, "output_min": -0.5,
                         "kp": 400, "damping_ratio": 1, "impedance_mode": "fixed",
                         "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
                         "interpolation": None, "ramp_ratio": 0.2},
               "base": {"type": "JOINT_VELOCITY", "interpolation": "null"}}}
    with open(CONTROLLER_JSON, "w") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------- #
# robosuite classes and helpers: shared with the standalone folder
# --------------------------------------------------------------------------- #
build_assets()
from sysmic_robocasa import *  # noqa: E402,F401,F403  (Sysmic robot, make_env, make_action, ...)
from sysmic_robocasa import self_test  # noqa: E402,F401


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Kitchen")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    extra = {}
    if args.task not in __import__("robosuite").ALL_ENVIRONMENTS or args.task == "Kitchen":
        extra = dict(layout_ids=args.layout, style_ids=args.style)
    # 1) Self-test without a window. (robosuite closes and reopens the viewer on every
    #    env.reset(), so testing inside the viewer made the window appear twice.)
    env = make_env(args.task, **extra)
    self_test(env)
    print(f"SYSMIC registered. Action size: {env.action_dim}")
    env.close()

    # 2) One viewer window. render_camera=None gives a free camera you can move with
    #    the mouse; RoboCasa's default camera sits right behind SYSMIC's tower.
    if not args.no_viewer:
        env = make_env(args.task, has_renderer=True, renderer="mjviewer",
                       render_camera=None, **extra)
        env.reset()
        print("Viewer open. The robot holds its pose; close the window to quit.")
        try:
            while not viewer_closed(env):
                env.step(make_action(env, right_arm=np.array(ARM_READY),
                                     left_arm=np.array(ARM_READY), lift=0.2))
                env.render()
        except KeyboardInterrupt:
            pass
        env.close()
        print("Stopped.")
