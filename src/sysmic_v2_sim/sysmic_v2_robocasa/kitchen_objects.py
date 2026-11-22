#!/usr/bin/env python3
"""A RoboCasa kitchen with plates, cups, mugs, bowls and cutlery on a chosen surface.

    python kitchen_objects.py                     # viewer, tableware on the counter in front of SYSMIC
    python kitchen_objects.py --layout 5 --style 3

From your own code:

    import sysmic_robocasa as S
    import kitchen_objects as K                    # registers "SysmicTableware"
    env = S.make_env("SysmicTableware", layout_ids=1, style_ids=1)
    env = S.make_env("SysmicTableware", tableware=[("mug", "mug"), ("plate", "plate")])
    env.reset()
    K.place_at(env, "mug", [2.0, -0.5, 1.0])      # optional: exact world position

TABLEWARE entries are (name, category) or (name, category, surface):

  category  a RoboCasa object category: plate, bowl, mug, cup, coffee_cup, fork, spoon,
            knife, pan, teapot, tray, cutting_board, ladle, spatula, ...
            (folders in robocasa/models/assets/objects/objaverse; one model of that
            category is picked at random each reset).
  surface   where it goes. Default: the counter region in front of the robot (REGION).
            Or a FixtureType (ISLAND, DINING_COUNTER, SHELF, SINK, CABINET, ...): then it
            is put anywhere on/in that fixture, if the layout has one.

RoboCasa finds the surface height itself, so objects always rest on top of it, and it
makes sure they do not overlap. For an exact position at any height, use place_at().

AprilTags for vla/ (repo root) vla_ros2_node.py (tags=False to leave them out)
  36h11, 35 mm black square, matching vla-pipeline-v2.py: ids 0-3 flat on the counter
  (0, 1 along the front edge, 2, 3 along the back), 4 and 5 on blocks 5 cm and 12 cm
  high (its CHECK_TAG_IDS). A tag whose spot is taken or hidden by an appliance moves
  to the nearest free, visible spot, or is left out (printed). env.tag_poses has the
  true poses and env.counter_height the counter top, for checking the pipeline.

    ros2 launch sysmic_v2_sim sim.launch.py task:=SysmicTableware lift:=0.45
    python3 vla_ros2_node.py --ros-args -p image_topic:=/head_camera/color/image_raw \\
        -p camera_info_topic:=/head_camera/color/camera_info -p image_is_rectified:=true \\
        -p object_name:=mug -p use_sim_time:=true

  The tags are placed to be visible from VIEW_POINT (about the head with the lift at
  0.45 m). The camera has to look down at the counter to see them: the head camera's
  pitch (rpy) and resolution are set in sensors.yaml.
"""
import os
import sys

import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TABLEWARE = [
    ("plate", "plate"),
    ("bowl", "bowl"),
    ("mug", "mug"),
    ("cup", "cup"),
    ("fork", "fork"),
    ("spoon", "spoon"),
]

# Area on the counter shared by the default objects: size (along, depth) in metres,
# pos of that area on the counter, each -1 ... 1 ("ref" = lined up with the robot,
# y = -1 is the front edge), and offset (along, depth) in metres. The 10 cm offset keeps
# a strip along the front edge free for the near AprilTags.
REGION = dict(size=(0.8, 0.4), pos=("ref", -1.0), offset=(0.0, 0.10))

# AprilTag 36h11 markers for the VLA perception pipeline (vla/ (repo root) vla-pipeline-v2.py).
# Must match TAG_SIZE_METERS / TABLE_TAG_IDS / CHECK_TAG_IDS there.
TAG_SIZE = 0.035          # side of the black square, m
TAG_MARGIN_CELLS = 2      # white border around it, in tag cells (1 cell = TAG_SIZE / 8)
# "near" = along the counter's front edge (in front of the tableware, never hidden by it),
# "far" = along its back edge. Left offsets are from the robot's centre line, so the tags
# stay in a forward-looking camera's view. If a spot is taken or hidden, the tag moves to
# the nearest free, visible spot.
TABLE_TAGS = {            # id -> (row, to the robot's left in m)
    0: ("near", 0.18),
    1: ("near", -0.18),
    2: ("far", 0.24),
    3: ("far", -0.24),
}
CHECK_TAGS = {            # id -> (height of the tag surface above the counter in m, row, left in m)
    4: (0.05, "far", 0.08),
    5: (0.12, "far", -0.08),
}
BLOCK_SIZE = 0.06         # footprint of the blocks that carry the check tags, m
BLOCK_INSET = 0.07        # blocks stand this much closer to the robot than the far row, m
# Tags must be visible from here: (ahead of the robot's base, above the counter), m.
# Roughly the head camera with the lift raised.
VIEW_POINT = (0.15, 0.45)


def _tag_bits(tag_id):
    """8 x 8 cells of an AprilTag 36h11 (1 = white), row 0 at the top of the printed tag."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(d, tag_id, 8, borderBits=1) // 255


def _tag_body(tag_id, block_height=0.0):
    """MJCF body (mocap, so it never moves) of one tag, optionally on a block.

    The tag is built from boxes, not a texture: a white square plus one black box per
    run of black cells. Its local z is up, local y is the top edge of the printed tag.
    """
    import xml.etree.ElementTree as ET

    cell = TAG_SIZE / 8
    white = (8 + 2 * TAG_MARGIN_CELLS) * cell
    body = ET.Element("body", name=f"apriltag{tag_id}", mocap="true", pos="0 0 -5")
    vis = dict(type="box", group="1", contype="0", conaffinity="0")
    z0 = 0.0
    if block_height > 0:
        h = block_height - 0.0012          # tag surface ends up at exactly block_height
        hs = f"{BLOCK_SIZE / 2} {BLOCK_SIZE / 2} {h / 2}"
        ET.SubElement(body, "geom", name=f"apriltag{tag_id}_block", type="box", size=hs,
                      pos=f"0 0 {h / 2}", group="0", rgba="0.75 0.62 0.45 1", density="500")
        ET.SubElement(body, "geom", name=f"apriltag{tag_id}_block_vis", size=hs, pos=f"0 0 {h / 2}",
                      rgba="0.75 0.62 0.45 1", **vis)
        z0 = h
    ET.SubElement(body, "geom", name=f"apriltag{tag_id}_white", size=f"{white / 2} {white / 2} 0.0005",
                  pos=f"0 0 {z0 + 0.0005}", rgba="1 1 1 1", **vis)
    bits = _tag_bits(tag_id)
    for r in range(8):
        c = 0
        while c < 8:
            if bits[r, c]:
                c += 1
                continue
            start = c
            while c < 8 and not bits[r, c]:
                c += 1
            x = ((start + c) / 2 - 4) * cell          # image column -> +x (right)
            y = (4 - (r + 0.5)) * cell                # image row 0 -> +y (top)
            ET.SubElement(body, "geom", name=f"apriltag{tag_id}_r{r}c{start}",
                          size=f"{(c - start) * cell / 2} {cell / 2} 0.0004",
                          pos=f"{x} {y} {z0 + 0.0008}", rgba="0 0 0 1", **vis)
    return body


class SysmicTableware(Kitchen):
    """Kitchen with TABLEWARE placed on it; the robot starts facing the counter.

    kitchen_drop  metres to lower the whole kitchen (counters, cabinets, appliances, walls,
           tableware, tags) after each reset; the floor and the robot stay. Makes the
           counter lower relative to the robot's cameras, so a head camera at full lift
           sees the whole counter. 0 = RoboCasa's normal heights.
    tags   True: AprilTags 0-3 flat on the counter around the tableware, 4 and 5 on blocks
           (see TABLE_TAGS / CHECK_TAGS). After each reset env.tag_poses holds where they are.
    """

    def __init__(self, tableware=None, tags=True, kitchen_drop=0.0, *args, **kwargs):
        self.kitchen_drop = float(kitchen_drop)
        self.tableware = [tuple(t) for t in (tableware or TABLEWARE)]
        self.use_tags = tags
        self.tag_poses = {}
        self.counter_height = None
        super().__init__(*args, **kwargs)

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        if not self.use_tags or self.model.worldbody.find("./body[@name='apriltag0']") is not None:
            return  # RoboCasa retries _load_model recursively; add the tags only once
        for tag_id in TABLE_TAGS:
            self.model.worldbody.append(_tag_body(tag_id))
        for tag_id, (height, _, _) in CHECK_TAGS.items():
            self.model.worldbody.append(_tag_body(tag_id, height))

    def _reset_internal(self):
        super()._reset_internal()
        if self.kitchen_drop:
            self._lower_kitchen(self.kitchen_drop)
            self.back_off_robot()
        if self.use_tags:
            self._place_tags()

    def back_off_robot(self, max_back=0.3, step=0.01, margin=0.01):
        """Move the robot straight back until no kitchen geom presses into it.

        With the kitchen lowered, the cabinet fronts come down into the space under the
        counter where the robot's base platform sits. Returns how far it moved (m).
        """
        import mujoco

        m, d = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        prefixes = (robot.robot_model.naming_prefix, robot.robot_model.base.naming_prefix,
                    *(g.naming_prefix for g in robot.gripper.values()))
        robot_roots = {b for b in range(m.nbody)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefixes)}
        ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{robot.robot_model.naming_prefix}wheel_ground")
        adr = m.jnt_qposadr[m.joint(f"{robot.robot_model.base.naming_prefix}joint_mobile_forward").id]

        def touching():
            mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                if c.dist >= 0.0 or ground in (c.geom1, c.geom2):
                    continue
                a = m.geom_bodyid[c.geom1] in robot_roots
                b = m.geom_bodyid[c.geom2] in robot_roots
                if a != b:
                    return True
            return False

        moved = 0.0
        while touching() and moved < max_back:
            d.qpos[adr] -= step
            moved += step
        if moved:  # a little extra, so nothing (e.g. a dishwasher door) is left resting on it
            d.qpos[adr] -= margin
            moved += margin
            mujoco.mj_forward(m, d)
        if moved:
            print(f"[SysmicTableware] robot moved {moved * 100:.0f} cm back to clear the lowered kitchen")
        return moved

    def _lower_kitchen(self, drop):
        """Move every top-level kitchen body down by `drop` m; floor, robot and tags stay."""
        import mujoco

        m, d = self.sim.model._model, self.sim.data._data
        # Open doors and pulled-out racks that cleared the robot at normal height can hit its
        # base once lowered (layout 1's open dishwasher shoved it 36 cm back): close them.
        for fixture in getattr(self, "fixtures", {}).values():
            joints = getattr(fixture, "_joint_names", None) or {}
            try:
                if "rack" in joints:
                    fixture.set_joint_state(env=self, min=0.0, max=0.0, joint_names=[joints["rack"]])
                if getattr(fixture, "door_joint_names", None):
                    fixture.close_door(env=self)
            except Exception as exc:  # a fixture without usable door joints: leave it
                print(f"[SysmicTableware] could not close {getattr(fixture, 'name', fixture)}: {exc}")
        mujoco.mj_forward(m, d)
        # Drop-down doors (dishwasher, oven: horizontal hinge) fall open under their own
        # weight. Lowered, they swing into the robot's base and push it away: lock them shut.
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and "door" in name.lower()
                    and abs(d.xaxis[j][2]) < 0.3):
                closed = float(d.qpos[m.jnt_qposadr[j]])
                m.jnt_range[j] = (closed, closed + 1e-4)
                m.jnt_limited[j] = 1
        if getattr(self, "_lowered_model", None) is not m:  # a hard reset builds a new model
            robot = self.robots[0]
            keep = tuple({robot.robot_model.naming_prefix, robot.robot_model.base.naming_prefix,
                          *(g.naming_prefix for g in robot.gripper.values()), "floor", "apriltag"})
            objects = {int(m.body_rootid[self.obj_body_id[n]]) for n in self.objects}
            for b in range(1, m.nbody):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if m.body_parentid[b] == 0 and b not in objects and not name.startswith(keep):
                    m.body_pos[b][2] -= drop
            self._lowered_model = m
        for n, obj in self.objects.items():  # free bodies: RoboCasa just placed them
            adr = m.jnt_qposadr[m.joint(obj.joints[0]).id]
            d.qpos[adr + 2] -= drop
        mujoco.mj_forward(m, d)

    def _place_tags(self):
        """Put the tags on free, visible counter in front of the robot, facing it."""
        import mujoco

        m, d = self.sim.model._model, self.sim.data._data
        name2id = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        robot = self.robots[0].robot_model
        support = name2id(f"{robot.base.naming_prefix}support")
        tag_ids = list(TABLE_TAGS) + list(CHECK_TAGS)
        tag_bodies = {i: name2id(f"apriltag{i}") for i in tag_ids}
        ignore_roots = {m.body_rootid[support]} | {m.body_rootid[b] for b in tag_bodies.values()}
        object_roots = {m.body_rootid[self.obj_body_id[n]] for n in self.objects}

        # put the tags out of the way while measuring the counter
        for b in tag_bodies.values():
            d.mocap_pos[m.body_mocapid[b]] = [0, 0, -5]
        mujoco.mj_forward(m, d)

        objs = np.array([d.xpos[self.obj_body_id[n]] for n in self.objects])
        top = objs[:, 2].min() if len(objs) else 1.0
        heading = d.xmat[support].reshape(3, 3)[:, 0]
        fwd = heading[:2] / np.linalg.norm(heading[:2])
        left = np.array([-fwd[1], fwd[0]])
        base = d.xpos[support][:2]
        geomid = np.zeros(1, np.int32)

        def surface(xy):
            """(height, hit an object?) of what lies under xy, ignoring robot and tags."""
            start = np.array([xy[0], xy[1], top + 0.30])
            down = np.array([0.0, 0.0, -1.0])
            while True:
                dist = mujoco.mj_ray(m, d, start, down, None, 1, -1, geomid)
                if dist < 0:
                    return None, False
                hit = start + dist * down
                root = m.body_rootid[m.geom_bodyid[geomid[0]]]
                if root in object_roots:
                    return hit[2], True
                if root not in ignore_roots:
                    return hit[2], False
                start = hit + 0.001 * down

        # walk straight ahead of the robot to find the counter top and its front / back edge
        steps = np.arange(0.0, 1.5, 0.01)
        line = [surface(base + a * fwd) for a in steps]
        on_counter = [z is not None and (o or abs(z - top) < 0.03) for z, o in line]
        if not any(on_counter):
            print("[SysmicTableware] no counter in front of the robot; AprilTags left out")
            return
        first = on_counter.index(True)
        last = first
        while last + 1 < len(steps) and on_counter[last + 1]:
            last += 1
        bare = [line[i][0] for i in range(first, last + 1) if not line[i][1]]
        h0 = float(np.median(bare)) if bare else top - 0.01
        front, back = steps[first], steps[last]
        self.counter_height = h0

        eye = np.array([*(base + VIEW_POINT[0] * fwd), h0 + VIEW_POINT[1] + self.kitchen_drop])

        def visible(p):
            """Nothing but the robot or the tags between point p and the eye."""
            start, end = np.array(p, float), eye
            while True:
                vec = end - start
                length = np.linalg.norm(vec)
                dist = mujoco.mj_ray(m, d, start, vec / length, None, 1, -1, geomid)
                if dist < 0 or dist >= length:
                    return True
                if m.body_rootid[m.geom_bodyid[geomid[0]]] not in ignore_roots:
                    return False
                start = start + (dist + 0.001) * vec / length

        def free(xy, half, height=0.0):
            pts = [xy + sx * half * fwd + sy * half * left for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
            for p in pts:
                z, on_obj = surface(p)
                if z is None or on_obj or abs(z - h0) > 0.01:
                    return False
            tag_half = white / 2
            corners = [xy + sx * tag_half * fwd + sy * tag_half * left for sx in (-1, 1) for sy in (-1, 1)]
            return all(visible([c[0], c[1], h0 + height + 0.02]) for c in corners + [xy])

        white = (8 + 2 * TAG_MARGIN_CELLS) * TAG_SIZE / 8
        search = [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a)) for r in (0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21)
                                 for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        yaw = np.arctan2(-left[1], -left[0])  # tag +x = robot's right, tag +y = away from robot
        quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        placed, self.tag_poses = [], {}
        nominal = {i: (0.0, row, l) for i, (row, l) in TABLE_TAGS.items()}
        nominal.update(CHECK_TAGS)
        for tag_id, (height, row, side) in nominal.items():
            half = (BLOCK_SIZE if height > 0 else white) / 2 + 0.01
            along = front + half + 0.01 if row == "near" else back - half - 0.01
            if height > 0:
                along -= BLOCK_INSET  # blocks stand a bit forward so their tags stay in view
            for da, dl in search:
                xy = base + (along + da) * fwd + (side + dl) * left
                if all(np.linalg.norm(xy - q) > 2 * half + 0.01 for q in placed) and free(xy, half, height):
                    break
            else:
                print(f"[SysmicTableware] no free counter space for AprilTag {tag_id}; left out")
                continue
            placed.append(xy)
            pos = np.array([xy[0], xy[1], h0 + 0.0002])
            d.mocap_pos[m.body_mocapid[tag_bodies[tag_id]]] = pos
            d.mocap_quat[m.body_mocapid[tag_bodies[tag_id]]] = quat
            self.tag_poses[tag_id] = dict(pos=pos + [0, 0, height + (0.0012 if height == 0 else 0)],
                                          yaw=float(yaw), height_above_counter=height)
        mujoco.mj_forward(m, d)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.cab))
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):
        cfgs, region_owner = [], None
        for entry in self.tableware:
            name, category = entry[:2]
            placement_extra = {}
            surface = entry[2] if len(entry) > 2 else None
            if surface is None:
                placement = dict(fixture=self.counter, sample_region_kwargs=dict(ref=self.cab), **REGION)
                if region_owner is None:
                    region_owner = name
                    placement_extra = dict(init_robot_here=True)  # robot starts facing the tableware
                else:  # all default objects share one area, so they end up together
                    placement["reuse_region_from"] = region_owner
            else:
                placement = dict(fixture=self.register_fixture_ref(f"{name}_surface", dict(id=surface)),
                                 size=(0.4, 0.4))
            cfgs.append(dict(name=name, obj_groups=category, placement=placement, **placement_extra))
        return cfgs


def place_at(env, name, xyz, yaw=0.0):
    """Move object `name` to world position xyz (metres), heading yaw (rad).

    Call after env.reset(). The object drops from there, so set z 1-2 cm above the
    surface you want it on (objects spawned by RoboCasa print their z on the counter).
    """
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # MuJoCo w, x, y, z
    env.sim.data.set_joint_qpos(env.objects[name].joints[0], np.concatenate([xyz, quat]))
    env.sim.forward()


def object_positions(env):
    return {n: env.sim.data.get_body_xpos(o.root_body).copy() for n, o in env.objects.items()}


def main():
    import argparse

    import sysmic_robocasa as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    args = ap.parse_args()

    env = S.make_env("SysmicTableware", layout_ids=args.layout, style_ids=args.style,
                     has_renderer=True, renderer="mjviewer", render_camera=None)
    env.reset()
    for name, pos in object_positions(env).items():
        print(f"{name:8s} x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}")
    try:
        while not S.viewer_closed(env):
            env.step(S.joint_action(env, {}))
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
