"""SYSMIC 2.0 sensors: cameras, lidars and IMUs, configured by sensors.yaml.

    import sysmic_robocasa as S
    env = S.make_env("Kitchen", layout_ids=1, style_ids=1)     # mounts the enabled sensors
    sensors = S.Sensors(env)
    env.reset()
    env.step(S.joint_action(env, {}))
    scan = sensors.read("lidar_top")            # dict: ranges (m, inf = no hit), angles, ...
    imu = sensors.read("base_imu")              # dict: accel, gyro, orientation
    rgb = sensors.read("head_camera_color")     # dict: image (H, W, 3) uint8, intrinsics
    depth = sensors.read("head_camera_depth")   # dict: image (H, W) uint16 millimetres

How each sensor is simulated
  camera  a MuJoCo <camera> on the link, rendered offscreen (visual geoms only)
  lidar   one ray per beam from the mount point, cast at the lidar's rate with
          mujoco.mj_multiRay against visual geoms. The mount link itself (the
          lidar's housing) is ignored; the rest of the robot does block beams.
  imu     MuJoCo accelerometer + gyro + framequat on a site, plus Gaussian noise

Which sensors exist and their parameters come from sensors.yaml next to this file,
or from the file named by $SYSMIC_SENSORS, or from make_env(sensors="path").
Positions are in the URDF link's frame. The sensor's own frame has x forward
(viewing direction), y left, z up; rpy rotates it (degrees, positive pitch = down).

`show: true` draws the sensor's body as a box of `size` [depth, width, height]
(visual only, no collision, no mass), in MuJoCo and in the robot description for
RViz. A camera's box sits behind its lens: the front face is BODY_GAP behind the
sensor origin, so it never appears in the camera's own image.

This module only needs numpy and PyYAML at import time (MuJoCo is imported when
used), so the ROS launch file can read the same config and TF frames from it.
"""
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "sensors.yaml")

# URDF link -> (model part, body name inside that part, before robosuite's prefix)
_LINKS = {"base_link": ("robot", "base_link"), "lift_carriage_link": ("robot", "torso_lift_link")}
for _s in ("left", "right"):
    _LINKS[f"arm_{_s}_wrist_roll_link"] = (f"gripper_{_s}", "gripper_base")
    _LINKS[f"arm_{_s}_jaw_link"] = (f"gripper_{_s}", "jaw")
    for _i, _l in enumerate(["base_link", "shoulder_pan_link", "upper_arm_link",
                             "forearm_link", "wrist_link"]):
        _LINKS[f"arm_{_s}_{_l}"] = ("robot", f"{_s}_link{_i}")

# MuJoCo cameras look along their -z with +y up. Columns: camera axes in the sensor
# frame (x forward, y left, z up).
_CAM_IN_SENSOR = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
OPTICAL_RPY = (-90.0, 0.0, -90.0)  # ROS optical frame (z forward, x right, y down)
BODY_GAP = 0.001  # m between a camera box's front face and the lens point
DEFAULT_BODY_RGBA = "0.12 0.12 0.13 1"

_ACTIVE = {"path": None, "config": None}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load(path=None):
    """Read and check a sensors.yaml. Returns the parsed dict."""
    path = path or os.environ.get("SYSMIC_SENSORS") or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("sim", {}).setdefault("control_freq", 30)
    for group in ("cameras", "lidars", "imus"):
        cfg[group] = cfg.get(group) or {}
    for group, items in (("cameras", cfg["cameras"]), ("lidars", cfg["lidars"]),
                         ("imus", cfg["imus"])):
        for name, s in items.items():
            if s.get("link") not in _LINKS:
                raise ValueError(f"{path}: {group}.{name}.link '{s.get('link')}' is not one of "
                                 f"{', '.join(sorted(_LINKS))}")
            s.setdefault("enabled", True)
            s.setdefault("pos", [0, 0, 0])
            s.setdefault("rpy", [0, 0, 0])
    cfg["_path"] = os.path.abspath(path)
    return cfg


def set_active(path=None):
    """Choose the config used by the next env that is created."""
    _ACTIVE["config"] = load(path)
    _ACTIVE["path"] = _ACTIVE["config"]["_path"]
    return _ACTIVE["config"]


def active():
    if _ACTIVE["config"] is None:
        set_active()
    return _ACTIVE["config"]


def _rot(rpy_deg):
    """Rotation matrix from roll, pitch, yaw in degrees (fixed axes x, y, z)."""
    r, p, y = np.radians(rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat(R):
    """(w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def streams(cfg=None):
    """Every enabled data stream, flattened: {stream name: description}.

    A D435i gives up to three streams (<name>_color, <name>_depth, <name>_imu);
    every other sensor gives one stream named like the sensor.
    """
    cfg = cfg or active()
    out = {}
    for name, c in cfg["cameras"].items():
        if not c["enabled"]:
            continue
        base = dict(sensor=name, link=c["link"], rate=float(c.get("rate", 30)))
        R = _rot(c["rpy"])
        pos = np.array(c["pos"], float)
        if c.get("model") == "realsense_d435i":
            for kind in ("color", "depth"):
                sub = c.get(kind) or {}
                if not sub.get("enabled", True):
                    continue
                off = np.array(sub.get("offset", [0, 0, 0]), float)
                out[f"{name}_{kind}"] = dict(
                    base, type="camera", kind=kind, pos=pos + R @ off, R=R,
                    width=int(sub.get("width", 640)), height=int(sub.get("height", 480)),
                    hfov=float(sub.get("hfov", 69)), min_range=float(sub.get("min_range", 0.0)),
                    max_range=float(sub.get("max_range", 1e9)),
                    topic=sub.get("topic", f"/{name}/{kind}/image_raw"),
                    frame=f"{name}_{kind}_optical_frame")
            imu = c.get("imu") or {}
            if imu.get("enabled", False):
                out[f"{name}_imu"] = dict(
                    base, type="imu", pos=pos, R=R, rate=float(imu.get("rate", 30)),
                    accel_noise=float(imu.get("accel_noise", 0.0)),
                    gyro_noise=float(imu.get("gyro_noise", 0.0)),
                    topic=imu.get("topic", f"/{name}/imu"), frame=f"{name}_imu_optical_frame")
        else:
            out[name] = dict(base, type="camera", kind="color", pos=pos, R=R,
                             width=int(c.get("width", 640)), height=int(c.get("height", 480)),
                             hfov=float(c.get("hfov", 75)), min_range=0.0, max_range=1e9,
                             topic=c.get("topic", f"/{name}/image_raw"),
                             frame=f"{name}_optical_frame")
    for name, l in cfg["lidars"].items():
        if l["enabled"]:
            out[name] = dict(sensor=name, type="lidar", link=l["link"],
                             pos=np.array(l["pos"], float), R=_rot(l["rpy"]),
                             rate=float(l.get("rate", 10)),
                             min_range=float(l.get("min_range", 0.1)),
                             max_range=float(l.get("max_range", 10.0)),
                             angle_min=float(l.get("angle_min", -180)),
                             angle_max=float(l.get("angle_max", 180)),
                             resolution=float(l.get("resolution", 1.0)),
                             noise=float(l.get("noise", 0.0)),
                             topic=l.get("topic", f"/{name}/scan"), frame=f"{name}_link")
    for name, i in cfg["imus"].items():
        if i["enabled"]:
            out[name] = dict(sensor=name, type="imu", link=i["link"],
                             pos=np.array(i["pos"], float), R=_rot(i["rpy"]),
                             rate=float(i.get("rate", 30)),
                             accel_noise=float(i.get("accel_noise", 0.0)),
                             gyro_noise=float(i.get("gyro_noise", 0.0)),
                             topic=i.get("topic", f"/{name}"), frame=f"{name}_link")
    return out


def bodies(cfg=None):
    """Boxes to draw for enabled sensors with `show: true`:
    {sensor: dict(link, centre_in_link, R, centre_in_sensor, half, rgba)}."""
    cfg = cfg or active()
    out = {}
    for group in ("cameras", "lidars", "imus"):
        for name, c in cfg[group].items():
            if not (c["enabled"] and c.get("show", False)):
                continue
            size = np.array(c.get("size", [0.02, 0.02, 0.02]), float)
            local = np.array([-(size[0] / 2 + BODY_GAP), 0.0, 0.0]) if group == "cameras" \
                else np.zeros(3)
            R = _rot(c["rpy"])
            out[name] = dict(link=c["link"], R=R, centre_in_sensor=local,
                             centre_in_link=np.array(c["pos"], float) + R @ local,
                             half=size / 2, rgba=c.get("body_color", DEFAULT_BODY_RGBA))
    return out


def tf_frames(cfg=None):
    """Fixed frames for robot_state_publisher: [(parent, child, xyz, rpy_rad)].

    Every enabled sensor gets <sensor>_link on its URDF link; cameras also get their
    ROS optical frame(s), named as in the published messages.
    """
    cfg = cfg or active()
    frames, done = [], set()
    opt = tuple(math.radians(a) for a in OPTICAL_RPY)
    for name, s in streams(cfg).items():
        sensor = s["sensor"]
        group = cfg["cameras"].get(sensor) or cfg["lidars"].get(sensor) or cfg["imus"].get(sensor)
        link = f"{sensor}_link"
        if link not in done:
            frames.append((s["link"], link, tuple(group["pos"]),
                           tuple(math.radians(a) for a in group["rpy"])))
            done.add(link)
        if s["frame"] != link:  # camera streams: optical frame at the stream's own origin
            R_inv = _rot(group["rpy"]).T
            local = R_inv @ (s["pos"] - np.array(group["pos"], float))
            frames.append((link, s["frame"], tuple(local), opt))
    return frames


# --------------------------------------------------------------------------- #
# Mounting on the robosuite model (called from the robot / gripper classes)
# --------------------------------------------------------------------------- #
def _model_name(stream):
    return f"sysmic_{stream}"


def mount(model, part, cfg=None):
    """Add the cameras / IMU sites of `part` ("robot", "gripper_left", "gripper_right")
    to a robosuite model, after its names got their prefix."""
    import xml.etree.ElementTree as ET

    prefix = model.naming_prefix
    elements = {b.get("name"): b for b in model.worldbody.iter("body")}
    for sensor, b in bodies(cfg).items():  # visible sensor housings
        link_part, body = _LINKS[b["link"]]
        if link_part == part:
            ET.SubElement(elements[prefix + body], "geom", name=prefix + f"sysmic_{sensor}_body",
                          type="box", pos=_fmt(b["centre_in_link"]), quat=_fmt(_quat(b["R"])),
                          size=_fmt(b["half"]), rgba=b["rgba"], contype="0", conaffinity="0",
                          group="1", density="0")
    for stream, s in streams(cfg).items():
        link_part, body = _LINKS[s["link"]]
        if link_part != part:
            continue
        el = elements.get(prefix + body)
        if el is None:
            raise ValueError(f"sensor {stream}: body {prefix + body} not in the {part} model")
        name = prefix + _model_name(stream)
        if s["type"] == "camera":
            R = s["R"] @ _CAM_IN_SENSOR
            fovy = math.degrees(2 * math.atan(math.tan(math.radians(s["hfov"]) / 2)
                                              * s["height"] / s["width"]))
            ET.SubElement(el, "camera", name=name, mode="fixed", pos=_fmt(s["pos"]),
                          quat=_fmt(_quat(R)), fovy=f"{fovy:.4f}")
        elif s["type"] == "imu":
            ET.SubElement(el, "site", name=name, pos=_fmt(s["pos"]), quat=_fmt(_quat(s["R"])),
                          size="0.005", rgba="0 0 0 0", group="3")
            for kind in ("accelerometer", "gyro", "framequat"):
                attrs = dict(name=f"{name}_{kind}")
                if kind == "framequat":
                    attrs.update(objtype="site", objname=name)
                else:
                    attrs["site"] = name
                ET.SubElement(model.sensor, kind, **attrs)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
class Sensors:
    """Reads the enabled sensors of a SYSMIC env (created with sysmic_robocasa.make_env)."""

    def __init__(self, env, cfg=None, seed=None):
        import mujoco

        self._mj = mujoco
        self.env = env
        self.cfg = cfg or active()
        self.streams = streams(self.cfg)
        self.rng = np.random.default_rng(seed)
        self._renderers = {}
        self._next = {}
        self._resolve()

    def _resolve(self):
        """Look up MuJoCo ids. Call again after env.reset() rebuilt the model."""
        mj, robot = self._mj, self.env.robots[0]
        m = self.env.sim.model._model
        self._model = m
        prefixes = {"robot": robot.robot_model.naming_prefix,
                    "gripper_left": robot.gripper["left"].naming_prefix,
                    "gripper_right": robot.gripper["right"].naming_prefix}
        for stream, s in self.streams.items():
            s["name"] = stream
            part, body = _LINKS[s["link"]]
            s["body_id"] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, prefixes[part] + body)
            name = prefixes[part] + _model_name(stream)
            if s["type"] == "camera":
                s["cam_id"] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, name)
                if s["cam_id"] < 0:
                    raise RuntimeError(f"camera {name} is not in the model: create the env with "
                                       "sysmic_robocasa.make_env after choosing the config")
            elif s["type"] == "imu":
                s["adr"] = {}
                for kind in ("accelerometer", "gyro", "framequat"):
                    sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, f"{name}_{kind}")
                    if sid < 0:
                        raise RuntimeError(f"IMU sensor {name}_{kind} is not in the model")
                    s["adr"][kind] = int(m.sensor_adr[sid])
            elif s["type"] == "lidar":
                n = int(round((s["angle_max"] - s["angle_min"]) / s["resolution"]))
                if s["angle_max"] - s["angle_min"] < 360 - 1e-6:
                    n += 1  # both ends included for a partial sweep
                s["angles"] = np.radians(s["angle_min"] + s["resolution"] * np.arange(n))
                s["local_dirs"] = np.stack([np.cos(s["angles"]), np.sin(s["angles"]),
                                            np.zeros(n)], axis=1)
        self.close()  # renderers belong to the previous model

    # ---- scheduling ----
    def due(self, t=None):
        """Streams whose next sample time has come (call once per control step)."""
        t = float(self.env.sim.data._data.time) if t is None else t
        half_step = 0.5 / self.env.control_freq
        out = []
        for stream, s in self.streams.items():
            period = 1.0 / s["rate"]
            nxt = self._next.get(stream)
            if nxt is None or t < nxt - 2 * period:  # first call, or time went back (reset)
                nxt = t
            if t + half_step >= nxt:
                out.append(stream)
                nxt += period
                if nxt <= t:  # fell behind: do not try to catch up
                    nxt = t + period
            self._next[stream] = nxt
        return out

    def reset(self):
        self._next.clear()
        self._resolve()

    # ---- reading ----
    def _mount_pose(self, s):
        d = self.env.sim.data._data
        Rb = d.xmat[s["body_id"]].reshape(3, 3)
        return d.xpos[s["body_id"]] + Rb @ s["pos"], Rb @ s["R"]

    def read(self, stream):
        """Read one stream now. Returns a dict (see the module docstring)."""
        s = self.streams[stream]
        t = float(self.env.sim.data._data.time)
        base = dict(stream=stream, type=s["type"], t=t, topic=s["topic"], frame=s["frame"])
        if s["type"] == "lidar":
            return dict(base, **self._lidar(s))
        if s["type"] == "imu":
            return dict(base, **self._imu(s))
        return dict(base, **self._camera(s))

    def _lidar(self, s):
        mj = self._mj
        m, d = self._model, self.env.sim.data._data
        origin, R = self._mount_pose(s)
        n = len(s["angles"])
        vec = (s["local_dirs"] @ R.T).ravel()
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        visual = np.array([0, 1, 0, 0, 0, 0], np.uint8)  # what a laser sees: the visual meshes
        mj.mj_multiRay(m, d, origin, vec, visual, 1, s["body_id"], geomid, dist, n,
                       s["max_range"])
        ranges = dist.astype(np.float32)
        hit = ranges >= 0
        if s["noise"] > 0:
            ranges[hit] += self.rng.normal(0, s["noise"], hit.sum()).astype(np.float32)
        ranges[~hit | (ranges > s["max_range"])] = np.inf
        return dict(ranges=ranges, angle_min=float(s["angles"][0]),
                    angle_max=float(s["angles"][-1]),
                    angle_increment=math.radians(s["resolution"]),
                    range_min=s["min_range"], range_max=s["max_range"],
                    scan_time=1.0 / s["rate"])

    def _imu(self, s):
        sd = self.env.sim.data._data.sensordata
        a = s["adr"]
        acc = np.array(sd[a["accelerometer"]:a["accelerometer"] + 3])
        gyro = np.array(sd[a["gyro"]:a["gyro"] + 3])
        w, x, y, z = sd[a["framequat"]:a["framequat"] + 4]
        if s["accel_noise"] > 0:
            acc += self.rng.normal(0, s["accel_noise"], 3)
        if s["gyro_noise"] > 0:
            gyro += self.rng.normal(0, s["gyro_noise"], 3)
        return dict(accel=acc, gyro=gyro, orientation=np.array([x, y, z, w]),  # ROS order
                    accel_noise=s["accel_noise"], gyro_noise=s["gyro_noise"])

    def _renderer(self, w, h, depth):
        key = (w, h, depth)
        if key not in self._renderers:
            mj, m = self._mj, self._model
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
            m.vis.global_.offheight = max(m.vis.global_.offheight, h)
            r = mj.Renderer(m, height=h, width=w)
            if depth:
                r.enable_depth_rendering()
            opt = mj.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[1] = 1   # visual geoms only
            opt.sitegroup[:] = 0   # no marker spheres
            self._renderers[key] = (r, opt)
        return self._renderers[key]

    def intrinsics(self, stream):
        s = self.streams[stream]
        fovy = 2 * math.atan(math.tan(math.radians(s["hfov"]) / 2) * s["height"] / s["width"])
        f = (s["height"] / 2) / math.tan(fovy / 2)
        return dict(width=s["width"], height=s["height"], fx=f, fy=f,
                    cx=(s["width"] - 1) / 2, cy=(s["height"] - 1) / 2)

    def _camera(self, s):
        depth = s["kind"] == "depth"
        r, opt = self._renderer(s["width"], s["height"], depth)
        r.update_scene(self.env.sim.data._data, camera=s["cam_id"], scene_option=opt)
        img = r.render()
        out = dict(intrinsics=self.intrinsics(s["name"]))
        if depth:
            valid = (img >= s["min_range"]) & (img <= s["max_range"])
            mm = np.where(valid, np.round(img * 1000), 0).astype(np.uint16)
            out.update(image=mm, encoding="16UC1")
        else:
            out.update(image=img, encoding="rgb8")
        return out

    def close(self):
        for r, _ in self._renderers.values():
            r.close()
        self._renderers.clear()


def describe(cfg=None):
    """One line per enabled stream, for logs."""
    lines = []
    for name, s in streams(cfg).items():
        extra = {"camera": lambda: f"{s['width']}x{s['height']} hfov {s['hfov']:g}",
                 "lidar": lambda: f"{s['min_range']:g}-{s['max_range']:g} m, {s['resolution']:g} deg",
                 "imu": lambda: "accel + gyro + orientation"}[s["type"]]()
        lines.append(f"{name:24s} {s['type']:6s} {s['rate']:4g} Hz  {extra:28s} -> {s['topic']}")
    return lines
