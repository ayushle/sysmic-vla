#!/usr/bin/env python3
"""SYSMIC 2.0 in a RoboCasa kitchen, as a server for the ROS 2 bridge.

Normally started by the launch file:

    ros2 launch sysmic_v2_sim sim.launch.py

By hand, in the robocasa conda env (Python 3.11):

    python sysmic_sim_server.py                    # kitchen layout 1, style 1
    python sysmic_sim_server.py --layout 5 --style 12
    python sysmic_sim_server.py --no-viewer        # headless

The two sides talk over UDP on this machine only (JSON text):

    state    server -> 127.0.0.1:9880   every control step (sim.control_freq, 30 Hz)
    command  bridge -> 127.0.0.1:9881   whenever ROS sends something

State packet
    {"t": sim time (s), "seq": n,
     "names": [...], "pos": [...], "vel": [...],        joints, URDF names
     "odom": {"x","y","yaw","vx","vy","wz"},            pose since start, twist in base frame
     "world": {"x","y","yaw"}}                          pose in the kitchen's frame
     "sensors": {stream: reading}                       lidar / IMU readings due this step
     "catalog": {"cameras": [{topic, frame, stream}]}   about once a second

Command packet (any subset of the keys)
    {"cmd_vel": [v m/s, w rad/s],
     "joints": {"lift_joint": 0.3, "arm_left_elbow_joint": -1.0,
                "arm_right_gripper_joint": 1.2, ...},
     "stop": true,
     "camera_subscribers": [topic, ...]}               camera topics somebody listens to

Camera frames (only for watched topics) -> tcp 127.0.0.1:9882, each one:
    >II header length, payload length | header JSON {stream, topic, frame, t, encoding,
    intrinsics, height, width} | raw pixels: rgb8 (H*W*3 bytes) or 16UC1 (mm, H*W*2)

Sensors come from sensors.yaml (see sysmic_v2_robocasa/sysmic_sensors.py).

Safety: if no cmd_vel arrives for CMD_VEL_TIMEOUT seconds the base stops.
"""
import argparse
import json
import queue
import socket
import struct
import threading
import time

import mujoco
import numpy as np

import sysmic_robosuite as S

STATE_ADDR = ("127.0.0.1", 9880)
COMMAND_ADDR = ("127.0.0.1", 9881)
IMAGE_ADDR = ("127.0.0.1", 9882)  # TCP; the bridge listens, camera frames go there
SUBSCRIBER_TIMEOUT = 3.0  # s; render a camera only while the bridge reports subscribers
CMD_VEL_TIMEOUT = 0.5  # s of wall time
MAX_SPEED, MAX_TURN = 0.5, 1.0  # m/s, rad/s: top speeds of the base motors
# The base motors run slightly slow against joint friction; add it back.
SPEED_TRIM, TURN_TRIM = 0.003, 0.012

ARM = [f"{side}_{j}" for side in ("left", "right") for j in S.ARM_JOINTS]  # internal names
CASTERS = ("FL", "FR", "RL", "RR")

# URDF joint name -> internal name used by the robosuite model and Commands
URDF_TO_INTERNAL = {f"arm_{side}_{u}": f"{side}_{j}" for side in ("left", "right")
                    for u, j in zip(S.URDF_ARM_JOINTS, S.ARM_JOINTS)}
URDF_TO_INTERNAL.update({f"arm_{side}_gripper_joint": f"{side}_gripper"
                         for side in ("left", "right")})
# commands may also use the internal names (left_elbow_flex, right_gripper, ...)
ALIASES = {**URDF_TO_INTERNAL, **{v: v for v in URDF_TO_INTERNAL.values()}}


def joint_map(env):
    """URDF joint name -> joint name in the RoboCasa model (the 23 movable URDF joints)."""
    robot = env.robots[0]
    r = robot.robot_model.naming_prefix  # robot0_
    b = robot.robot_model.base.naming_prefix  # mobilebase0_
    m = {"lift_joint": f"{r}torso_lift_joint"}
    for urdf, internal in URDF_TO_INTERNAL.items():
        if internal.endswith("_gripper"):
            side = internal.split("_")[0]
            m[urdf] = f"{robot.gripper[side].naming_prefix}jaw_joint"
        else:
            m[urdf] = r + internal
    for side in ("left", "right"):
        m[f"wheel_{side}_joint"] = f"{b}passive_wheel_{side}_spin"
    for c in CASTERS:
        m[f"caster_{c}_swivel_joint"] = f"{b}passive_caster_{c}_swivel"
        m[f"caster_{c}_wheel_joint"] = f"{b}passive_caster_{c}_wheel"
    return m


class Commands:
    """Latest targets received from ROS; arms, lift and grippers hold their last value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cmd_vel = (0.0, 0.0)
        self.cmd_vel_time = 0.0
        self.lift = 0.2
        self.arm = {name: S.ARM_READY[i % 5] for i, name in enumerate(ARM)}
        self.gripper = {"left": S.GRIPPER_OPEN, "right": S.GRIPPER_OPEN}
        self.camera_topics, self.camera_topics_time = set(), 0.0

    def watched_cameras(self, now):
        """Camera topics somebody subscribes to (reported by the bridge)."""
        if now - self.camera_topics_time > SUBSCRIBER_TIMEOUT:
            return set()
        return self.camera_topics

    def apply(self, msg, now):
        """Merge one command packet. Returns a list of names that were not understood."""
        unknown = []
        if "camera_subscribers" in msg:
            self.camera_topics = set(msg["camera_subscribers"] or [])
            self.camera_topics_time = now
        if msg.get("stop"):
            self.cmd_vel = (0.0, 0.0)
        if "cmd_vel" in msg:
            v, w = msg["cmd_vel"]
            self.cmd_vel = (float(v), float(w))
            self.cmd_vel_time = now
        for name, value in (msg.get("joints") or {}).items():
            if name.startswith(("wheel_", "caster_")):
                continue  # passive in the sim (e.g. sent by joint_state_publisher_gui)
            name_in = name
            name = ALIASES.get(name, name)
            value = float(value)
            if not np.isfinite(value):
                unknown.append(name_in)
            elif name == "lift_joint":
                self.lift = float(np.clip(value, *S.LIFT_RANGE))
            elif name in self.arm:
                self.arm[name] = value
            elif name in ("left_gripper", "right_gripper"):
                self.gripper[name.split("_")[0]] = value
            else:
                unknown.append(name_in)
        return unknown

    def base(self, now):
        """(forward, turn) as fractions of top speed; zero once cmd_vel is stale."""
        if now - self.cmd_vel_time > CMD_VEL_TIMEOUT:
            return 0.0, 0.0
        v, w = self.cmd_vel
        v = v + np.sign(v) * SPEED_TRIM
        w = w + np.sign(w) * TURN_TRIM
        return float(np.clip(v / MAX_SPEED, -1, 1)), float(np.clip(w / MAX_TURN, -1, 1))

    def action(self, env, now):
        return S.make_action(
            env, base=self.base(now), lift=self.lift,
            right_arm=[self.arm[f"right_{j}"] for j in S.ARM_JOINTS],
            left_arm=[self.arm[f"left_{j}"] for j in S.ARM_JOINTS],
            right_gripper=S.gripper_action(self.gripper["right"]),
            left_gripper=S.gripper_action(self.gripper["left"]))


class StateReader:
    """Reads joint and base state out of the simulation."""

    def __init__(self, env):
        self.env = env
        self.names = list(joint_map(env))
        model = env.sim.model._model
        ids = [model.joint(n).id for n in joint_map(env).values()]
        self.qpos = [int(model.jnt_qposadr[i]) for i in ids]
        self.qvel = [int(model.jnt_dofadr[i]) for i in ids]
        prefix = env.robots[0].robot_model.naming_prefix
        self.base_body = model.body(f"{prefix}base_link").id
        self.origin = None  # odom frame = where the robot stood at start
        self.last = None  # previous (x, y, yaw, time), for the twist

    def base_pose(self):
        data = self.env.sim.data._data
        pos = data.xpos[self.base_body]
        mat = data.xmat[self.base_body].reshape(3, 3)
        return float(pos[0]), float(pos[1]), float(np.arctan2(mat[1, 0], mat[0, 0]))

    def read(self, seq, peek=False):
        model, data = self.env.sim.model._model, self.env.sim.data._data
        x, y, yaw = self.base_pose()
        if self.origin is None:
            self.origin = (x, y, yaw)
        ox, oy, oyaw = self.origin
        c, s = np.cos(oyaw), np.sin(oyaw)
        dx, dy = x - ox, y - oy
        # twist in the base frame, averaged over the last control step so that it
        # integrates to exactly the pose change reported in "odom"
        t = float(data.time)
        if self.last is not None and t > self.last[3]:
            px, py, pyaw, pt = self.last
            dt = t - pt
            wx, wy = (x - px) / dt, (y - py) / dt
            mid = pyaw + ((yaw - pyaw + np.pi) % (2 * np.pi) - np.pi) / 2
            vx = np.cos(mid) * wx + np.sin(mid) * wy
            vy = -np.sin(mid) * wx + np.cos(mid) * wy
            wz = ((yaw - pyaw + np.pi) % (2 * np.pi) - np.pi) / dt
        else:
            vel = np.zeros(6)  # [angular, linear] in the base frame
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_body, vel, 1)
            vx, vy, wz = vel[3], vel[4], vel[2]
        if not peek:
            self.last = (x, y, yaw, t)
        return {
            "t": float(data.time), "seq": seq, "names": self.names,
            "pos": [float(data.qpos[i]) for i in self.qpos],
            "vel": [float(data.qvel[i]) for i in self.qvel],
            "odom": {"x": float(c * dx + s * dy), "y": float(-s * dx + c * dy),
                     "yaw": float((yaw - oyaw + np.pi) % (2 * np.pi) - np.pi),
                     "vx": float(vx), "vy": float(vy), "wz": float(wz)},
            "world": {"x": x, "y": y, "yaw": yaw},
        }


def open_sockets():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(COMMAND_ADDR)
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return rx, tx


def drain(rx, commands, now, log=print):
    """Apply every command packet that has arrived."""
    count = 0
    while True:
        try:
            packet, _ = rx.recvfrom(65535)
        except BlockingIOError:
            return count
        try:
            unknown = commands.apply(json.loads(packet.decode()), now)
        except (ValueError, TypeError, KeyError) as exc:
            log(f"ignored a bad command packet: {exc}")
            continue
        if unknown:
            log(f"ignored unknown joints: {', '.join(unknown)}")
        count += 1


class ImageSender(threading.Thread):
    """Streams raw camera frames to the bridge over TCP on this machine (no encoding:
    compressing in Python was the bottleneck). Frames are dropped, never queued up,
    when the link is slow."""

    def __init__(self, log=print):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=8)
        self.sock = None
        self.log = log
        self.running = True
        self._next_try = 0.0

    @property
    def connected(self):
        if self.sock is None and time.time() > self._next_try:
            self._next_try = time.time() + 2.0
            try:
                self.sock = socket.create_connection(IMAGE_ADDR, timeout=0.2)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.settimeout(None)
                self.log(f"camera stream connected to the bridge (tcp {IMAGE_ADDR[1]})")
            except OSError:
                self.sock = None
        return self.sock is not None

    def put(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass  # drop: the consumer is slower than the cameras

    def run(self):
        while self.running:
            try:
                f = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            img = np.ascontiguousarray(f["image"])
            payload = memoryview(img).cast("B")
            header = json.dumps({k: f[k] for k in ("stream", "topic", "frame", "t", "encoding",
                                                   "intrinsics")}
                                | {"height": int(img.shape[0]), "width": int(img.shape[1])}).encode()
            sock = self.sock
            if sock is None:
                continue
            try:
                sock.sendall(struct.pack(">II", len(header), len(payload)) + header)
                sock.sendall(payload)
            except OSError:
                self.log("camera stream to the bridge lost; retrying")
                self.sock = None
                try:
                    sock.close()
                except OSError:
                    pass


def _jsonable(reading):
    """Lidar / IMU reading -> plain JSON types (ranges in mm resolution, inf kept)."""
    out = {}
    for k, v in reading.items():
        if isinstance(v, np.ndarray):
            out[k] = [round(float(x), 4) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def serve(env, render=True, realtime=True, max_steps=None, log=print):
    rx, tx = open_sockets()
    commands, reader = Commands(), StateReader(env)
    sensors = S.Sensors(env)
    catalog = {"cameras": [dict(topic=s["topic"], frame=s["frame"], stream=n)
                           for n, s in sensors.streams.items() if s["type"] == "camera"]}
    images = ImageSender(log)
    images.start()
    period = 1.0 / env.control_freq
    seq, last_report, behind, last_catalog = 0, time.time(), 0.0, 0.0
    try:
        while max_steps is None or seq < max_steps:
            start = time.time()
            drain(rx, commands, start, log)
            if render and S.viewer_closed(env):
                break
            env.step(commands.action(env, start))
            if render:
                env.render()
            state = reader.read(seq)
            watched = commands.watched_cameras(start)
            readings = {}
            for name in sensors.due():
                s = sensors.streams[name]
                if s["type"] != "camera":
                    readings[name] = _jsonable(sensors.read(name))
                elif s["topic"] in watched and images.connected:
                    images.put(sensors.read(name))
            if readings:
                state["sensors"] = readings
            if start - last_catalog > 1.0:  # tells the bridge which camera topics exist
                state["catalog"] = catalog
                last_catalog = start
            tx.sendto(json.dumps(state).encode(), STATE_ADDR)
            seq += 1
            spent = time.time() - start
            behind = 0.95 * behind + 0.05 * (period / max(spent, period) if realtime else 1.0)
            if realtime and spent < period:
                time.sleep(period - spent)
            if time.time() - last_report > 5.0:
                last_report = time.time()
                x, y, yaw = (reader.read(seq, peek=True)["odom"][k] for k in ("x", "y", "yaw"))
                log(f"sim time {env.sim.data._data.time:7.1f} s | odom x {x:+.2f} y {y:+.2f} "
                    f"yaw {yaw:+.2f} | speed {behind:.2f}x real time")
    except KeyboardInterrupt:
        pass
    finally:
        images.running = False
        sensors.close()
        rx.close()
        tx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Kitchen")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--viewer", choices=["true", "false"], default="true",
                    help="same switch in a form launch files can pass")
    ap.add_argument("--sensors", default=None,
                    help="sensors.yaml to use (default: $SYSMIC_SENSORS or the standard one)")
    args = ap.parse_args()

    import robosuite

    extra = {}
    if args.task not in robosuite.ALL_ENVIRONMENTS or args.task == "Kitchen":
        extra = dict(layout_ids=args.layout, style_ids=args.style)
    view = not args.no_viewer and args.viewer == "true"
    env = S.make_env(args.task, sensors=args.sensors, has_renderer=view,
                     **(dict(renderer="mjviewer", render_camera=None) if view else {}), **extra)
    env.reset()
    cfg = S.sysmic_sensors.active()
    print(f"Sensors from {cfg['_path']} (control rate {env.control_freq:g} Hz):")
    for line in S.sysmic_sensors.describe(cfg) or ["(none enabled)"]:
        print("  " + line)
    print(f"SYSMIC simulation server running. State -> udp {STATE_ADDR[1]}, "
          f"commands <- udp {COMMAND_ADDR[1]}, camera frames -> tcp {IMAGE_ADDR[1]}. "
          "Ctrl+C or close the viewer to stop.")
    serve(env, render=view)
    env.close()
    print("Stopped.")


if __name__ == "__main__":
    main()
