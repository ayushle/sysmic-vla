#!/usr/bin/python3
"""ROS 2 Humble bridge for the SYSMIC 2.0 RoboCasa simulation.

Started by the launch file (ros2 launch sysmic_v2_sim sim.launch.py), or by hand:

    ros2 run sysmic_v2_sim ros2_bridge.py

It uses the system Python that ROS 2 is built for, never the conda one, which is
why the first line of this file is /usr/bin/python3.

Publishes
    /clock          rosgraph_msgs/Clock         simulation time
    /joint_states   sensor_msgs/JointState      all 23 movable URDF joints
    /odom           nav_msgs/Odometry           pose since start, twist in base_link
    /tf             odom -> base_link
    sensors         every enabled stream of sensors.yaml, on its configured topic:
                    cameras -> sensor_msgs/Image + <topic dir>/camera_info,
                    lidars -> sensor_msgs/LaserScan, IMUs -> sensor_msgs/Imu.
                    A camera is only rendered while its topic has a subscriber.

Subscribes
    /cmd_vel                 geometry_msgs/Twist       linear.x (m/s), angular.z (rad/s)
    /sysmic/joint_command    sensor_msgs/JointState    name[] + position[]:
                             lift_joint (m), arm_{left,right}_{pan,shoulder,elbow,
                             wrist_flex,wrist_roll,gripper}_joint (rad)

Nodes that consume these topics should set use_sim_time:=true.
"""
import array
import json
import math
import socket
import struct
import threading

STATE_ADDR = ("127.0.0.1", 9880)  # must match sim/sysmic_sim_server.py
COMMAND_ADDR = ("127.0.0.1", 9881)
IMAGE_ADDR = ("127.0.0.1", 9882)  # TCP server here; the simulation connects and sends frames
ODOM_FRAME, BASE_FRAME = "odom", "base_link"


def stamp_parts(t):
    """Simulation time in seconds -> (sec, nanosec)."""
    sec = int(math.floor(t))
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000
    return sec, nanosec


def yaw_to_quat(yaw):
    """(x, y, z, w) for a rotation about Z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def twist_command(linear_x, angular_z):
    return json.dumps({"cmd_vel": [float(linear_x), float(angular_z)]}).encode()


def joint_command(names, positions):
    """JointState-style lists -> command packet, or None if there is nothing usable."""
    joints = {n: float(p) for n, p in zip(names, positions) if math.isfinite(p)}
    return json.dumps({"joints": joints}).encode() if joints else None


def main():
    import rclpy
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan
    from tf2_ros import TransformBroadcaster

    class SysmicBridge(Node):
        def __init__(self):
            super().__init__("sysmic_v2_bridge")
            self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rx.bind(STATE_ADDR)
            self.rx.setblocking(False)
            self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            self.clock_pub = self.create_publisher(Clock, "/clock", 10)
            self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
            self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
            self.tf_pub = TransformBroadcaster(self)
            self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
            self.create_subscription(JointState, "/sysmic/joint_command", self.on_joints, 10)

            self.scan_pubs, self.imu_pubs, self.image_pubs = {}, {}, {}
            self.lock = threading.Lock()
            self.image_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.image_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.image_server.bind(IMAGE_ADDR)
            self.image_server.listen(1)
            threading.Thread(target=self.image_loop, daemon=True).start()
            self.create_timer(0.5, self.report_subscribers)

            self.packets = 0
            self.last_sim_time = None
            self.create_timer(0.005, self.poll)  # 200 Hz; the sim sends at its control rate (30 Hz)
            self.create_timer(5.0, self.report)
            self.get_logger().info(
                f"Waiting for the simulation on udp {STATE_ADDR[1]} "
                "(it is started by sim.launch.py).")

        # ---- ROS -> simulation ----
        def on_cmd_vel(self, msg):
            self.tx.sendto(twist_command(msg.linear.x, msg.angular.z), COMMAND_ADDR)

        def on_joints(self, msg):
            if len(msg.name) != len(msg.position):
                self.get_logger().warn("joint_command ignored: name and position differ in length")
                return
            packet = joint_command(list(msg.name), list(msg.position))
            if packet:
                self.tx.sendto(packet, COMMAND_ADDR)

        # ---- simulation -> ROS ----
        def poll(self):
            while True:
                try:
                    packet, _ = self.rx.recvfrom(65535)
                except BlockingIOError:
                    return
                try:
                    self.publish(json.loads(packet.decode()))
                except (ValueError, KeyError, TypeError) as exc:
                    self.get_logger().warn(f"bad state packet ignored: {exc}")

        def publish(self, state):
            t = float(state["t"])
            if self.last_sim_time is not None and t < self.last_sim_time:
                self.get_logger().info("simulation time went backwards (scene was reset)")
            self.last_sim_time = t
            self.packets += 1
            sec, nanosec = stamp_parts(t)

            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = sec, nanosec
            self.clock_pub.publish(clock)

            js = JointState()
            js.header.stamp.sec, js.header.stamp.nanosec = sec, nanosec
            js.name = list(state["names"])
            js.position = [float(v) for v in state["pos"]]
            js.velocity = [float(v) for v in state["vel"]]
            self.joint_pub.publish(js)

            o = state["odom"]
            qx, qy, qz, qw = yaw_to_quat(float(o["yaw"]))

            odom = Odometry()
            odom.header.stamp.sec, odom.header.stamp.nanosec = sec, nanosec
            odom.header.frame_id = ODOM_FRAME
            odom.child_frame_id = BASE_FRAME
            odom.pose.pose.position.x = float(o["x"])
            odom.pose.pose.position.y = float(o["y"])
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            odom.twist.twist.linear.x = float(o["vx"])
            odom.twist.twist.linear.y = float(o["vy"])
            odom.twist.twist.angular.z = float(o["wz"])
            self.odom_pub.publish(odom)

            tf = TransformStamped()
            tf.header.stamp.sec, tf.header.stamp.nanosec = sec, nanosec
            tf.header.frame_id = ODOM_FRAME
            tf.child_frame_id = BASE_FRAME
            tf.transform.translation.x = float(o["x"])
            tf.transform.translation.y = float(o["y"])
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_pub.sendTransform(tf)

            for cam in (state.get("catalog") or {}).get("cameras", []):
                self.image_publishers(cam["topic"])
            for reading in (state.get("sensors") or {}).values():
                if reading["type"] == "lidar":
                    self.publish_scan(reading)
                elif reading["type"] == "imu":
                    self.publish_imu(reading)

        # ---- sensors ----
        def header(self, msg, t, frame):
            msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_parts(float(t))
            msg.header.frame_id = frame

        def publish_scan(self, r):
            pub = self.scan_pubs.get(r["topic"])
            if pub is None:
                pub = self.scan_pubs[r["topic"]] = self.create_publisher(LaserScan, r["topic"], 10)
            msg = LaserScan()
            self.header(msg, r["t"], r["frame"])
            msg.angle_min, msg.angle_max = float(r["angle_min"]), float(r["angle_max"])
            msg.angle_increment = float(r["angle_increment"])
            msg.scan_time = float(r["scan_time"])
            msg.time_increment = 0.0
            msg.range_min, msg.range_max = float(r["range_min"]), float(r["range_max"])
            msg.ranges = array.array("f", r["ranges"])
            pub.publish(msg)

        def publish_imu(self, r):
            pub = self.imu_pubs.get(r["topic"])
            if pub is None:
                pub = self.imu_pubs[r["topic"]] = self.create_publisher(Imu, r["topic"], 50)
            msg = Imu()
            self.header(msg, r["t"], r["frame"])
            (msg.orientation.x, msg.orientation.y, msg.orientation.z,
             msg.orientation.w) = (float(v) for v in r["orientation"])
            (msg.angular_velocity.x, msg.angular_velocity.y,
             msg.angular_velocity.z) = (float(v) for v in r["gyro"])
            (msg.linear_acceleration.x, msg.linear_acceleration.y,
             msg.linear_acceleration.z) = (float(v) for v in r["accel"])
            ga, aa = float(r["gyro_noise"]) ** 2, float(r["accel_noise"]) ** 2
            msg.angular_velocity_covariance = [ga, 0.0, 0.0, 0.0, ga, 0.0, 0.0, 0.0, ga]
            msg.linear_acceleration_covariance = [aa, 0.0, 0.0, 0.0, aa, 0.0, 0.0, 0.0, aa]
            msg.orientation_covariance = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]
            pub.publish(msg)

        def image_publishers(self, topic):
            with self.lock:
                pubs = self.image_pubs.get(topic)
                if pubs is None:
                    info_topic = topic.rsplit("/", 1)[0] + "/camera_info"
                    pubs = self.image_pubs[topic] = (
                        self.create_publisher(Image, topic, 2),
                        self.create_publisher(CameraInfo, info_topic, 2))
                return pubs

        def report_subscribers(self):
            """Tell the simulation which cameras to render (rendering is expensive)."""
            with self.lock:
                watched = [t for t, (img, info) in self.image_pubs.items()
                           if img.get_subscription_count() + info.get_subscription_count() > 0]
            self.tx.sendto(json.dumps({"camera_subscribers": watched}).encode(), COMMAND_ADDR)

        def image_loop(self):
            def recv_exact(conn, n):
                buf = bytearray(n)
                view, got = memoryview(buf), 0
                while got < n:
                    k = conn.recv_into(view[got:], n - got)
                    if not k:
                        raise ConnectionError("closed")
                    got += k
                return buf

            while rclpy.ok():
                try:
                    conn, _ = self.image_server.accept()
                except OSError:
                    return
                self.get_logger().info("camera stream from the simulation connected")
                try:
                    while rclpy.ok():
                        hlen, plen = struct.unpack(">II", recv_exact(conn, 8))
                        head = json.loads(recv_exact(conn, hlen).decode())
                        self.publish_image(head, recv_exact(conn, plen))
                except (ConnectionError, OSError, ValueError) as exc:
                    self.get_logger().info(f"camera stream closed ({exc})")
                finally:
                    conn.close()

        def publish_image(self, head, pixels):
            image_pub, info_pub = self.image_publishers(head["topic"])
            msg = Image()
            self.header(msg, head["t"], head["frame"])
            msg.height, msg.width = int(head["height"]), int(head["width"])
            msg.encoding = head["encoding"]  # rgb8 or 16UC1 (millimetres)
            msg.step = msg.width * (2 if msg.encoding == "16UC1" else 3)
            msg.is_bigendian = 0
            # array.array skips the per-byte check rclpy does for lists / bytes
            msg.data = array.array("B", pixels)
            k = head["intrinsics"]
            info = CameraInfo()
            info.header = msg.header
            info.height, info.width = msg.height, msg.width
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            info.k = [k["fx"], 0.0, k["cx"], 0.0, k["fy"], k["cy"], 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [k["fx"], 0.0, k["cx"], 0.0, 0.0, k["fy"], k["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
            image_pub.publish(msg)
            info_pub.publish(info)

        def report(self):
            if self.packets == 0:
                self.get_logger().warn("No data from the simulation yet. Is sysmic_sim_server.py running?")
            else:
                self.get_logger().info(
                    f"{self.packets / 5.0:.0f} state msgs/s, sim time {self.last_sim_time:.1f} s")
            self.packets = 0

    rclpy.init()
    node = SysmicBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
