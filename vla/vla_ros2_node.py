#!/usr/bin/env python3
"""ROS 2 wrapper around vla-pipeline-v2.py.

Takes camera frames from a ROS image topic instead of V4L2, runs the same v2 pipeline
(undistort -> AprilTag-anchored metric depth -> Grounding DINO + SAM -> clouds) and publishes
the results as ROS messages. All the perception code is imported from vla-pipeline-v2.py, so
tuning done there (tag ids, tag size, edge bias, dense fit, ...) applies here unchanged.

Subscribes
  <image_topic>              sensor_msgs/Image (or CompressedImage if `compressed` is true)
  <camera_info_topic>        sensor_msgs/CameraInfo (optional; '' = use the calibration in vla-pipeline-v2.py)
  ~/object_name              std_msgs/String   change the target object at runtime

Publishes (all stamped with the input image's stamp, frame = camera optical frame: x right, y down, z forward)
  ~/object_cloud             sensor_msgs/PointCloud2  x y z rgb, masked object only
  ~/scene_cloud              sensor_msgs/PointCloud2  x y z rgb, whole scene (subsampled by scene_stride)
  ~/image_undistorted        sensor_msgs/Image        bgr8, the frame every output refers to
  ~/camera_info_undistorted  sensor_msgs/CameraInfo   K_new, zero distortion
  ~/depth                    sensor_msgs/Image        32FC1, metres
  ~/mask                     sensor_msgs/Image        mono8, 0 / 255
  ~/status_image             sensor_msgs/Image        bgr8, tags + mask + calibration overlay
  ~/calib                    std_msgs/String          JSON: calibration diagnostics, table plane, tag detections

Services
  ~/capture                  std_srvs/Trigger   process the newest frame now (works in both modes)

Parameters
  image_topic          /camera/image_raw
  compressed           false      subscribe to CompressedImage on image_topic (e.g. /camera/image_raw/compressed)
  camera_info_topic    ''         if set, K / D come from CameraInfo instead of the hardcoded calibration
  image_is_rectified   false      true if image_topic is already rectified (image_rect): no undistortion,
                                  K is taken from CameraInfo P (or the hardcoded K if no camera_info)
  object_name          cup
  mode                 continuous continuous = process frames back to back; trigger = only on ~/capture
  min_period           0.0        continuous mode: minimum seconds between two runs
  frame_id             ''         override the output frame id (default: the input image's frame id)
  publish_scene        true
  scene_stride         4          pixel stride for the scene cloud (1 = full resolution, ~2 M points at 1080p)
  object_stride        1          pixel stride for the object cloud
  low_vram             false      unload models between uses (slow, but fits small GPUs)
  preload_models       true       load depth + segmentation models at startup
  pipeline_path        ''         path to vla-pipeline-v2.py (default: next to this file)

Run
  source /opt/ros/jazzy/setup.bash
  python3 vla_ros2_node.py --ros-args -p image_topic:=/camera/image_raw -p object_name:=cup
  python3 vla_ros2_node.py mug --ros-args -p image_topic:=/camera/image_raw
                                         the object can also be given as plain arguments
                                         ("red block" or red block); they override object_name.
                                         In the default continuous mode the node keeps running
                                         the newest frame and republishes ~/object_cloud each time.
  python3 vla_ros2_node.py --ros-args -p mode:=trigger   then   ros2 service call /vla_perception/capture std_srvs/srv/Trigger
  ros2 topic pub --once /vla_perception/object_name std_msgs/msg/String "{data: 'red block'}"
"""
import importlib.util
import json
import os
import threading
import time
import traceback

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

import cv2
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger


def load_pipeline(path):
    """Import vla-pipeline-v2.py (the hyphens make it unimportable by name)."""
    spec = importlib.util.spec_from_file_location("vla_pipeline_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PC2_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


# Image conversion without cv_bridge (the Jazzy cv_bridge is built against NumPy 1 and breaks under NumPy 2)
_ENC = {"bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3), "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3), "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1), "32FC1": (np.float32, 1), "yuv422_yuy2": (np.uint8, 2), "yuyv": (np.uint8, 2),
        "uyvy": (np.uint8, 2), "yuv422": (np.uint8, 2)}


def imgmsg_to_bgr(msg):
    enc = msg.encoding
    if enc.startswith("bayer_"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        code = {"bayer_rggb8": cv2.COLOR_BayerRG2BGR, "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGB2BGR, "bayer_grbg8": cv2.COLOR_BayerGR2BGR}[enc]
        return cv2.cvtColor(a, code)
    if enc not in _ENC:
        raise ValueError(f"unsupported encoding '{enc}'")
    dt, ch = _ENC[enc]
    dt = np.dtype(dt).newbyteorder(">" if msg.is_bigendian else "<")
    row = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch * dt.itemsize]
    a = row.copy().view(dt).reshape(msg.height, msg.width, ch)
    if enc in ("bgr8", "8UC3"):
        return a
    if enc == "rgb8":
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    if enc in ("yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(a, cv2.COLOR_YUV2BGR_YUY2)
    if enc in ("uyvy", "yuv422"):
        return cv2.cvtColor(a, cv2.COLOR_YUV2BGR_UYVY)
    a = a[..., 0]
    if a.dtype != np.uint8:                           # 16-bit / float mono: scale to 8 bit
        a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


def compressed_to_bgr(msg):
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode '{msg.format}'")
    return img


def to_imgmsg(header, arr, encoding):
    arr = np.ascontiguousarray(arr)
    m = Image()
    m.header = header
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.is_bigendian = False
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


def make_cloud(header, pts, bgr):
    """PointCloud2 (x y z rgb, unorganised) from Nx3 float points and Nx3 uint8 BGR colours."""
    n = len(pts)
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    if n:
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        bgr = bgr.astype(np.uint32)
        arr["rgb"] = ((bgr[:, 2] << 16) | (bgr[:, 1] << 8) | bgr[:, 0]).view(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, n
    msg.fields = _PC2_FIELDS
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = arr.tobytes()
    return msg


class VlaPerceptionNode(Node):
    def __init__(self, **kwargs):
        super().__init__("vla_perception", **kwargs)
        p = self.declare_parameter
        p("image_topic", "/camera/image_raw")
        p("compressed", False)
        p("camera_info_topic", "")
        p("image_is_rectified", False)
        p("object_name", "cup")
        p("mode", "continuous")
        p("min_period", 0.0)
        p("frame_id", "")
        p("publish_scene", True)
        p("scene_stride", 4)
        p("object_stride", 1)
        p("low_vram", False)
        p("preload_models", True)
        p("pipeline_path", "")
        g = lambda name: self.get_parameter(name).value

        path = g("pipeline_path") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "vla-pipeline-v2.py")
        self.vla = load_pipeline(path)
        self.de = self.vla.depth_estimation(low_vram=g("low_vram"))
        self.get_logger().info(f"pipeline loaded from {path}")

        self.object_name = g("object_name")
        self.mode = g("mode")
        if self.mode not in ("continuous", "trigger"):
            raise ValueError("mode must be 'continuous' or 'trigger'")

        self._lock = threading.Lock()          # guards _latest / _cam_info
        self._proc_lock = threading.Lock()     # one pipeline run at a time (models are not thread safe)
        self._new_frame = threading.Event()
        self._latest = None                    # (header, bgr frame)
        self._last_processed_stamp = None
        self._cam_info = None
        self._stop = False

        sensor_cb = ReentrantCallbackGroup()
        srv_cb = MutuallyExclusiveCallbackGroup()

        if g("compressed"):
            self.create_subscription(CompressedImage, g("image_topic"), self._on_compressed,
                                     qos_profile_sensor_data, callback_group=sensor_cb)
        else:
            self.create_subscription(Image, g("image_topic"), self._on_image,
                                     qos_profile_sensor_data, callback_group=sensor_cb)
        if g("camera_info_topic"):
            self.create_subscription(CameraInfo, g("camera_info_topic"), self._on_info,
                                     qos_profile_sensor_data, callback_group=sensor_cb)
        self.create_subscription(String, "~/object_name", self._on_object, 10, callback_group=sensor_cb)

        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub_obj = self.create_publisher(PointCloud2, "~/object_cloud", reliable)
        self.pub_scene = self.create_publisher(PointCloud2, "~/scene_cloud", reliable)
        self.pub_img = self.create_publisher(Image, "~/image_undistorted", reliable)
        self.pub_info = self.create_publisher(CameraInfo, "~/camera_info_undistorted", reliable)
        self.pub_depth = self.create_publisher(Image, "~/depth", reliable)
        self.pub_mask = self.create_publisher(Image, "~/mask", reliable)
        self.pub_status = self.create_publisher(Image, "~/status_image", reliable)
        self.pub_calib = self.create_publisher(String, "~/calib", reliable)
        self.create_service(Trigger, "~/capture", self._on_capture, callback_group=srv_cb)

        if g("preload_models") and not g("low_vram"):
            self.get_logger().info("preloading models...")
            self.de.depth_model_init()
            self.de.seg_model_init()

        self._worker = None
        if self.mode == "continuous":
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
        self.get_logger().info(
            f"ready: image '{g('image_topic')}'{' (compressed)' if g('compressed') else ''}, "
            f"camera_info '{g('camera_info_topic') or 'built-in calibration'}', object '{self.object_name}', "
            f"mode {self.mode}")

    # ------------------------------------------------------------------ inputs
    def _store(self, header, frame):
        with self._lock:
            self._latest = (header, frame)
        self._new_frame.set()

    def _on_image(self, msg):
        try:
            self._store(msg.header, imgmsg_to_bgr(msg))
        except Exception as e:
            self.get_logger().error(f"image conversion failed: {e}", throttle_duration_sec=5.0)

    def _on_compressed(self, msg):
        try:
            self._store(msg.header, compressed_to_bgr(msg))
        except Exception as e:
            self.get_logger().error(f"image decode failed: {e}", throttle_duration_sec=5.0)

    def _on_info(self, msg):
        with self._lock:
            self._cam_info = msg

    def _on_object(self, msg):
        name = msg.data.strip()
        if name:
            self.object_name = name
            self.get_logger().info(f"object set to '{name}'")

    # ------------------------------------------------------------------ processing
    def _intrinsics(self, w, h):
        """(K, D, undistort) for a frame of size (w, h)."""
        rectified = self.get_parameter("image_is_rectified").value
        with self._lock:
            info = self._cam_info
        if info is not None and (info.width, info.height) == (w, h):
            if rectified:
                return np.array(info.p, np.float64).reshape(3, 4)[:, :3].copy(), np.zeros((1, 5)), False
            D = np.array(info.d, np.float64).reshape(1, -1) if len(info.d) else np.zeros((1, 5))
            return np.array(info.k, np.float64).reshape(3, 3), D, True
        if info is not None:
            self.get_logger().warn(f"camera_info is {info.width}x{info.height}, image {w}x{h}; "
                                   "using the built-in calibration", throttle_duration_sec=10.0)
        K = self.vla.intrinsics_for(w, h)
        if rectified:
            return K, np.zeros((1, 5)), False
        return K, self.vla.DIST_COEFFS.copy(), True

    def process_latest(self):
        """Run the pipeline on the newest frame and publish. Returns (ok, message)."""
        with self._lock:
            latest = self._latest
        if latest is None:
            return False, "no image received yet"
        header_in, frame = latest
        with self._proc_lock:
            t0 = time.time()
            self._last_processed_stamp = (header_in.stamp.sec, header_in.stamp.nanosec)
            h, w = frame.shape[:2]
            try:
                K, D, und = self._intrinsics(w, h)
                self.de.PCD_STRIDE = max(1, int(self.get_parameter("object_stride").value))
                sc = self.de.scene_clouds(frame, K, D, object_name=self.object_name, undistort=und)
            except Exception as e:
                self.get_logger().error(f"pipeline failed: {e}\n{traceback.format_exc()}")
                return False, f"pipeline error: {e}"
            if sc is None:
                self._publish_calib(header_in, None)
                return False, f"no result (calibration: {self.de.last_calib})"
            self._publish(header_in, sc)
            msg = (f"{len(sc['obj_pts'])} object points, method {sc['calib'].get('method')}, "
                   f"{time.time() - t0:.2f} s")
            self.get_logger().info(msg)
            return True, msg

    def _loop(self):
        min_period = float(self.get_parameter("min_period").value)
        last = 0.0
        while rclpy.ok() and not self._stop:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()
            dt = min_period - (time.time() - last)
            if dt > 0:
                time.sleep(dt)
            last = time.time()
            self.process_latest()

    def _on_capture(self, req, resp):
        resp.success, resp.message = self.process_latest()
        return resp

    # ------------------------------------------------------------------ outputs
    def _header(self, header_in):
        h = Header()
        h.stamp = header_in.stamp
        h.frame_id = self.get_parameter("frame_id").value or header_in.frame_id
        return h

    def _publish(self, header_in, sc):
        hdr = self._header(header_in)
        frame = sc["frame"]
        self.pub_obj.publish(make_cloud(hdr, sc["obj_pts"], sc["obj_bgr"]))

        if self.get_parameter("publish_scene").value and self.pub_scene.get_subscription_count() > 0:
            s = max(1, int(self.get_parameter("scene_stride").value))
            pix = sc["scene_pix"]
            keep = (pix[:, 0] % s == 0) & (pix[:, 1] % s == 0)
            pts, pix = sc["scene_pts"][keep], pix[keep]
            self.pub_scene.publish(make_cloud(hdr, pts, frame[pix[:, 1], pix[:, 0]]))

        if self.pub_img.get_subscription_count() > 0:
            m = to_imgmsg(hdr, frame, "bgr8")
            self.pub_img.publish(m)
        info = CameraInfo()
        info.header = hdr
        info.height, info.width = frame.shape[:2]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        K = np.asarray(sc["K"], np.float64)
        info.k = K.ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        info.p = np.c_[K, np.zeros(3)].ravel().tolist()
        self.pub_info.publish(info)

        if self.pub_depth.get_subscription_count() > 0:
            m = to_imgmsg(hdr, sc["z_map"].astype(np.float32), "32FC1")
            self.pub_depth.publish(m)
        if self.pub_mask.get_subscription_count() > 0:
            m = to_imgmsg(hdr, sc["mask"].astype(np.uint8), "mono8")
            self.pub_mask.publish(m)
        if self.pub_status.get_subscription_count() > 0:
            m = to_imgmsg(hdr, self.vla.draw_status(frame, self.de, sc), "bgr8")
            self.pub_status.publish(m)
        self._publish_calib(header_in, sc)

    def _publish_calib(self, header_in, sc):
        d = dict(stamp=[header_in.stamp.sec, header_in.stamp.nanosec], object=self.object_name,
                 ok=sc is not None, calib=self.de.last_calib,
                 plane=None if sc is None else sc["plane"],
                 detections=self.de.detections)
        self.pub_calib.publish(String(data=json.dumps(d, default=_json_default)))

    def destroy_node(self):
        self._stop = True
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        super().destroy_node()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def main():
    import sys

    from rclpy.parameter import Parameter
    from rclpy.utilities import remove_ros_args

    rclpy.init()
    words = [a for a in remove_ros_args(sys.argv)[1:] if a.strip()]
    overrides = [Parameter("object_name", value=" ".join(words))] if words else []
    node = VlaPerceptionNode(parameter_overrides=overrides)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
