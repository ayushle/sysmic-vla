#!/usr/bin/env python3
"""Score vla/vla_ros2_node.py (repo root) against the simulation's ground truth.

Needs the simulation (ros2 launch sysmic_v2_sim vla_test.launch.py) and the node running
in trigger mode on one of the robot's cameras, e.g.

    ~/envs/cvenev/bin/python ~/sysmic-sim/sysmic_ws/vla/vla_ros2_node.py --ros-args \\
        -p image_topic:=/left_gripper_camera/image_raw \\
        -p camera_info_topic:=/left_gripper_camera/camera_info \\
        -p image_is_rectified:=true -p mode:=trigger -p use_sim_time:=true

then, with ROS sourced:

    ~/envs/cvenev/bin/python vla_eval.py --camera left_gripper_camera --runs 3

For every object in the kitchen and every run it sets the node's object name, asks the
simulation for a ground truth snapshot of that camera (true depth along the optical axis
and which object each pixel shows; see sim/sysmic_sim_server.py), calls ~/capture and
compares what the node publishes on ~/depth, ~/mask, ~/object_cloud and ~/calib:

  mask IoU            node mask vs the object's true pixels
  depth MSE / RMSE    node depth - true depth on pixels both call the object (mm^2 / mm)
  depth bias          mean of that difference (mm; + = too far)
  point RMSE          3D distance between node and true points at the same pixels (mm)
  centroid error      node object cloud centroid vs true centroid of the visible surface (mm)
  scene depth RMSE    whole image, every pixel with a node depth (mm)
  calibration         tags seen, fit method, tag leave-one-out error, flatness, check tags

Writes report.md, results.json and one overlay image per object to --out.
"""
import argparse
import datetime
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger

COMMAND_ADDR = ("127.0.0.1", 9881)  # sim/sysmic_sim_server.py


def image_to_array(msg):
    dtype = {"32FC1": np.float32, "mono8": np.uint8, "8UC1": np.uint8, "16UC1": np.uint16}[msg.encoding]
    a = np.frombuffer(bytes(msg.data), dtype=np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<"))
    return a.reshape(msg.height, msg.step // np.dtype(dtype).itemsize)[:, :msg.width].astype(np.float64)


class Collector:
    """Keeps the newest message of each node output, tagged with its image stamp."""

    def __init__(self, node, ns):
        self.lock = threading.Lock()
        self.latest = {}
        for topic, typ in (("depth", Image), ("mask", Image), ("object_cloud", PointCloud2),
                           ("calib", String)):
            node.create_subscription(typ, f"{ns}/{topic}", self._cb(topic), 5)

    def _cb(self, topic):
        def cb(msg):
            with self.lock:
                self.latest[topic] = (time.time(), msg)
        return cb

    def wait_after(self, t0, timeout=60.0):
        """All four outputs received after wall time t0."""
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                got = {k: m for k, (t, m) in self.latest.items() if t >= t0}
            if len(got) == 4 or ("calib" in got and not json.loads(got["calib"].data).get("ok", True)):
                return got
            time.sleep(0.05)
        return got


def request_ground_truth(camera, path, timeout=10.0):
    if os.path.exists(path):
        os.remove(path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps({"gt_snapshot": {"camera": camera, "path": path}}).encode(), COMMAND_ADDR)
    sock.close()
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path):
            time.sleep(0.05)
            return dict(np.load(path, allow_pickle=False))
        time.sleep(0.05)
    raise TimeoutError("no ground truth from the simulation (is sysmic_sim_server.py running?)")


def backproject(z, mask, fx, fy, cx, cy):
    v, u = np.nonzero(mask)
    d = z[v, u]
    return np.c_[(u - cx) * d / fx, (v - cy) * d / fy, d]


def score(gt, target, out, calib):
    """Metrics for one capture; all lengths in mm."""
    fx, fy, cx, cy = gt["intrinsics"]
    names = [str(n) for n in gt["object_names"]]
    true_mask = gt["object_index"] == names.index(target)
    true_z = gt["depth"].astype(np.float64)
    r = dict(true_pixels=int(true_mask.sum()))
    r["calib"] = calib
    if "depth" not in out or "mask" not in out:
        r["error"] = "node produced no depth / mask (see calib)"
        return r
    z = image_to_array(out["depth"])
    mask = image_to_array(out["mask"]) > 0
    r["node_pixels"] = int(mask.sum())
    inter, union = mask & true_mask, mask | true_mask
    r["iou"] = float(inter.sum() / union.sum()) if union.any() else 0.0
    # which object the node actually segmented (by pixel vote)
    idx = gt["object_index"][mask]
    idx = idx[idx >= 0]
    r["segmented_as"] = names[int(np.bincount(idx).argmax())] if len(idx) else "background"
    valid = inter & (z > 0) & np.isfinite(z)
    r["compared_pixels"] = int(valid.sum())
    if valid.any():
        dz = (z[valid] - true_z[valid]) * 1000.0
        r["depth_mse_mm2"] = float(np.mean(dz ** 2))
        r["depth_rmse_mm"] = float(np.sqrt(r["depth_mse_mm2"]))
        r["depth_bias_mm"] = float(np.mean(dz))
        r["depth_mae_mm"] = float(np.mean(np.abs(dz)))
        p = backproject(z, valid, fx, fy, cx, cy)
        q = backproject(true_z, valid, fx, fy, cx, cy)
        e = np.linalg.norm(p - q, axis=1) * 1000.0
        r["point_mse_mm2"] = float(np.mean(e ** 2))
        r["point_rmse_mm"] = float(np.sqrt(r["point_mse_mm2"]))
    if true_mask.any():
        true_centroid = backproject(true_z, true_mask, fx, fy, cx, cy).mean(axis=0)
        if "object_cloud" in out:
            pts = point_cloud2.read_points_numpy(out["object_cloud"], field_names=("x", "y", "z"),
                                                 skip_nans=True)
            if len(pts):
                c = pts.astype(np.float64).mean(axis=0)
                r["centroid_error_mm"] = float(np.linalg.norm(c - true_centroid) * 1000.0)
                r["centroid_error_xyz_mm"] = ((c - true_centroid) * 1000.0).round(1).tolist()
                r["cloud_points"] = int(len(pts))
    scene = (z > 0) & np.isfinite(z) & (true_z > 0) & (true_z < 5.0)
    if scene.any():
        ds = (z[scene] - true_z[scene]) * 1000.0
        r["scene_depth_rmse_mm"] = float(np.sqrt(np.mean(ds ** 2)))
        r["scene_depth_bias_mm"] = float(np.mean(ds))
    r["_overlay"] = (mask, true_mask)
    return r


def overlay(path, mask, true_mask, title):
    img = np.full(mask.shape + (3,), 40, np.uint8)
    img[true_mask & ~mask] = (0, 0, 220)      # missed: red
    img[mask & ~true_mask] = (220, 120, 0)    # extra: blue
    img[mask & true_mask] = (0, 200, 0)       # correct: green
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(out_dir, args, results, objects):
    keys = ["iou", "depth_mse_mm2", "depth_rmse_mm", "depth_bias_mm", "point_rmse_mm",
            "centroid_error_mm", "scene_depth_rmse_mm"]
    L = [f"# VLA perception vs simulation ground truth",
         "", f"- Date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"- Camera: `{args.camera}` ({args.ns}), {args.runs} runs per object",
         "- Errors are node minus truth, on pixels both the node and the truth assign to the object.",
         "  MSE in mm², the rest in mm. IoU: node mask vs true object pixels.", "",
         "## Per object (mean over runs, ± std)", "",
         "| Object | Segmented as | IoU | Depth MSE (mm²) | Depth RMSE | Depth bias | 3D point RMSE | Centroid error | Scene depth RMSE |",
         "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for obj in objects:
        rs = [r for r in results if r["object"] == obj]
        row = {}
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r.get(k), float)]
            row[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None
        seg = sorted({r.get("segmented_as", "-") for r in rs})
        summary[obj] = row

        def cell(k, nd=1):
            v = row[k]
            if v is None:
                return "-"
            m, s, n = v
            return f"{m:.{nd}f} ± {s:.{nd}f}" + ("" if n == len(rs) else f" ({n}/{len(rs)})")
        L.append(f"| {obj} | {', '.join(seg)} | {cell('iou', 2)} | {cell('depth_mse_mm2', 0)} | "
                 f"{cell('depth_rmse_mm')} | {cell('depth_bias_mm')} | {cell('point_rmse_mm')} | "
                 f"{cell('centroid_error_mm')} | {cell('scene_depth_rmse_mm')} |")
    L += ["", "## Every run", "",
          "| Object | Run | ok | Segmented as | IoU | Depth MSE (mm²) | Depth RMSE | Bias | Centroid err | Centroid dx, dy, dz (mm) | Tags | Fit | Tag LOO (mm) | Flatness (mm) | Time (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.get("calib") or {}
        L.append(f"| {r['object']} | {r['run']} | {r.get('ok')} | {r.get('segmented_as', '-')} | "
                 f"{fmt(r.get('iou'), 2)} | {fmt(r.get('depth_mse_mm2'), 0)} | {fmt(r.get('depth_rmse_mm'))} | "
                 f"{fmt(r.get('depth_bias_mm'))} | {fmt(r.get('centroid_error_mm'))} | "
                 f"{r.get('centroid_error_xyz_mm', '-')} | {c.get('tags_seen', '-')} | {c.get('method', '-')} | "
                 f"{fmt(c.get('tag_loo_mm'))} | {fmt(c.get('flatness_rms_mm'))} | {fmt(r.get('seconds'))} |")
    checks = [(r["object"], r["run"], k, v) for r in results
              for k, v in ((r.get("calib") or {}).get("check_tags") or {}).items()]
    if checks:
        L += ["", "## Check tags (true height vs tag pose and vs depth)", "",
              "| Object run | Tag | True (mm) | From tag pose (mm) | From depth (mm) | Depth error (mm) |",
              "|---|---|---|---|---|---|"]
        for obj, run, k, v in checks:
            L.append(f"| {obj} #{run} | {k} | {v['true_m'] * 1000:.0f} | {v['from_tag_pose_m'] * 1000:.1f} | "
                     f"{v['from_depth_m'] * 1000:.1f} | {v['depth_err_mm']:+.1f} |")
    for r in results:
        if r.get("error"):
            L.append(f"\n- {r['object']} run {r['run']}: {r['error']}")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="left_gripper_camera", help="sensors.yaml stream name")
    ap.add_argument("--ns", default="/vla_perception", help="the node's namespace / name")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--objects", nargs="*", help="default: every object in the kitchen")
    ap.add_argument("--prompt", action="append", default=[], metavar="OBJECT=TEXT",
                    help="text to send instead of the object's name, e.g. cup='red cup'")
    ap.add_argument("--out", default=os.path.expanduser(
        f"~/sysmic-sim/vla_eval/{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = dict(p.split("=", 1) for p in args.prompt)

    rclpy.init()
    node = rclpy.create_node("vla_eval")
    collector = Collector(node, args.ns)
    pub = node.create_publisher(String, f"{args.ns}/object_name", 10)
    capture = node.create_client(Trigger, f"{args.ns}/capture")
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not capture.wait_for_service(timeout_sec=30):
        raise SystemExit(f"{args.ns}/capture not available: is vla_ros2_node.py running?")

    gt_path = os.path.join(args.out, "gt_latest.npz")
    gt = request_ground_truth(args.camera, gt_path)
    objects = args.objects or [str(n) for n in gt["object_names"]]
    print(f"objects: {', '.join(objects)}; output: {args.out}")
    time.sleep(1.0)  # let the node's subscriptions match ours

    results = []
    for obj in objects:
        text = prompts.get(obj, obj)
        for _ in range(3):
            pub.publish(String(data=text))
            time.sleep(0.3)
        for run in range(1, args.runs + 1):
            gt = request_ground_truth(args.camera, gt_path)
            np.savez_compressed(os.path.join(args.out, f"gt_{obj}_{run}.npz"), **gt)
            time.sleep(0.5)  # a fresh camera frame after the snapshot
            t0 = time.time()
            fut = capture.call_async(Trigger.Request())
            while not fut.done():
                time.sleep(0.05)
            resp = fut.result()
            got = collector.wait_after(t0)
            calib_msg = json.loads(got["calib"].data) if "calib" in got else {}
            r = dict(object=obj, prompt=text, run=run, ok=bool(resp.success), message=resp.message,
                     seconds=time.time() - t0)
            r.update(score(gt, obj, got if resp.success else {}, calib_msg.get("calib")))
            m = r.pop("_overlay", None)
            if m is not None and run == 1:
                overlay(os.path.join(args.out, f"mask_{obj}.png"), *m,
                        f"{obj}: IoU {r.get('iou', 0):.2f}  green=both red=missed blue=extra")
            results.append(r)
            print(f"{obj} run {run}: ok={r['ok']} iou={fmt(r.get('iou'), 2)} "
                  f"depth RMSE={fmt(r.get('depth_rmse_mm'))} mm bias={fmt(r.get('depth_bias_mm'))} mm "
                  f"centroid={fmt(r.get('centroid_error_mm'))} mm  [{r['message']}]", flush=True)
    os.remove(gt_path)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=str)
    write_report(args.out, args, results, objects)
    print(f"report: {os.path.join(args.out, 'report.md')}")
    ex.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
