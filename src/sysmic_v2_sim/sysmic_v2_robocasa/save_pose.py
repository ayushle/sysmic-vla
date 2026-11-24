#!/usr/bin/python3
"""Save the running simulation's robot pose (joints, odom, every TF frame) for vla_test.launch.py.

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 save_pose.py vla_test_pose.txt

Then rebuild is not needed (symlink install); the next vla_test.launch.py starts in it.
"""
import math, sys, time, datetime
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

out = sys.argv[1] if len(sys.argv) > 1 else "vla_test_pose.txt"
rclpy.init()
n = Node("pose_saver")
buf = Buffer(); TransformListener(buf, n)
state = {}
n.create_subscription(JointState, "/joint_states", lambda m: state.__setitem__("js", m), 10)
n.create_subscription(Odometry, "/odom", lambda m: state.__setitem__("odom", m), 10)
edges = {}
def on_tf(m):
    for t in m.transforms: edges[t.child_frame_id] = t
n.create_subscription(TFMessage, "/tf", on_tf, 100)
n.create_subscription(TFMessage, "/tf_static", on_tf, QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
end = time.time() + 3
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.1)
js, od = state["js"], state["odom"]
q = od.pose.pose.orientation
yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
f = lambda v: f"{v:.6f}"
L = [f"# SYSMIC robot pose saved {datetime.datetime.now():%Y-%m-%d %H:%M:%S} from the running simulation.",
     "# Read by sim/sysmic_sim_server.py --pose (vla_test.launch.py uses this file by default).",
     "# odom: base pose relative to where the robot started; joints: commandable joints (rad, lift in m).",
     "# tf: every frame relative to odom, and tf_edges: the parent -> child transforms as published.",
     "# Units: metres, radians; quaternions are [x, y, z, w].", "",
     "odom:", f"  x: {f(od.pose.pose.position.x)}", f"  y: {f(od.pose.pose.position.y)}", f"  yaw: {f(yaw)}", "",
     "joints:"]
cmd = [nm for nm in js.name if nm == "lift_joint" or nm.endswith("_pitch_joint")
       or (nm.startswith("arm_") and nm.endswith("_joint"))]  # + camera pitch servos
L += [f"  {nm}: {f(p)}" for nm, p in zip(js.name, js.position) if nm in cmd]
L += ["", "all_joint_states:"] + [f"  {nm}: {f(p)}" for nm, p in zip(js.name, js.position)]
L += ["", "tf:  # frame: {parent: odom, xyz, quat}"]
for fr in sorted(set(edges) | {e.header.frame_id for e in edges.values()}):
    if fr == "odom": continue
    try:
        t = buf.lookup_transform("odom", fr, Time()).transform
    except Exception as e:
        L.append(f"  # {fr}: not reachable from odom ({type(e).__name__})"); continue
    L.append(f"  {fr}: {{xyz: [{f(t.translation.x)}, {f(t.translation.y)}, {f(t.translation.z)}], "
             f"quat: [{f(t.rotation.x)}, {f(t.rotation.y)}, {f(t.rotation.z)}, {f(t.rotation.w)}]}}")
L += ["", "tf_edges:  # child: {parent, xyz, quat}"]
for ch in sorted(edges):
    t = edges[ch].transform
    L.append(f"  {ch}: {{parent: {edges[ch].header.frame_id}, xyz: [{f(t.translation.x)}, {f(t.translation.y)}, {f(t.translation.z)}], "
             f"quat: [{f(t.rotation.x)}, {f(t.rotation.y)}, {f(t.rotation.z)}, {f(t.rotation.w)}]}}")
open(out, "w").write("\n".join(L) + "\n")
print(len(edges), "tf edges saved to", out)
