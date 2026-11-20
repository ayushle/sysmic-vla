"""Link colours for SYSMIC 2.0. sysmic.urdf has no materials, so both the MuJoCo /
RoboCasa model (sysmic_to_mjcf.py) and RViz (launch/sim.launch.py) take them from here.
Needs only numpy, so the ROS launch file (system Python) can import it too.

Decals: parts of a mesh that get their own colour, e.g. the raised "SYSMIC"
lettering on the lift carriage. split_decals() writes two visual meshes for such a
link (decal faces, rest of the mesh) into the cache folder; the URDF meshes are
never touched and collisions keep using the original mesh.
"""
import os
import struct

import numpy as np

COLORS = (  # (substring of link name, rgba), first match wins
    ("jaw", "0.95 0.55 0.1 1"),
    ("arm_", "0.92 0.92 0.92 1"),
    ("caster", "0.12 0.12 0.12 1"),
    ("wheel", "0.08 0.08 0.08 1"),
    ("lift", "0.6 0.62 0.65 1"),
    ("base_link", "0.25 0.25 0.27 1"),
)
DEFAULT = "0.7 0.7 0.7 1"

# link -> decal: faces whose vertices all lie inside the box (link frame, metres).
# "SYSMIC" stands 1.5 mm proud of the lift's front plate (plate x = 0.2090 m,
# letter tops x = 0.2105 m), so the box starts just in front of the plate.
DECALS = {
    "lift_carriage_link": dict(
        name="sysmic_text", rgba="0.05 0.05 0.05 1",
        lo=(0.20905, -0.085, -0.105), hi=(0.2110, 0.085, -0.065)),
}

CACHE = os.environ.get("SYSMIC_SIM_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "sysmic_v2_sim")


def link_color(name):
    """rgba string for a URDF link name."""
    return next((c for key, c in COLORS if key in name), DEFAULT)


def _read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path}: only binary STL is supported for decals")
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
        [("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec


def _write_stl(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"sysmic decal split".ljust(80, b" "))
        f.write(struct.pack("<I", len(rec)))
        f.write(rec.tobytes())
    os.replace(tmp, path)


def split_decals(link, mesh_path):
    """[(mesh file, rgba), ...] visuals for a link: the mesh itself if it has no decal,
    else (rest of the mesh, link colour) and (decal faces, decal colour)."""
    decal = DECALS.get(link)
    if decal is None:
        return [(mesh_path, link_color(link))]
    out_dir = os.path.join(CACHE, "decals")
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    body_path = os.path.join(out_dir, f"{base}_body.stl")
    decal_path = os.path.join(out_dir, f"{base}_{decal['name']}.stl")
    src_time = max(os.path.getmtime(mesh_path), os.path.getmtime(__file__))
    if not all(os.path.exists(p) and os.path.getmtime(p) >= src_time
               for p in (body_path, decal_path)):
        rec = _read_stl(mesh_path)
        v = rec["v"].reshape(-1, 3, 3)
        inside = np.all((v >= np.array(decal["lo"])) & (v <= np.array(decal["hi"])), axis=(1, 2))
        if not inside.any():
            raise ValueError(f"decal '{decal['name']}' matches no faces of {mesh_path}")
        os.makedirs(out_dir, exist_ok=True)
        _write_stl(body_path, rec[~inside])
        _write_stl(decal_path, rec[inside])
    return [(body_path, link_color(link)), (decal_path, decal["rgba"])]
