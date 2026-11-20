"""Where things live. Shared by the simulation scripts."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_urdf():
    """Locate sysmic.urdf: the package's own urdf/ folder.

    Order: $SYSMIC_URDF (set by the launch file), the package's urdf/ (source tree,
    also reached through --symlink-install links, or share/sysmic_v2_sim/urdf when
    installed), then a sysmicV2-urdf checkout next to the package.
    """
    env = os.environ.get("SYSMIC_URDF")
    if env:
        return os.path.abspath(env)
    for here in (os.path.dirname(os.path.realpath(__file__)), HERE):
        for rel in (("..", "urdf"), ("..", "..", "sysmicV2-urdf")):
            c = os.path.join(here, *rel, "sysmic.urdf")
            if os.path.isfile(c):
                return os.path.normpath(c)
    raise FileNotFoundError("sysmic.urdf not found. Expected sysmic_v2_sim/urdf/sysmic.urdf, "
                            "or set SYSMIC_URDF=/path/to/sysmic.urdf")


# Robot description: sysmic.urdf + meshes/ (mesh paths are relative to its folder).
# Never modified by the scripts.
URDF = _find_urdf()
DESCRIPTION = os.path.dirname(URDF)

# Standalone RoboCasa folder (robot classes + exported model files)
ROBOCASA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "sysmic_v2_robocasa"))
if not os.path.isdir(ROBOCASA_DIR):  # installed (non-symlink) layout
    ROBOCASA_DIR = os.path.normpath(os.path.join(HERE, "..", "sysmic_v2_robocasa"))

# Generated MuJoCo / robosuite models. Rebuilt automatically when the URDF or
# the scripts change, so this folder can be deleted at any time.
GENERATED = os.environ.get("SYSMIC_SIM_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "sysmic_v2_sim")
ROBOT_XML = os.path.join(GENERATED, "sysmic_v2.xml")
SCENE_XML = os.path.join(GENERATED, "sysmic_v2_scene.xml")
ROBOSUITE_DIR = os.path.join(GENERATED, "robosuite_sysmic")
