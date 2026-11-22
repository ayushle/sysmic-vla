"""VLA perception pipeline v2: one webcam -> undistorted image -> metric depth -> object + scene clouds.

Changes from vla-complete-pipeline.py, all chosen with the synthetic depth bench
(~/grasp_pipeline/depth_bench):

  Camera   real webcam via V4L2 at the calibration resolution (MJPG 1920x1080), small buffer,
           stale frames flushed. K is rescaled if the camera delivers another size with the same aspect.
  Lens     the frame is undistorted ONCE at input (precomputed remap, alpha=0 so no black border).
           Everything downstream (tags, depth model, SAM, rays, grasp projection) uses the
           undistorted frame with K_new and zero distortion. The 3D camera frame is unchanged.
  Tags     AprilTag 36h11 only. Table tags (TABLE_TAG_IDS) calibrate; check tags (CHECK_TAG_IDS,
           on blocks of known height) only VERIFY. Corners are pushed out by TAG_EDGE_BIAS_PX
           before PnP, which removes the "tag looks smaller -> too far" bias (1-3 % in the bench).
  Depth    Depth Anything V2 relative (vitb), affine fit in disparity: raw = a / Z + b.
           Step 1: fit on the table tags. Step 2 (dense): the table plane from the tag poses gives
           the true Z of every table pixel, so (a, b) is refit on thousands of table pixels
           (objects are gated out, robust trimming). Falls back to step 1 if step 2 is not possible.
  Checks   every call reports: tag LOO error, table flatness (RMS about the tag plane) and the
           height error of each check tag. These are the numbers to watch on the real setup.

Mechanical setup this assumes (see the README block at the end of this docstring):
  * one rigid camera, 40-45 deg from vertical, 0.6-0.7 m from the workspace centre
  * 4 table tags flat on the table, at the near/far, left/right corners of the workspace
  * 2 check tags on blocks of known height (e.g. 5 cm and 12 cm), inside the view
  * tags 35-40 mm (black square), printed, measured with a ruler, set in TAG_SIZE_METERS

Run:
  python vla-pipeline-v2.py --object cup                 live view; 'c' = capture + cloud, 's' = save, 'q' = quit
  python vla-pipeline-v2.py --object cup --once          one frame, save outputs, exit
  python vla-pipeline-v2.py --calib-tag 0 --range 0.652  estimate TAG_EDGE_BIAS_PX from a tag whose
                                                         camera-to-tag-centre distance you measured
"""
import argparse
import gc
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R


# ======================================================================================
# Configuration (edit here)
# ======================================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*paths):
    """First of these paths that exists (None entries skipped); the last one if none does."""
    paths = [p for p in paths if p]
    return next((p for p in paths if os.path.exists(p)), paths[-1])


CALIB_SIZE = (1920, 1080)                 # (w, h) the intrinsics below were calibrated at
CAMERA_MATRIX = np.array([[1.47292993e+03, 0.0, 9.70781376e+02],
                          [0.0, 1.47094362e+03, 5.46826145e+02],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS = np.array([[-1.72672604e-02, 4.29628548e-01, 7.90175149e-04,
                         9.61346143e-04, -1.01309129e+00]], dtype=np.float64)

TAG_SIZE_METERS = 0.035                   # side of the BLACK square, measured on the print
TAG_EDGE_BIAS_PX = 0.0                    # from --calib-tag (bench value for a blurred, gamma'd image: ~0.7)
TABLE_TAG_IDS = [0, 1, 2, 3]              # flat on the table: used for the fit and the plane
CHECK_TAG_IDS = {4: 0.05, 5: 0.12}        # id -> true height of the tag surface above the table (m)


# ======================================================================================
# Camera
# ======================================================================================
def intrinsics_for(w, h, K=CAMERA_MATRIX, calib_size=CALIB_SIZE):
    """K for a frame of size (w, h). Distortion coefficients do not depend on resolution."""
    cw, ch = calib_size
    if (w, h) == (cw, ch):
        return K.copy()
    if abs(w / h - cw / ch) > 1e-3:
        raise ValueError(f"frame {w}x{h} has another aspect ratio than the calibration {cw}x{ch}; recalibrate")
    s = w / cw
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s
    K2[0, 2] = (K[0, 2] + 0.5) * s - 0.5          # pixel-centre convention
    K2[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K2


class Undistorter:
    """Precomputed remap: raw frame -> undistorted frame with K_new and zero distortion."""

    def __init__(self, K, dist, size, alpha=0.0):
        w, h = size
        self.K_raw, self.dist_raw, self.size = K, dist, size
        self.K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        self.dist = np.zeros((1, 5))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, self.K, (w, h), cv2.CV_16SC2)

    def __call__(self, frame):
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


class img_input:
    """Real webcam(s) via V4L2. publishimg() returns the RAW frame; undistort() gives the frame + K to use."""

    def __init__(self, cam_name="cam", port="/dev/video0", size=CALIB_SIZE, fourcc="MJPG"):
        self.cam_name = cam_name.split(",")
        self.port = port.split(",")
        self.dev_dict = dict(zip(self.cam_name, self.port))
        self.size, self.fourcc = size, fourcc
        self.caps = {}
        self._und = {}
        self.CAMERA_MATRIX = CAMERA_MATRIX.copy()      # kept for compatibility with older callers
        self.DIST_COEFFS = DIST_COEFFS.copy()

    def init_cam(self, name):
        if name in self.caps:
            return
        dev = self.dev_dict[name]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.release_all()
            raise SystemExit(f"Error: could not open {name} at {dev}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))   # 1080p needs MJPG on USB2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)           # ignored by fixed-focus cameras; focus changes K
        for _ in range(10):                          # let auto exposure settle
            cap.read()
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != tuple(self.size):
            print(f"[cam] WARNING: {name} gives {w}x{h}, asked {self.size[0]}x{self.size[1]}; K will be rescaled")
        print(f"[cam] {name} at {dev}: {w}x{h}")
        self.caps[name] = cap

    def publishimg(self, name, flush=2):
        """Newest RAW frame (BGR). Grabs a few frames first so a buffered stale one is not returned."""
        cap = self.caps.get(name)
        if cap is None or not cap.isOpened():
            return None
        for _ in range(flush):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            print(f"Error: failed to grab a frame from {name}")
            return None
        return frame

    def undistort(self, frame):
        """(undistorted frame, K_new, zero dist). Maps are built once per frame size."""
        h, w = frame.shape[:2]
        if (w, h) not in self._und:
            self._und[(w, h)] = Undistorter(intrinsics_for(w, h), DIST_COEFFS, (w, h))
        u = self._und[(w, h)]
        return u(frame), u.K.copy(), u.dist.copy()

    def release_cam(self, name):
        cap = self.caps.pop(name, None)
        if cap is not None:
            cap.release()

    def release_all(self):
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()


class txt_inp:
    def __init__(self):
        self.item = []

    def add_item(self, item):
        self.item.append(item + ".")

    def get_item(self):
        return self.item.pop(0)


def rvec_to_rpy_deg(rvec):
    rot_matrix, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot_matrix).as_euler("xyz", degrees=True)


# ======================================================================================
# Depth + tags + segmentation + clouds
# ======================================================================================
class depth_estimation:
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, low_vram=False):
        self.ENCODER = "vitb"
        self.WEIGHTS_PATH = _first_existing(
            os.environ.get("VLA_DEPTH_WEIGHTS"),
            os.path.join(_HERE, "models", "depth_anything_v2_vitb.pth"),       # setup_vla.sh
            os.path.expanduser("~/cnn/depth/depth_anything_v2_vitb.pth"))
        self.REPO_PATH = _first_existing(
            os.environ.get("VLA_DEPTH_REPO"),
            os.path.join(_HERE, "third_party", "Depth-Anything-V2"),         # setup_vla.sh
            os.path.expanduser("~/cnn/depth/Depth-Anything-V2"))
        self.INPUT_SIZE = 518
        self.FIT_SPACE = "disparity"          # Depth Anything V2 relative is affine in 1/Z
        self.MAX_DEPTH_M = 2.5
        self.MIN_X_SPREAD = 0.03              # min spread of 1/Z (relative) for a 2-parameter fit
        self.DEDUPE_PX = 15
        self.PCD_STRIDE = 1
        self.TAG_SIZE_METERS = TAG_SIZE_METERS
        self.TAG_EDGE_BIAS_PX = TAG_EDGE_BIAS_PX
        self.TABLE_TAG_IDS = list(TABLE_TAG_IDS) if TABLE_TAG_IDS is not None else None
        self.CHECK_TAG_IDS = dict(CHECK_TAG_IDS)
        # dense plane-anchored refit
        self.DENSE_FIT = True
        self.DENSE_STRIDE = 4                 # pixel stride for the table samples
        self.DENSE_GATE_M = 0.015             # pixel counts as table if the tag fit puts it within this of the plane
        self.DENSE_HULL_GROW = 1.25           # table region = hull of the table tags, grown about its centre
        self.DENSE_MIN_PX = 2000
        # flying pixels: depth at object silhouettes is blended between object and background and
        # smears points along the rays (mug diameter 14.9 cm instead of 7.0 in the bench without this)
        self.OBJ_ERODE_PX = 3                 # erode the SAM mask before taking object points
        self.FLYING_REL_GRAD = 0.004          # drop pixels whose |grad Z| / Z per pixel exceeds this
        self.PLANE_MAX_ANGLE_DEG = 10.0
        self.PLANE_MAX_OFFSET_M = 0.01

        self.CAMERA_MATRIX = None
        self.DIST_COEFFS = None
        self.detections = []
        self.last_calib = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_FP16 = True
        self.low_vram = low_vram
        self.model = None
        self.dino = self.dino_proc = None
        self.sam = self.sam_proc = None

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
        self._ray_cache = {}
        sys.path.append(self.REPO_PATH)

    @property
    def OBJ_POINTS(self):
        h = self.TAG_SIZE_METERS / 2.0
        return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)

    # ---------------------------------------------------------------- models
    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=(self.device == "cuda" and self.USE_FP16))

    def _free_gpu(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def depth_model_init(self):
        if self.model is not None:
            return
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            sys.exit("Error: could not import depth_anything_v2. Check REPO_PATH.")
        print(f"Loading Depth Anything V2 ({self.ENCODER}) on {self.device}...")
        model = DepthAnythingV2(**self.MODEL_CONFIGS[self.ENCODER])
        model.load_state_dict(torch.load(self.WEIGHTS_PATH, map_location="cpu"))
        self.model = model.to(self.device).eval()

    def depth_model_unload(self):
        self.model = None
        self._free_gpu()

    def seg_model_init(self):
        if self.dino is not None and self.sam is not None:
            return
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                                  SamProcessor, SamModel)
        print(f"Loading Grounding DINO + SAM on {self.device}...")
        dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).eval()
        sam_id = "facebook/sam-vit-base"
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    def seg_model_unload(self):
        self.dino = self.dino_proc = None
        self.sam = self.sam_proc = None
        self._free_gpu()

    # ---------------------------------------------------------------- tags
    def _set_camera(self, K, dist):
        K = np.asarray(K, np.float64)
        dist = np.zeros((1, 5)) if dist is None else np.asarray(dist, np.float64)
        if not (np.array_equal(K, self.CAMERA_MATRIX) and np.array_equal(dist, self.DIST_COEFFS)):
            self._ray_cache.clear()
        self.CAMERA_MATRIX, self.DIST_COEFFS = K, dist

    def marker_pose(self, frame, CAMERA_MATRIX, DIST_COEFFS):
        """AprilTag 36h11 poses. Corners are expanded by TAG_EDGE_BIAS_PX before PnP."""
        self._set_camera(CAMERA_MATRIX, DIST_COEFFS)
        self.detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return
        for c, marker_id in zip(corners, ids.ravel()):
            img_points = c.reshape(4, 2).astype(np.float64)
            pnp_points = self._expand_corners(img_points, self.TAG_EDGE_BIAS_PX)
            ok, rvec, tvec = cv2.solvePnP(self.OBJ_POINTS, pnp_points, self.CAMERA_MATRIX, self.DIST_COEFFS,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or tvec[2, 0] <= 0:
                continue
            Rm, _ = cv2.Rodrigues(rvec)
            self.detections.append({
                "dictionary": "APRILTAG_36h11",
                "id": int(marker_id),
                "corners_px": img_points.tolist(),
                "xyz_m": [float(v) for v in tvec.ravel()],
                "rpy_deg": [float(v) for v in rvec_to_rpy_deg(rvec)],
                "normal": Rm[:, 2].tolist(),
            })

    @staticmethod
    def _expand_corners(c, bias_px):
        """Move each corner away from the tag centre: each edge moves out by bias_px."""
        if not bias_px:
            return c
        ctr = c.mean(0)
        v = c - ctr
        return c + v / np.linalg.norm(v, axis=1, keepdims=True) * bias_px * np.sqrt(2)

    def _unique_tags(self):
        kept = []
        for d in self.detections:
            corners = np.array(d["corners_px"], dtype=np.float32)
            centre = corners.mean(axis=0)
            if any(np.linalg.norm(centre - k["centre"]) < self.DEDUPE_PX for k in kept):
                continue
            kept.append({"id": d["id"], "corners": corners, "centre": centre, "z": float(d["xyz_m"][2]),
                         "xyz": np.array(d["xyz_m"]), "normal": np.array(d["normal"])})
        return kept

    def table_plane(self, tags):
        """Plane (n, d) from table-tag poses, n toward the camera, height = p.n + d. None if inconsistent."""
        if len(tags) < 2:
            return None, "fewer than 2 table tags"
        P = np.array([t["xyz"] for t in tags])
        N = np.array([t["normal"] if np.dot(t["normal"], -t["xyz"]) > 0 else -t["normal"] for t in tags])
        n = N.mean(0)
        n /= np.linalg.norm(n)
        ang = np.degrees(np.arccos(np.clip(N @ n, -1, 1))).max()
        d = float(-np.mean(P @ n))
        off = np.abs(P @ n + d).max()
        if ang > self.PLANE_MAX_ANGLE_DEG or off > self.PLANE_MAX_OFFSET_M:
            return None, f"table tags disagree (normals {ang:.1f} deg apart, offset {off * 100:.1f} cm)"
        if len(P) >= 3:                                  # tag centres give a better normal than tag rotations
            c = P.mean(0)
            _, _, vt = np.linalg.svd(P - c)
            n3 = vt[-1] if vt[-1] @ n > 0 else -vt[-1]
            if np.degrees(np.arccos(np.clip(n3 @ n, -1, 1))) < self.PLANE_MAX_ANGLE_DEG:
                n, d = n3, float(-np.mean(P @ n3))
        return (n, d), "ok"

    # ---------------------------------------------------------------- depth
    def depth_model_output(self, frame):
        self.depth_model_init()
        try:
            with torch.inference_mode(), self._amp():
                raw = self.model.infer_image(frame, self.INPUT_SIZE)
            return np.asarray(raw).astype(np.float32)
        finally:
            if self.low_vram:
                self.depth_model_unload()

    @staticmethod
    def sample_raw(raw, corners):
        mask = np.zeros(raw.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
        vals = raw[mask > 0]
        return float(np.median(vals)) if vals.size else None

    def to_x(self, z):
        z = np.asarray(z, dtype=np.float64)
        return 1.0 / z if self.FIT_SPACE == "disparity" else z

    def fit_affine(self, raws, zs):
        x = self.to_x(zs)
        A = np.stack([x, np.ones_like(x)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, np.asarray(raws, dtype=np.float64), rcond=None)
        return float(a), float(b)

    def predict_z(self, raw, a, b):
        x = (np.asarray(raw, dtype=np.float64) - b) / a
        z = 1.0 / np.maximum(x, 1.0 / self.MAX_DEPTH_M) if self.FIT_SPACE == "disparity" else x
        return np.clip(z, 0.0, self.MAX_DEPTH_M)

    def _spread_ok(self, zs):
        x = self.to_x(zs)
        return np.ptp(x) / np.mean(np.abs(x)) >= self.MIN_X_SPREAD

    def _dense_fit(self, raw, tags, plane, a0, b0):
        """Refit (a, b) on table pixels whose true Z comes from the tag plane."""
        n, d = plane
        h, w = raw.shape
        hull = cv2.convexHull(np.concatenate([t["corners"] for t in tags]).astype(np.float32))
        ctr = hull.reshape(-1, 2).mean(0)
        hull = ((hull.reshape(-1, 2) - ctr) * self.DENSE_HULL_GROW + ctr).astype(np.int32)
        region = np.zeros((h, w), np.uint8)
        cv2.fillPoly(region, [hull], 1)
        s = self.DENSE_STRIDE
        vv, uu = np.mgrid[0:h:s, 0:w:s]
        sel = region[vv, uu] > 0
        vv, uu = vv[sel], uu[sel]
        rays = self.get_rays(h, w)[vv, uu]
        r3 = np.c_[rays, np.ones(len(rays))]
        denom = r3 @ n
        ok = denom < -1e-6
        Zp = np.full(len(r3), np.nan)
        Zp[ok] = -d / denom[ok]                         # where each ray meets the table plane
        rv = raw[vv, uu].astype(np.float64)
        Zt = self.predict_z(rv, a0, b0)
        table = ok & (Zp > 0.05) & (Zp < self.MAX_DEPTH_M) & (np.abs(Zt - Zp) < self.DENSE_GATE_M)
        if table.sum() < self.DENSE_MIN_PX:
            return None, f"only {int(table.sum())} table pixels"
        Zp, rv = Zp[table], rv[table]
        q = np.percentile(1 / Zp, [5, 95])
        if (q[1] - q[0]) / np.mean(1 / Zp) < self.MIN_X_SPREAD:
            return None, "table pixels at nearly the same Z (camera too close to straight down)"
        A = np.c_[1.0 / Zp, np.ones(len(Zp))]
        keep = np.ones(len(rv), bool)
        for _ in range(3):                               # robust: trim outliers (other objects, shadows)
            (a, b), *_ = np.linalg.lstsq(A[keep], rv[keep], rcond=None)
            res = rv - A @ np.array([a, b])
            mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
            keep = np.abs(res) < 3 * mad
        if a <= 0:
            return None, "dense fit gave a <= 0"
        # flatness: RMS height of the kept table pixels about the tag plane, with the new fit
        Zf = self.predict_z(rv[keep], a, b)
        pts = Zf[:, None] * np.c_[rays[table][keep], np.ones(keep.sum())]
        flat = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
        return (float(a), float(b)), dict(n_px=int(keep.sum()), flatness_rms_mm=flat * 1000)

    def metric_depth_mapping(self, frame, CAMERA_MATRIX, DIST_COEFFS):
        """Metric depth for `frame`. Pass the UNDISTORTED frame with K_new and zero dist
        (scene_clouds does this), or a raw frame with its distortion: both are handled.
        Returns (z_map, depth_mm_uint16) or None. Diagnostics in self.last_calib."""
        self.marker_pose(frame, CAMERA_MATRIX, DIST_COEFFS)
        tags = self._unique_tags()
        cal = dict(tags_seen=[t["id"] for t in tags])
        self.last_calib = cal
        if self.TABLE_TAG_IDS is None:
            table = [t for t in tags if t["id"] not in self.CHECK_TAG_IDS]
        else:
            table = [t for t in tags if t["id"] in self.TABLE_TAG_IDS]
        checks = [t for t in tags if t["id"] in self.CHECK_TAG_IDS]
        fit_tags = table
        if len(table) < 2 or not self._spread_ok([t["z"] for t in table]):
            fit_tags = table + checks                     # fallback: use the check tags in the fit too
            cal["warning"] = "table tags alone cannot fit; check tags used in the fit (no independent check)"
            if len(fit_tags) < 2 or not self._spread_ok([t["z"] for t in fit_tags]):
                print(f"[SKIP] cannot calibrate: tags {cal['tags_seen']} (need 2+ at different distances; "
                      f"tilt the camera 40-45 deg)")
                return None

        raw = self.depth_model_output(frame)
        r = np.array([self.sample_raw(raw, t["corners"]) for t in fit_tags])
        zs = np.array([t["z"] for t in fit_tags])
        a, b = self.fit_affine(r, zs)
        if a <= 0:
            print(f"[SKIP] fitted scale a={a:.4f} <= 0 (model ordering disagrees with tags)")
            return None
        cal.update(method="tags", a=a, b=b, fit_tags=[t["id"] for t in fit_tags])
        errs = []
        for k in range(len(fit_tags)) if len(fit_tags) >= 3 else []:
            keep = np.arange(len(fit_tags)) != k
            if self._spread_ok(zs[keep]):
                ak, bk = self.fit_affine(r[keep], zs[keep])
                if ak > 0:
                    errs.append(abs(float(self.predict_z(r[k], ak, bk)) - zs[k]))
        cal["tag_loo_mm"] = float(np.mean(errs) * 1000) if errs else float("nan")

        plane, why = self.table_plane(table)
        cal["plane"] = None if plane is None else dict(n=plane[0].tolist(), d=plane[1])
        if plane is None:
            cal["plane_status"] = why
        elif self.DENSE_FIT:
            res, info = self._dense_fit(raw, table, plane, a, b)
            if res is None:
                cal["dense_status"] = info
            else:
                a, b = res
                cal.update(method="dense table plane", a=a, b=b, **info)

        z_map = self.predict_z(raw, a, b).astype(np.float32)

        if plane is not None:                            # independent check on the tags on blocks
            n, d = plane
            h, w = z_map.shape
            rays = self.get_rays(h, w)
            out = {}
            for t in checks:
                u, v = np.clip(np.round(t["centre"]).astype(int), 0, [w - 1, h - 1])
                z = self.sample_raw(z_map, t["corners"])
                p = z * np.array([rays[v, u, 0], rays[v, u, 1], 1.0])
                h_depth = float(p @ n + d)
                h_tag = float(t["xyz"] @ n + d)
                out[t["id"]] = dict(true_m=self.CHECK_TAG_IDS[t["id"]], from_tag_pose_m=h_tag,
                                    from_depth_m=h_depth, depth_err_mm=(h_depth - self.CHECK_TAG_IDS[t["id"]]) * 1000)
            cal["check_tags"] = out

        self._print_calib(cal)
        return z_map, np.clip(z_map * 1000.0, 0, 65535).astype(np.uint16)

    @staticmethod
    def _print_calib(cal):
        s = f"[OK] {cal['method']} | a={cal['a']:.4f} b={cal['b']:.4f} | tags {cal['tags_seen']}"
        if not np.isnan(cal.get("tag_loo_mm", np.nan)):
            s += f" | tag LOO {cal['tag_loo_mm']:.1f} mm"
        if "flatness_rms_mm" in cal:
            s += f" | table flatness {cal['flatness_rms_mm']:.1f} mm RMS ({cal['n_px']} px)"
        print(s)
        for i, c in cal.get("check_tags", {}).items():
            print(f"     check tag {i}: true {c['true_m'] * 100:.1f} cm | tag pose {c['from_tag_pose_m'] * 100:.1f} cm"
                  f" | depth {c['from_depth_m'] * 100:.1f} cm ({c['depth_err_mm']:+.0f} mm)")
        for k in ("warning", "plane_status", "dense_status"):
            if k in cal:
                print(f"     {k}: {cal[k]}")

    # ---------------------------------------------------------------- segmentation
    def mask_generator(self, frame, object_name):
        self.seg_model_init()
        try:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = object_name.lower().strip()
            if not text.endswith("."):
                text += "."
            inputs = self.dino_proc(images=image_pil, text=text, return_tensors="pt").to(self.device)
            with torch.inference_mode(), self._amp():
                outputs = self.dino(**inputs)
            results = self.dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25,
                target_sizes=[image_pil.size[::-1]])[0]
            if len(results["boxes"]) == 0:
                print(f"Failed to find any object matching '{object_name}'.")
                return None
            order = results["scores"].argsort(descending=True)
            boxes = results["boxes"][order]
            print(f"Success: Found {len(boxes)} instance(s).")
            best_box = boxes[0].float().cpu().tolist()
            sam_inputs = self.sam_proc(image_pil, input_boxes=[[best_box]], return_tensors="pt").to(self.device)
            with torch.inference_mode(), self._amp():
                sam_out = self.sam(**sam_inputs)
            masks = self.sam_proc.image_processor.post_process_masks(
                sam_out.pred_masks.float().cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu())[0]
            best_idx = sam_out.iou_scores[0, 0].argmax().item()
            return masks[0, best_idx].numpy().astype(np.uint8) * 255
        finally:
            if self.low_vram:
                self.seg_model_unload()

    # ---------------------------------------------------------------- clouds
    def get_rays(self, h, w):
        """Normalised coords (x/z, y/z) per pixel for the current K / dist (identity undistortion if dist = 0)."""
        if (h, w) not in self._ray_cache:
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            pix = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
            n = cv2.undistortPoints(pix, self.CAMERA_MATRIX, self.DIST_COEFFS).reshape(h, w, 2)
            self._ray_cache[(h, w)] = n.astype(np.float32)
        return self._ray_cache[(h, w)]

    def scene_clouds(self, frame_raw, CAMERA_MATRIX=None, DIST_COEFFS=None, object_name="cup", stride=None,
                     undistort=True):
        """RAW webcam frame in, everything the grasp stage needs out.

        The frame is undistorted first; all outputs refer to the undistorted image:
            frame      (H, W, 3) uint8   undistorted BGR (draw overlays / run Molmo on THIS)
            K, dist    K_new and zeros   (use these for any projection into `frame`)
            z_map      (H, W) float32    metric depth (m)
            mask       (H, W) uint8      object mask 0 / 255
            obj_pts / obj_pix / obj_bgr, scene_pts / scene_pix   (camera frame, metres)
            plane      {"n", "d"} table plane from the tags (height = p.n + d), or None
            detections tag detections (same format as the recorded JSONs)
            calib      calibration diagnostics
        3D points are in the physical camera frame; undistortion does not move it.
        """
        h0, w0 = frame_raw.shape[:2]
        K0 = intrinsics_for(w0, h0) if CAMERA_MATRIX is None else np.asarray(CAMERA_MATRIX, np.float64)
        D0 = DIST_COEFFS if DIST_COEFFS is not None else DIST_COEFFS_DEFAULT
        if undistort:
            und = self._undistorter(K0, D0, (w0, h0))
            frame, K, dist = und(frame_raw), und.K, und.dist
        else:
            frame, K, dist = frame_raw, K0, np.asarray(D0, np.float64)

        result = self.metric_depth_mapping(frame, K, dist)
        if result is None:
            return None
        z_map, _ = result
        mask = self.mask_generator(frame, object_name)
        if mask is None:
            return None

        stride = stride or self.PCD_STRIDE
        h, w = z_map.shape
        rays = self.get_rays(h, w)
        solid = self.solid_pixels(z_map)
        obj_mask = mask > 0
        if self.OBJ_ERODE_PX:
            k = 2 * self.OBJ_ERODE_PX + 1
            obj_mask = cv2.erode(obj_mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = z_map[vv, uu]
        valid = (z > 0.01) & (z < self.MAX_DEPTH_M - 1e-3) & solid[vv, uu]
        pts = np.stack([rays[vv, uu, 0] * z, rays[vv, uu, 1] * z, z], axis=-1).astype(np.float32)
        pix = np.stack([uu, vv], axis=-1).astype(np.int32)
        obj = valid & obj_mask[vv, uu]
        if not obj.any():
            print("No valid depth points inside the mask.")
            return None
        return {
            "frame": frame, "K": K, "dist": dist,
            "z_map": z_map, "mask": mask, "solid": solid,
            "obj_pts": pts[obj], "obj_pix": pix[obj], "obj_bgr": frame[vv, uu][obj],
            "scene_pts": pts[valid], "scene_pix": pix[valid],
            "plane": self.last_calib.get("plane"),
            "detections": list(self.detections),
            "calib": self.last_calib,
        }

    def solid_pixels(self, z_map):
        """False at depth discontinuities (flying pixels): relative depth gradient per pixel too large."""
        if not self.FLYING_REL_GRAD:
            return np.ones(z_map.shape, bool)
        gx = cv2.Sobel(z_map, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(z_map, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        return np.hypot(gx, gy) / np.maximum(z_map, 1e-3) < self.FLYING_REL_GRAD

    def _undistorter(self, K, dist, size):
        key = (size, np.asarray(K).tobytes(), np.asarray(dist).tobytes())
        if getattr(self, "_und_key", None) != key:
            self._und, self._und_key = Undistorter(K, dist, size), key
        return self._und

    def pcd_generator(self, frame_raw, CAMERA_MATRIX=None, DIST_COEFFS=None, object_name="cup"):
        """Bytes of a coloured binary .PCD of the masked object (camera frame), or None."""
        sc = self.scene_clouds(frame_raw, CAMERA_MATRIX, DIST_COEFFS, object_name)
        return None if sc is None else pcd_bytes(sc["obj_pts"], sc["obj_bgr"])


DIST_COEFFS_DEFAULT = DIST_COEFFS


def pcd_bytes(pts, bgr):
    b, g, r = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    arr = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["rgb"] = packed.view(np.float32)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {len(arr)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(arr)}\nDATA binary\n")
    return header.encode("ascii") + arr.tobytes()


# ======================================================================================
# Tag edge-bias calibration (TAG_EDGE_BIAS_PX)
# ======================================================================================
def calibrate_edge_bias(frames, de, tag_id, true_range_m):
    """Find the corner expansion (px) that makes the tag's PnP distance match a measured distance.
    true_range_m: straight-line distance camera (lens centre) -> tag centre, measured with a ruler."""
    def mean_range(bias):
        de.TAG_EDGE_BIAS_PX = bias
        rs = []
        for f, K, D in frames:
            de.marker_pose(f, K, D)
            rs += [np.linalg.norm(d["xyz_m"]) for d in de.detections if d["id"] == tag_id]
        return float(np.mean(rs)) if rs else None

    r0 = mean_range(0.0)
    if r0 is None:
        raise SystemExit(f"tag {tag_id} not seen")
    lo, hi = -3.0, 3.0                       # range decreases as the corners move out
    for _ in range(40):
        mid = (lo + hi) / 2
        if mean_range(mid) > true_range_m:
            lo = mid
        else:
            hi = mid
    bias = (lo + hi) / 2
    print(f"tag {tag_id}: PnP range {r0 * 1000:.1f} mm without correction, measured {true_range_m * 1000:.1f} mm "
          f"({(r0 / true_range_m - 1) * 100:+.2f} %)")
    print(f"-> set TAG_EDGE_BIAS_PX = {bias:.3f}   (range with it: {mean_range(bias) * 1000:.1f} mm)")
    return bias


# ======================================================================================
# Live loop
# ======================================================================================
def draw_status(frame, de, sc=None):
    vis = frame.copy()
    for d in de.detections:
        c = np.array(d["corners_px"], np.int32)
        col = (42, 120, 214) if d["id"] in de.CHECK_TAG_IDS else (214, 120, 42)
        cv2.polylines(vis, [c], True, col, 3)
        cv2.putText(vis, f"{d['id']} z={d['xyz_m'][2]:.3f}", tuple(c[0] + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if sc is not None:
        m = sc["mask"] > 0
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([60, 200, 60])).astype(np.uint8)
    cal = de.last_calib
    lines = [f"tags {cal.get('tags_seen', [])}  method: {cal.get('method', '-')}"]
    if "flatness_rms_mm" in cal:
        lines.append(f"table flatness {cal['flatness_rms_mm']:.1f} mm   tag LOO {cal.get('tag_loo_mm', float('nan')):.1f} mm")
    for i, c in cal.get("check_tags", {}).items():
        lines.append(f"check tag {i}: {c['depth_err_mm']:+.0f} mm")
    for k, s in enumerate(lines):
        cv2.putText(vis, s, (20, 40 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
        cv2.putText(vis, s, (20, 40 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return vis


def save_outputs(sc, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, stem)
    cv2.imwrite(p + "_undistorted.png", sc["frame"])
    cv2.imwrite(p + "_mask.png", sc["mask"])
    cv2.imwrite(p + "_depth_mm.png", np.clip(sc["z_map"] * 1000, 0, 65535).astype(np.uint16))
    with open(p + "_object.pcd", "wb") as f:
        f.write(pcd_bytes(sc["obj_pts"], sc["obj_bgr"]))
    meta = dict(K=sc["K"].tolist(), dist=sc["dist"].tolist(), plane=sc["plane"], calib=sc["calib"],
                detections=sc["detections"], timestamp=time.time())
    with open(p + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(f"saved {p}_*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", default="/dev/video0")
    ap.add_argument("--object", default="cup")
    ap.add_argument("--once", action="store_true", help="capture one frame, process, save, exit")
    ap.add_argument("--out", default=os.path.expanduser("~/sysmic-vla/captures"))
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--calib-tag", type=int, help="tag id for TAG_EDGE_BIAS_PX calibration")
    ap.add_argument("--range", type=float, help="measured camera->tag-centre distance (m) for --calib-tag")
    args = ap.parse_args()

    cams = img_input("cam", args.cam)
    cams.init_cam("cam")
    de = depth_estimation(low_vram=args.low_vram)
    try:
        if args.calib_tag is not None:
            if args.range is None:
                raise SystemExit("--calib-tag needs --range")
            frames = []
            for _ in range(15):
                f, K, D = cams.undistort(cams.publishimg("cam"))
                frames.append((f, K, D))
            calibrate_edge_bias(frames, de, args.calib_tag, args.range)
            return

        if args.once:
            raw = cams.publishimg("cam")
            sc = de.scene_clouds(raw, object_name=args.object)
            if sc is not None:
                save_outputs(sc, args.out, time.strftime("%Y%m%d_%H%M%S"))
                cv2.imwrite(os.path.join(args.out, "last_status.png"), draw_status(sc["frame"], de, sc))
            return

        print("live: 'c' capture + cloud, 's' save last result, 'q' quit")
        sc = None
        while True:
            raw = cams.publishimg("cam", flush=0)
            if raw is None:
                break
            frame, K, D = cams.undistort(raw)
            de.marker_pose(frame, K, D)                       # cheap: live tag overlay
            vis = draw_status(frame, de, sc)                  # live frame, last captured mask on top
            cv2.imshow("vla v2 (undistorted)", cv2.resize(vis, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                sc = de.scene_clouds(cams.publishimg("cam"), object_name=args.object)
            if key == ord("s") and sc is not None:
                save_outputs(sc, args.out, time.strftime("%Y%m%d_%H%M%S"))
            if key == ord("r"):
                sc = None
    finally:
        cams.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
