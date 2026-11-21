"""Live depth from an Orbbec Femto Bolt, for the point cloud in the page and for the depth anchor.

The depth stream alone is opened (the browser can keep the Bolt's colour camera for the webcam picture). What goes to the page is
the RAW depth image, 640 x 576 uint16 millimetres (the unbinned narrow mode), plus the calibration once:

  rays    for every depth pixel, the direction (x/z, y/z) it looks along, from the factory intrinsics and lens distortion
          (cv2.undistortPoints, 6-coefficient rational model). Checked against the SDK's own point-cloud filter on a live
          frame: median 0.004 mm at 1.5 m.
  R, t    factory depth-to-colour extrinsics: point_colour = R point_depth + t (mm)
  k       colour intrinsics as fractions of the picture size (the Bolt's 16:9 modes are one view at different sizes)

The page's graphics card turns that into one point per depth pixel (368,640 of them) in the COLOUR camera's frame, x right, y
down, z forward, and colours each from the video. The server therefore does no per-point work: a frame is copied and served. An
earlier version made the points here with the SDK's filter and thinned them to 30,000; converting every frame slowed inference
from about 120 to 210 ms (this thread shares the interpreter with the model loop) and the cloud was visibly coarser.

depth_map() is the one thing computed here, and only while the depth anchor asks for it: a small depth image as the colour
camera would see it (depth/live.py).

Capture starts on the first request and stops by itself a few seconds after the last one. Needs pyorbbecsdk
(pip install pyorbbecsdk2). Tested on a Femto Bolt, firmware 1.1.3, wrapper 2.1.3, Windows.
"""
import threading
import time

import numpy as np


def subsample(points_mm, max_points, z_range_mm=(250.0, 4500.0)):
    """Valid points within the depth range, thinned evenly to at most max_points. points_mm: (N, 3) float."""
    z = points_mm[:, 2]
    p = points_mm[(z > z_range_mm[0]) & (z < z_range_mm[1])]
    if len(p) > max_points:
        p = p[np.linspace(0, len(p) - 1, max_points).astype(np.int64)]
    return p


def to_colour_frame(points_mm, R, t_mm):
    """Depth-camera points -> colour-camera frame (row vectors)."""
    return points_mm @ np.asarray(R, np.float64).T + np.asarray(t_mm, np.float64)


def pack(points_mm):
    """(N, 3) mm -> little-endian int16 triples (1 mm steps, +-32 m)."""
    return np.clip(np.round(points_mm), -32767, 32767).astype("<i2").tobytes()


def ray_table(fx, fy, cx, cy, dist8, width, height):
    """(height, width, 2) float32: for each depth pixel the direction (x/z, y/z) it looks along.
    dist8 = (k1, k2, p1, p2, k3, k4, k5, k6), OpenCV's rational model."""
    import cv2
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    # iterated to convergence: the default five rounds leave half a pixel of error in the corners of this strongly distorted lens
    rays = cv2.undistortPointsIter(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), K, np.asarray(dist8, np.float64), None, None,
                                   (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 60, 1e-9))
    return rays.reshape(height, width, 2).astype(np.float32)


def points_from_depth(depth_mm, rays, step=1):
    """Depth image + ray table -> (N, 3) points in the depth camera's frame (mm), invalid pixels dropped."""
    z = depth_mm[::step, ::step].astype(np.float32)
    r = rays[::step, ::step]
    ok = z > 0
    z = z[ok]
    return np.column_stack([r[ok][:, 0] * z, r[ok][:, 1] * z, z])


class FemtoCloud:
    MAP_HW = (180, 320)

    def __init__(self, width=640, height=576, fps=30, idle_stop_s=5.0, map_rate=10.0):
        self.size, self.fps, self.idle_stop_s, self.map_interval = (width, height), fps, idle_stop_s, 1.0 / map_rate
        self.lock = threading.Lock()
        self.thread = None
        self.latest = None            # (seq, depth bytes, arrival time)
        self.seq = 0
        self.last_request = 0.0
        self.error = None
        self.error_time = 0.0
        self.info = {}
        self.calibration = None       # {"w", "h", "R" (9, row-major), "t" (3, mm), "k" (4)}: fixed for the device
        self.rays = None              # (h, w, 2) float32
        self.map_until = 0.0
        self.map_latest = None        # (depth_mm HxW, K)

    # ---- called from request threads
    def get(self):
        """Latest depth frame as (seq, bytes) or None while there is none yet; starts the capture if it is not running."""
        self.last_request = time.time()
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                if self.error and time.time() - self.error_time < 3.0:
                    return None                               # it just failed: let the caller read the error, retry later
                self.error = None
                self.thread = threading.Thread(target=self._run, name="femto-cloud", daemon=True)
                self.thread.start()
            return self.latest

    def depth_map(self):
        """Latest depth image as the COLOUR camera would see it, (depth_mm, K), or None. For depth/live.py."""
        self.map_until = time.time() + 2.0
        self.get()
        return self.map_latest

    def status(self):
        running = self.thread is not None and self.thread.is_alive()
        return {"running": running, "error": self.error, **self.info}

    # ---- capture thread
    def _run(self):
        pipe = None
        try:
            import pyorbbecsdk as ob
            from .live import depth_map_from_points
            pipe = ob.Pipeline()
            w, h = self.size
            dp = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(w, h, ob.OBFormat.Y16, self.fps)
            cprof = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            ex = dp.get_extrinsic_to(cprof)
            # the device reports the rotation rounded to four decimals: make it a proper rotation again
            U, _, Vt = np.linalg.svd(np.array(ex.rot, np.float64).reshape(3, 3))
            R, t = U @ Vt, np.array(ex.transform, np.float64)
            di, dd, ci = dp.get_intrinsic(), dp.get_distortion(), cprof.get_intrinsic()
            self.rays = ray_table(di.fx, di.fy, di.cx, di.cy, (dd.k1, dd.k2, dd.p1, dd.p2, dd.k3, dd.k4, dd.k5, dd.k6), w, h)
            k = [ci.fx / cprof.get_width(), ci.fy / cprof.get_height(), ci.cx / cprof.get_width(), ci.cy / cprof.get_height()]
            self.calibration = {"w": w, "h": h, "R": [float(x) for x in R.ravel()], "t": [float(x) for x in t], "k": k,
                                "kd": [float(di.fx), float(di.fy), float(di.cx), float(di.cy)],
                                "dist": [float(x) for x in (dd.k1, dd.k2, dd.p1, dd.p2, dd.k3, dd.k4, dd.k5, dd.k6)]}
            cfg = ob.Config(); cfg.enable_stream(dp)
            pipe.start(cfg)
            dev = pipe.get_device().get_device_info()
            self.info = {"device": dev.get_name(), "connection": dev.get_connection_type(), "depth": [w, h], "colour_k": k}
            mh, mw = self.MAP_HW
            K_map = np.array([[k[0] * mw, 0, k[2] * mw], [0, k[1] * mh, k[3] * mh], [0, 0, 1.0]])
            t_fps, n_fps, t_frame, t_map = time.time(), 0, time.time(), 0.0
            while time.time() - self.last_request < self.idle_stop_s:
                frames = pipe.wait_for_frames(200)
                d = frames.get_depth_frame() if frames is not None else None
                if d is None:
                    if time.time() - t_frame > 3.0:           # unplugged, or taken by another program: stop, so a retry reopens it
                        raise TimeoutError("no depth frames from the camera for 3 s")
                    continue
                t_frame = time.time()
                depth = np.frombuffer(d.get_data(), np.uint16)
                scale = d.get_depth_scale()
                if abs(scale - 1.0) > 1e-6:
                    depth = np.clip(depth.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
                self.seq += 1
                self.latest = (self.seq, depth.tobytes(), t_frame)   # t_frame: when the frame reached this computer
                if t_frame < self.map_until and t_frame - t_map >= self.map_interval:
                    t_map = t_frame
                    pts = to_colour_frame(points_from_depth(depth.reshape(h, w), self.rays, step=2), R, t)
                    self.map_latest = (depth_map_from_points(pts, K_map, self.MAP_HW), K_map)
                n_fps += 1
                if time.time() - t_fps >= 2.0:
                    self.info["fps"] = round(n_fps / (time.time() - t_fps), 1); t_fps, n_fps = time.time(), 0
        except Exception as exc:                              # no SDK, no device, device busy: report it, never raise
            self.error = f"{type(exc).__name__}: {exc}"; self.error_time = time.time()
        finally:
            self.latest = None; self.map_latest = None
            try:
                if pipe is not None:
                    pipe.stop()
            except Exception:
                pass


# Depth lens model of the Femto Bolt the first recordings were made with (narrow unbinned mode); recordings made before the
# calibration carried "kd" and "dist" use these.
BOLT_KD = (505.34368896484375, 505.4107666015625, 336.45892333984375, 325.3226623535156)
BOLT_DIST = (15.141837120056152, 8.009384155273438, 1.3242264685686678e-05, -2.4030552594922483e-05,
             0.3177873492240906, 15.448055267333984, 13.116472244262695, 1.9120935201644897)


class RecordedCloud:
    """Stands in for FemtoCloud with the depth frames of a recording (GET /record), for testing without the camera. The
    recording has one depth frame per model result (about five a second); each is served as a NEW frame thirty times a second
    until the next one is due, so the fit loop is loaded as it would be live."""

    def __init__(self, path, fps=30.0, loop=True):
        import json
        z = np.load(path, allow_pickle=True)
        keep = np.where(z["has_depth"])[0]
        self.frames, self.times = z["depth"][keep], z["t"][keep] - z["t"][keep][0]
        self.rays = z["rays"]
        cal = json.loads(str(z["calibration"]))
        cal.setdefault("kd", list(BOLT_KD)); cal.setdefault("dist", list(BOLT_DIST))
        self.calibration = cal
        self.info = {"device": "recording " + path, "connection": "file", "depth": [cal["w"], cal["h"]], "colour_k": cal["k"], "fps": fps}
        self.error, self.fps, self.loop = None, fps, loop
        self.t0 = None
        self.map_latest = None

    def restart(self):
        self.t0 = time.time()

    def get(self):
        if self.t0 is None:
            self.t0 = time.time()
        el = time.time() - self.t0
        if self.loop:
            el = el % (self.times[-1] + 0.2)
        i = int(np.clip(np.searchsorted(self.times, el, side="right") - 1, 0, len(self.frames) - 1))
        seq = int((time.time() - self.t0) * self.fps) + 1
        return (seq, self.frames[i].tobytes(), self.t0 + (seq - 1) / self.fps)   # when this frame became the newest

    @property
    def latest(self):
        return self.get()

    def depth_map(self):
        return None

    def status(self):
        return {"running": True, "error": None, **self.info}
