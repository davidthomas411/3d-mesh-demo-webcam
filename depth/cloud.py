"""Live point cloud from an Orbbec Femto Bolt, for display in the page.

The depth stream alone is opened (the browser can keep the Bolt's colour camera for the webcam picture). Each depth frame is turned
into 3-D points by the SDK's own point-cloud filter, which applies the factory calibration including the lens distortion, and the
points are moved into the COLOUR camera's frame with the factory depth-to-colour extrinsics. That is the frame the body mesh lives
in when the page's webcam is the Bolt's colour camera: x right, y down, z forward, origin at the colour camera.

Capture starts on the first request and stops by itself a few seconds after the last one, so the device is free when the option is
off. Needs pyorbbecsdk (pip install pyorbbecsdk2). Tested on a Femto Bolt, firmware 1.1.3, SDK wrapper 2.1.3, Windows.

What it is not: a registration. The cloud is metric; the mesh's distance from the camera comes from the RGB model's estimate of
focal length and body size, so the two agree only as well as that estimate is right (depth/fusion.py is the code that corrects it).
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
    """(N, 3) mm -> little-endian int16 triples (1 mm steps, +-32 m): what /cloud sends."""
    return np.clip(np.round(points_mm), -32767, 32767).astype("<i2").tobytes()


class FemtoCloud:
    def __init__(self, width=640, height=576, fps=30, max_points=30000, idle_stop_s=5.0):
        self.size, self.fps, self.max_points, self.idle_stop_s = (width, height), fps, max_points, idle_stop_s
        self.lock = threading.Lock()
        self.thread = None
        self.latest = None            # (seq, bytes, n_points)
        self.seq = 0
        self.last_request = 0.0
        self.error = None
        self.error_time = 0.0
        self.info = {}

    # ---- called from request threads
    def get(self):
        """Latest cloud as (seq, bytes, n) or None while there is none yet; starts the capture if it is not running."""
        self.last_request = time.time()
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                if self.error and time.time() - self.error_time < 3.0:
                    return None                               # it just failed: let the caller read the error, retry later
                self.error = None
                self.thread = threading.Thread(target=self._run, name="femto-cloud", daemon=True)
                self.thread.start()
            return self.latest

    def status(self):
        running = self.thread is not None and self.thread.is_alive()
        return {"running": running, "error": self.error, **self.info}

    # ---- capture thread
    def _run(self):
        pipe = None
        try:
            import pyorbbecsdk as ob
            pipe = ob.Pipeline()
            dp = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(self.size[0], self.size[1], ob.OBFormat.Y16, self.fps)
            cprof = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            ex = dp.get_extrinsic_to(cprof)
            # the device reports the rotation rounded to four decimals: make it a proper rotation again
            U, _, Vt = np.linalg.svd(np.array(ex.rot, np.float64).reshape(3, 3))
            R, t = U @ Vt, np.array(ex.transform, np.float64)
            cfg = ob.Config(); cfg.enable_stream(dp)
            pipe.start(cfg)
            pc = ob.PointCloudFilter(); pc.set_create_point_format(ob.OBFormat.POINT)
            dev = pipe.get_device().get_device_info()
            self.info = {"device": dev.get_name(), "connection": dev.get_connection_type(), "depth": list(self.size)}
            t_fps, n_fps = time.time(), 0
            while time.time() - self.last_request < self.idle_stop_s:
                frames = pipe.wait_for_frames(200)
                if frames is None or frames.get_depth_frame() is None:
                    continue
                pts = np.asarray(pc.calculate(pc.process(frames)), np.float64).reshape(-1, 3)
                pts = to_colour_frame(subsample(pts, self.max_points), R, t)
                self.seq += 1
                self.latest = (self.seq, pack(pts), len(pts))
                n_fps += 1
                if time.time() - t_fps >= 2.0:
                    self.info["fps"] = round(n_fps / (time.time() - t_fps), 1); t_fps, n_fps = time.time(), 0
        except Exception as exc:                              # no SDK, no device, device busy: report it, never raise
            self.error = f"{type(exc).__name__}: {exc}"; self.error_time = time.time()
        finally:
            self.latest = None
            try:
                if pipe is not None:
                    pipe.stop()
            except Exception:
                pass
