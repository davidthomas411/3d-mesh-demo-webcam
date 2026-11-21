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
    # 320 x 288 is the Bolt's binned narrow mode: 92,000 points a frame (30,000 are shown), less noise than the unbinned mode
    # and a quarter of the work. Clouds are made at most `rate` times a second: this thread shares the interpreter with the
    # inference loop, and converting every frame at 640 x 576 slowed inference from about 120 to 210 ms.
    def __init__(self, width=320, height=288, fps=30, max_points=30000, idle_stop_s=5.0, rate=15.0):
        self.size, self.fps, self.max_points, self.idle_stop_s = (width, height), fps, max_points, idle_stop_s
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.thread = None
        self.latest = None            # (seq, bytes, n_points)
        self.seq = 0
        self.last_request = 0.0
        self.error = None
        self.error_time = 0.0
        self.info = {}
        self.map_until = 0.0          # a depth image aligned to the colour camera is made while somebody asks for it
        self.map_latest = None        # (depth_mm HxW, K)

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

    MAP_HW = (180, 320)

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
            ci = cprof.get_intrinsic()
            k = [ci.fx / cprof.get_width(), ci.fy / cprof.get_height(), ci.cx / cprof.get_width(), ci.cy / cprof.get_height()]
            # colour intrinsics as fractions of the picture size (the Bolt's 16:9 modes are the same view at different sizes)
            self.info = {"device": dev.get_name(), "connection": dev.get_connection_type(), "depth": list(self.size), "colour_k": k}
            h, w = self.MAP_HW
            K_map = np.array([[k[0] * w, 0, k[2] * w], [0, k[1] * h, k[3] * h], [0, 0, 1.0]])
            t_fps, n_fps = time.time(), 0
            t_frame = time.time()
            while time.time() - self.last_request < self.idle_stop_s:
                frames = pipe.wait_for_frames(200)
                if frames is None or frames.get_depth_frame() is None:
                    if time.time() - t_frame > 3.0:           # unplugged, or taken by another program: stop, so a retry reopens it
                        raise TimeoutError("no depth frames from the camera for 3 s")
                    continue
                if time.time() - t_frame < self.min_interval:
                    continue                                  # newer than needed: leave the processor to the model
                t_frame = time.time()
                pts = np.asarray(pc.calculate(pc.process(frames)), np.float32).reshape(-1, 3)
                if time.time() < self.map_until:
                    from .live import depth_map_from_points
                    full = pts[pts[:, 2] > 0]
                    self.map_latest = (depth_map_from_points(to_colour_frame(full, R, t), K_map, self.MAP_HW), K_map)
                pts = to_colour_frame(subsample(pts, self.max_points), R, t)
                self.seq += 1
                self.latest = (self.seq, pack(pts), len(pts))
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
