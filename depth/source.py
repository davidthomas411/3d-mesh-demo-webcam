"""Depth sources. The Femto Bolt backend is UNTESTED here (no hardware); the synthetic and
recorded sources are what the test-suite exercises."""
import time
import numpy as np


class DepthSource:
    K = None            # 3x3 intrinsics of the depth image (colour-aligned for the Bolt)
    def read(self, timeout_ms=100):
        """-> (depth_mm uint16 HxW, device_timestamp_us) or (None, None)."""
        raise NotImplementedError
    def close(self):
        pass


class RecordedSource(DepthSource):
    """npz with depth (T,H,W) uint16 mm, K (3,3), optional ts_us (T,). Loops forever."""
    def __init__(self, path):
        z = np.load(path)
        self.depth, self.K = z["depth"], z["K"]
        self.ts = z["ts_us"] if "ts_us" in z else np.arange(len(self.depth)) * 33333
        self.i = 0
    def read(self, timeout_ms=100):
        j = self.i % len(self.depth); self.i += 1
        return self.depth[j], int(self.ts[j])


class SyntheticSource(DepthSource):
    def __init__(self, depth_fn, K):
        self.fn, self.K, self.i = depth_fn, K, 0
    def read(self, timeout_ms=100):
        self.i += 1
        return self.fn(self.i), int(time.time() * 1e6)


class FemtoBoltSource(DepthSource):
    """Orbbec Femto Bolt via pyorbbecsdk, depth aligned to colour (software align).

    UNTESTED against hardware. Verify against your installed pyorbbecsdk version
    (names below follow the v1.x examples: Pipeline/Config/OBAlignMode).
    Recommended mode: NFOV unbinned. Feed SAM 3D Body the Bolt's own RGB (see color()).
    """
    def __init__(self, depth_w=640, depth_h=576, fps=30):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        self.pipe, cfg = Pipeline(), Config()
        dp = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            depth_w, depth_h, OBFormat.Y16, fps)
        cp = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        cfg.enable_stream(dp); cfg.enable_stream(cp)
        cfg.set_align_mode(OBAlignMode.SW_MODE)
        self.pipe.start(cfg)
        p = self.pipe.get_camera_param().rgb_intrinsic
        self.K = np.array([[p.fx, 0, p.cx], [0, p.fy, p.cy], [0, 0, 1.0]])
        self._color = None

    def read(self, timeout_ms=100):
        frames = self.pipe.wait_for_frames(timeout_ms)
        if frames is None or frames.get_depth_frame() is None:
            return None, None
        d = frames.get_depth_frame()
        depth = np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width())
        depth = (depth.astype(np.float32) * d.get_depth_scale()).astype(np.uint16)
        self._color = frames.get_color_frame()
        return depth, int(d.get_timestamp_us()) if hasattr(d, "get_timestamp_us") else int(time.time() * 1e6)

    def close(self):
        self.pipe.stop()
