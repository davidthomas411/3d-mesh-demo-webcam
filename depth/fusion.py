"""Per-frame fusion of SAM 3D Body vertices with a metric depth frame."""
import numpy as np
from .anchor import visible_vertices, depth_anchor
from .icp import projective_icp
from .synth import vertex_normals


def mhr_to_camera_mm(vertices_m, cam_t_m, flip_yz=True):
    """SAM 3D Body output -> OpenCV camera frame in mm.

    The browser client places the mesh at (t0, -t1, -t2), i.e. the raw output is
    y-up/z-toward-camera. flip_yz mirrors that. VERIFY ON HARDWARE (see README).
    """
    v = np.asarray(vertices_m, np.float64) + np.asarray(cam_t_m, np.float64).reshape(1, 3)
    if flip_yz:
        v = v * np.array([1.0, -1.0, -1.0])
    return v * 1000.0


class DepthFusion:
    """Depth-anchor (scale about the camera origin) -> optional projective ICP refine.

    The anchor scale is low-pass filtered (EMA) because a body does not change size;
    only the filtered value is applied, so depth noise cannot make the mesh breathe.
    """

    def __init__(self, faces, K, icp_iters=4, alpha=0.25, cloth_mm=0.0, refine=True, min_valid=0.15):
        self.faces = np.asarray(faces, np.int32)
        self.K, self.icp_iters, self.alpha = K, icp_iters, alpha
        self.cloth_mm, self.refine, self.min_valid = cloth_mm, refine, min_valid
        self.scale = None

    def process(self, verts_cam_mm, depth_mm):
        hw = depth_mm.shape
        vis = visible_vertices(verts_cam_mm, self.faces, self.K, hw)
        out, a = depth_anchor(verts_cam_mm, depth_mm, self.K, vis)
        info = {"anchor": a, "valid_fraction": a.get("n", 0) / max(int(vis.sum()), 1)}
        if not a["ok"] or info["valid_fraction"] < self.min_valid:
            info["ok"] = False
            return verts_cam_mm, info
        s = a["value"]
        self.scale = s if self.scale is None else (1 - self.alpha) * self.scale + self.alpha * s
        out = verts_cam_mm * self.scale
        T = np.eye(4)
        if self.refine:
            vis2 = visible_vertices(out, self.faces, self.K, hw)
            T, icp = projective_icp(out, vertex_normals(out, self.faces), depth_mm, self.K,
                                    iters=self.icp_iters, point_mask=vis2, cloth_mm=self.cloth_mm)
            info["icp"] = icp
            out = out @ T[:3, :3].T + T[:3, 3]
        info.update(ok=True, scale=float(self.scale), T=T.tolist())
        return out, info
