"""Live depth anchor: put the monocular body mesh at its true distance and size with a metric depth image.

A body mesh estimated from one picture has an unknown overall scale: making it s times bigger and s times farther from the
camera leaves every pixel of its outline where it was. One number, s, is therefore all the depth camera has to supply, and
applying it (scale about the camera origin) does not disturb the overlay or the pose. s is the robust median of
measured depth / mesh depth over the mesh vertices the camera can see (depth.anchor.depth_anchor), low-pass filtered because a
body does not change size from frame to frame.

depth.fusion.DepthFusion does the same and then refines the pose by ICP, but finds the visible vertices with a pure-Python
triangle rasteriser (2.8 s a frame on this mesh). Here visibility is a vertex splat plus a facing test (a few ms), and the pose
is left to the RGB model.

Everything is in the camera frame the mesh comes in: x right, y down, z forward, millimetres.
"""
import numpy as np

from .anchor import depth_anchor
from .synth import project, vertex_normals


def depth_map_from_points(points_mm, K, hw):
    """Depth image (mm, 0 where empty) of a point cloud seen by a pinhole camera K: nearest point per pixel."""
    h, w = hw
    z = points_mm[:, 2]
    ok = z > 1.0
    u = np.round(K[0, 0] * points_mm[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * points_mm[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    flat = np.full(h * w, np.inf)
    np.minimum.at(flat, v[inside] * w + u[inside], z[ok][inside])
    flat[~np.isfinite(flat)] = 0.0
    return flat.reshape(h, w)


def fast_visible(verts, faces, K, hw, tol_mm=40.0):
    """Vertices the camera sees: facing it, and within tol of the nearest vertex that falls in the same pixel neighbourhood."""
    h, w = hw
    uv, z = project(verts, K)
    ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 1.0)
    facing = np.einsum("ij,ij->i", vertex_normals(verts, faces), verts) < 0.0
    zbuf = np.full((h + 2, w + 2), np.inf)                      # padded, so the 3 x 3 minimum needs no edge cases
    np.minimum.at(zbuf, (vi[ok] + 1, ui[ok] + 1), z[ok])
    near = np.minimum.reduce([zbuf[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    vis = np.zeros(len(verts), bool)
    vis[ok] = z[ok] <= near[vi[ok], ui[ok]] + tol_mm
    return vis & facing


class LiveAnchor:
    """Filtered scale of one camera stream. update() -> info dict for the page ({"ok", "scale", ...})."""

    def __init__(self, faces, alpha=0.2, min_points=400, limits=(0.6, 1.6), max_rms_mm=60.0):
        self.faces = np.asarray(faces, np.int64)
        self.alpha, self.min_points, self.limits, self.max_rms_mm = alpha, min_points, limits, max_rms_mm
        self.scale = None

    def update(self, verts_cam_mm, depth_mm, K):
        vis = fast_visible(verts_cam_mm, self.faces, K, depth_mm.shape)
        _, a = depth_anchor(verts_cam_mm, depth_mm, K, vis, min_pts=self.min_points)
        info = {"ok": False, "visible": int(vis.sum()), "samples": int(a.get("n", 0))}
        if not a["ok"]:
            info["reason"] = "the body is not in the depth image" if info["samples"] < self.min_points else a.get("reason", "")
        elif not (self.limits[0] < a["value"] < self.limits[1]):
            info["reason"] = f"implausible scale {a['value']:.2f}"
        elif a["rms_mm"] > self.max_rms_mm:
            # after the best single scale the depth still does not follow the mesh: something else is in front of the camera
            info["reason"] = f"depth does not match the mesh (fit {a['rms_mm']:.0f} mm rms)"
        else:
            self.scale = a["value"] if self.scale is None else (1 - self.alpha) * self.scale + self.alpha * a["value"]
            info.update(ok=True, raw=round(a["value"], 4), rms_mm=round(a["rms_mm"], 1), spread=round(a["spread"], 4))
        if self.scale is not None:
            info["scale"] = round(float(self.scale), 4)        # the last good value keeps being applied through a dropout
        return info
