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


class ArticulatedAnchor:
    """Depth fit of the body, part by part, along the camera's rays.

    One scale about the camera is right only if the model's pose is right and its overall distance is the one unknown. A
    monocular pose has more to give than that: the picture fixes where each joint is in the IMAGE, so what it cannot know is how
    far along its ray each joint is. That is one number per joint, and the depth camera measures it. Here every joint j gets its
    own scale s_j about the camera origin:

        joint'  = s_j * joint                      (same pixel, measured distance)
        vertex' = (sum_j w_vj s_j) * vertex        (w: the body model's own skinning weights)

    so a forearm reaching towards the camera comes forward, a leaning trunk leans, and the outline in the picture does not
    move. A part's size changes with its scale, which is what perspective demands: something nearer that covers the same pixels
    is smaller. s_j is the weighted mean of measured depth / mesh depth over the visible vertices joint j carries. A joint with
    little or no evidence (legs under a desk, the far side of the body) takes its parent's value, a child may not differ from
    its parent by more than max_step, and the values are filtered in time.

    What it does not do: move anything across the image (an error in the model's 2-D pose stays), or change a part's shape.
    """

    def __init__(self, faces, parents, w_joint, w_vert, w, n0=25.0, max_step=0.15, alpha=0.35, limits=(0.6, 1.6),
                 min_points=400, max_rms_mm=60.0):
        from scipy.sparse import csr_matrix
        self.faces = np.asarray(faces, np.int64)
        self.parents = np.asarray(parents, np.int64)
        nj, nv = len(self.parents), int(np.max(w_vert)) + 1
        W = csr_matrix((np.asarray(w, np.float64), (np.asarray(w_vert), np.asarray(w_joint))), shape=(nv, nj))
        self.W = W.multiply(1.0 / np.maximum(np.asarray(W.sum(1)), 1e-9)).tocsr()       # rows sum to one
        self.Wt = self.W.T.tocsr()
        self.order = list(np.argsort(self._depths(), kind="stable"))                    # parents before children
        self.n0, self.max_step, self.alpha, self.limits = n0, max_step, alpha, limits
        self.min_points, self.max_rms_mm = min_points, max_rms_mm
        self.s = None

    def _depths(self):
        d = np.zeros(len(self.parents), np.int64)
        for j in range(len(self.parents)):
            k, n = j, 0
            while self.parents[k] >= 0 and self.parents[k] != k and n < 200:
                k = self.parents[k]; n += 1
            d[j] = n
        return d

    def joint_scales(self, verts_cam_mm, depth_mm, K):
        """-> (s_j or None, info). Not filtered in time."""
        h, w = depth_mm.shape
        vis = fast_visible(verts_cam_mm, self.faces, K, (h, w))
        uv, z = project(verts_cam_mm, K)
        ui, vi = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
        ok = vis & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 1.0)
        d = np.zeros(len(z)); d[ok] = depth_mm[vi[ok], ui[ok]]
        ok &= d > 0
        info = {"ok": False, "visible": int(vis.sum()), "samples": int(ok.sum())}
        if ok.sum() < self.min_points:
            info["reason"] = "the body is not in the depth image"
            return None, info
        r = np.zeros(len(z)); r[ok] = d[ok] / z[ok]
        g = float(np.median(r[ok]))
        if not (self.limits[0] < g < self.limits[1]):
            info["reason"] = f"implausible scale {g:.2f}"
            return None, info
        ok &= np.abs(r - g) < 0.25 * g                          # background seen past the outline, an object in front
        m = ok.astype(np.float64)
        n_j = self.Wt @ m                                       # evidence per joint (sum of skinning weights)
        mean_j = (self.Wt @ (m * r)) / np.maximum(n_j, 1e-9)
        s = np.empty(len(self.parents))
        for j in self.order:
            p = self.parents[j]
            prior = g if (p < 0 or p == j) else s[p]
            c = n_j[j] / (n_j[j] + self.n0)
            s[j] = np.clip(c * mean_j[j] + (1.0 - c) * prior, prior * (1 - self.max_step), prior * (1 + self.max_step))
        sv = self.W @ s
        res_one, res_art = d[ok] - g * z[ok], d[ok] - sv[ok] * z[ok]
        seen = n_j > self.n0
        info.update(global_scale=round(g, 4), rms_single_mm=round(float(np.sqrt(np.mean(res_one ** 2))), 1),
                    rms_mm=round(float(np.sqrt(np.mean(res_art ** 2))), 1), samples=int(ok.sum()), parts_seen=int(seen.sum()),
                    part_range=[round(float(s[seen].min()), 3), round(float(s[seen].max()), 3)] if seen.any() else None)
        if info["rms_mm"] > self.max_rms_mm:
            info["reason"] = f"depth does not match the mesh (fit {info['rms_mm']:.0f} mm rms)"
            return None, info
        info["ok"] = True
        return s, info

    def update(self, verts_cam_mm, depth_mm, K):
        """-> (per-joint scales, filtered; None while there has never been a fit, info)."""
        s, info = self.joint_scales(verts_cam_mm, depth_mm, K)
        if s is not None:
            self.s = s if self.s is None else (1 - self.alpha) * self.s + self.alpha * s
        if self.s is not None:
            info["scale"] = round(float(np.median(self.s)), 4)
        return self.s, info

    def apply(self, verts_cam_mm, joints_cam_mm=None, s=None):
        """Scaled vertices (and joints) in the camera frame."""
        s = self.s if s is None else s
        v = verts_cam_mm * (self.W @ s)[:, None]
        return (v, None) if joints_cam_mm is None else (v, joints_cam_mm * s[:, None])
