"""Depth-anchored metric placement of a monocular body mesh.

SAM 3D Body returns pose/shape with an ambiguous overall scale/depth. Scaling the
mesh about the camera origin does not move any pixel of its 2-D overlay, so a single
robust scale factor measured against metric depth fixes the ambiguity for free.
"""
import numpy as np
from .synth import project, rasterize_depth


def visible_vertices(verts, faces, K, hw, tol_mm=15.0):
    """Vertices whose depth matches the mesh's own z-buffer (i.e. not self-occluded)."""
    h, w = hw
    zbuf = rasterize_depth(verts, faces, K, hw)
    uv, z = project(verts, K)
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 1)
    vis = np.zeros(len(verts), bool)
    vis[ok] = z[ok] <= zbuf[vi[ok], ui[ok]] + tol_mm
    return vis


def depth_anchor(verts, depth_mm, K, mask, mode="scale", min_pts=200, trim=0.2):
    """Return (transformed_verts, info). verts are camera-frame mm.

    mode="scale":     V' = s*V,  s = robust median(depth/z)   (overlay-preserving)
    mode="translate": V' = V + (0,0,dz), dz = robust median(depth - z)
    One-sided gating via a trimmed median keeps clothing/couch/outliers from dominating.
    """
    h, w = depth_mm.shape
    uv, z = project(verts, K)
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = mask & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 1)
    d = np.zeros(len(verts))
    d[ok] = depth_mm[vi[ok], ui[ok]]
    ok &= d > 0
    n = int(ok.sum())
    if n < min_pts:
        return verts, {"ok": False, "n": n, "reason": "too few depth samples"}
    ratio = d[ok] / z[ok] if mode == "scale" else d[ok] - z[ok]
    # reject gross outliers (couch / hands / flying pixels) around the median, then re-median
    med = np.median(ratio)
    dev = np.abs(ratio - med)
    keep = dev <= np.quantile(dev, 1 - trim)
    est = float(np.median(ratio[keep]))
    spread = float(1.4826 * np.median(np.abs(ratio[keep] - est)))
    if mode == "scale":
        out = verts * est
        resid = d[ok][keep] - (z[ok][keep] * est)
    else:
        out = verts + np.array([0, 0, est])
        resid = d[ok][keep] - (z[ok][keep] + est)
    return out, {"ok": True, "n": n, "mode": mode, "value": est, "spread": spread,
                 "rms_mm": float(np.sqrt(np.mean(resid ** 2)))}
