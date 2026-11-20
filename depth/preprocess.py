"""Depth clean-up for ToF artefacts: flying pixels, mask edges, support plane."""
import numpy as np
from scipy.ndimage import binary_erosion


def reject_flying(depth_mm, grad_mm=60.0):
    """Zero pixels whose depth differs from any 4-neighbour by more than grad_mm."""
    d = depth_mm.astype(np.float32)
    bad = np.zeros(d.shape, bool)
    for ax in (0, 1):
        for sh in (1, -1):
            n = np.roll(d, sh, ax)
            bad |= (np.abs(d - n) > grad_mm) & (n > 0)
    out = depth_mm.copy()
    out[bad] = 0
    return out


def erode_mask(mask, px=2):
    return binary_erosion(mask, iterations=px) if px > 0 else mask


def fit_plane_ransac(points, iters=200, tol_mm=8.0, seed=0):
    """Return (n, d) with n.p + d = 0 and the inlier mask; n oriented toward the camera origin."""
    rng = np.random.default_rng(seed)
    best, best_n, best_d = None, None, None
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        inl = np.abs(points @ n + d) < tol_mm
        if best is None or inl.sum() > best.sum():
            best, best_n, best_d = inl, n, d
    if best_d > 0:
        best_n, best_d = -best_n, -best_d
    return best_n, best_d, best


def remove_support_plane(depth_mm, K, tol_mm=8.0, sample=4000, seed=0):
    """Remove the couch/table: the dominant plane in the cloud. Returns (depth, plane)."""
    from .synth import backproject
    pts = backproject(depth_mm, K)
    valid = depth_mm > 0
    idx = np.flatnonzero(valid.ravel())
    if len(idx) < 100:
        return depth_mm, None
    rng = np.random.default_rng(seed)
    sub = rng.choice(idx, min(sample, len(idx)), replace=False)
    n, d, _ = fit_plane_ransac(pts.reshape(-1, 3)[sub], tol_mm=tol_mm, seed=seed)
    dist = pts @ n + d
    out = depth_mm.copy()
    out[(np.abs(dist) < tol_mm * 1.5)] = 0
    return out, (n, d)
