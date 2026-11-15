"""Projective point-to-plane ICP against a metric depth image (KinectFusion-style).

No KD-tree: correspondences come from projecting each source point into the depth
image. Robust (Huber) weights, optional one-sided loss for gowns/blankets.
"""
import numpy as np
from .synth import backproject, project


def depth_normals(depth_mm, K):
    p = backproject(depth_mm, K)
    dx = np.zeros_like(p); dy = np.zeros_like(p)
    dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dy[1:-1] = p[2:] - p[:-2]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nn > 1e-9, n / np.maximum(nn, 1e-9), 0)
    flip = (n * p).sum(-1) > 0  # make normals face the camera
    n[flip] *= -1
    return n, p


def _small_rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def projective_icp(points, normals, depth_mm, K, T0=None, iters=6, huber_mm=10.0,
                   max_dist_mm=60.0, min_cos=0.5, point_mask=None, cloth_mm=0.0,
                   dnorm=None):
    """Align source (points, normals; camera-frame mm) to depth. Returns (T 4x4, info).

    T maps source -> camera: p' = R p + t, starting from T0.
    cloth_mm: known thickness of gown/blanket. The observed surface is assumed cloth_mm
    outward of the skin along its camera-facing normal, so the target is moved back by that
    amount. (A one-sided hinge loss was rejected: it cannot observe motion toward the camera.)
    """
    T = np.eye(4) if T0 is None else T0.copy()
    h, w = depth_mm.shape
    if dnorm is None:
        dnorm = depth_normals(depth_mm, K)[0]
    if point_mask is None:
        point_mask = np.ones(len(points), bool)
    info = {"ok": False, "rms_mm": np.nan, "inlier_frac": 0.0, "n": 0}
    for _ in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        P = points @ R.T + t
        N = normals @ R.T
        uv, z = project(P, K)
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = point_mask & (z > 1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ok &= (N * P).sum(1) < 0          # front-facing wrt camera
        idx = np.flatnonzero(ok)
        d = depth_mm[vi[idx], ui[idx]].astype(np.float64)
        has = d > 0
        idx, d = idx[has], d[has]
        ui2, vi2 = ui[idx], vi[idx]
        Q = np.stack([(ui2 - K[0, 2]) * d / K[0, 0], (vi2 - K[1, 2]) * d / K[1, 1], d], 1)
        Nq = dnorm[vi2, ui2]
        good = (np.linalg.norm(Nq, axis=1) > 0.5) & (np.linalg.norm(Q - P[idx], axis=1) < max_dist_mm)
        good &= (Nq * N[idx]).sum(1) > min_cos
        idx, Q, Nq = idx[good], Q[good], Nq[good]
        info["n"] = len(idx)
        if len(idx) < 30:
            return T, info
        Pi = P[idx]
        r = (Nq * (Pi - Q)).sum(1) + cloth_mm   # signed distance model->plane (skin-corrected)
        wgt = np.where(np.abs(r) <= huber_mm, 1.0, huber_mm / np.maximum(np.abs(r), 1e-9))
        J = np.hstack([np.cross(Pi, Nq), Nq])
        A = (J * wgt[:, None]).T @ J + 1e-6 * np.eye(6)
        g = -(J * wgt[:, None]).T @ r
        x = np.linalg.solve(A, g)
        dT = np.eye(4)
        dT[:3, :3] = _small_rot(x[:3])
        dT[:3, 3] = x[3:]
        T = dT @ T
        info.update(ok=True, rms_mm=float(np.sqrt(np.average(r ** 2, weights=wgt))),
                    inlier_frac=float(np.mean(np.abs(r) <= huber_mm)))
        if np.linalg.norm(x[3:]) < 0.02 and np.linalg.norm(x[:3]) < 1e-5:
            break
    return T, info


def roi_icp_cloud(ref_pts, ref_nrm, live_depth_mm, K, T0=None, roi_mask=None, **kw):
    """Depth-to-reference surface registration (the SGRT-style 6DOF measurement).

    ref_pts/ref_nrm: reference-capture surface samples in camera frame. Returns (T, info)
    with T mapping the reference pose to the live pose (couch shift = T translation).
    """
    return projective_icp(ref_pts, ref_nrm, live_depth_mm, K, T0=T0, point_mask=roi_mask, **kw)
