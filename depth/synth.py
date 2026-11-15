"""Synthetic depth rendering: the test oracle for the Femto Bolt fusion code.

Camera convention everywhere in this package: OpenCV (x right, y down, z forward, pixel centres at integer coordinates),
units are millimetres, depth images are uint16 millimetres with 0 = invalid.
"""
import numpy as np


def make_K(width, height, hfov_deg=75.0):
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def project(points, K):
    z = points[:, 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * points[:, 0] / zs + K[0, 2]
    v = K[1, 1] * points[:, 1] / zs + K[1, 2]
    return np.stack([u, v], 1), z


def backproject(depth_mm, K):
    h, w = depth_mm.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth_mm.astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], -1)


def rasterize_depth(verts, faces, K, hw):
    """Z-buffer (float, mm; inf where empty) by per-triangle barycentric fill."""
    h, w = hw
    zbuf = np.full((h, w), np.inf)
    uv, z = project(verts, K)
    for f in faces:
        if np.any(z[f] <= 1.0):
            continue
        p = uv[f]
        x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), w - 1))
        y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(d) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a = ((p[1, 1] - p[2, 1]) * (gx - p[2, 0]) + (p[2, 0] - p[1, 0]) * (gy - p[2, 1])) / d
        b = ((p[2, 1] - p[0, 1]) * (gx - p[2, 0]) + (p[0, 0] - p[2, 0]) * (gy - p[2, 1])) / d
        c = 1 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        # perspective-correct depth: 1/z is affine in screen space
        iz = a / z[f[0]] + b / z[f[1]] + c / z[f[2]]
        zz = np.where(inside, 1.0 / iz, np.inf)
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        np.minimum(sub, zz, out=sub)
    return zbuf


def render_depth(verts, faces, K, hw, noise_mm=0.0, flying_px=0.0, dropout=0.0, seed=0):
    """Render a Femto-Bolt-like depth map (uint16 mm) with optional artefacts."""
    rng = np.random.default_rng(seed)
    z = rasterize_depth(verts, faces, K, hw)
    valid = np.isfinite(z)
    if noise_mm > 0:
        z = z + rng.normal(0, noise_mm, z.shape)
    if dropout > 0:
        valid &= rng.random(z.shape) > dropout
    if flying_px > 0:  # smear silhouette pixels toward the background (ToF flying pixels)
        from scipy.ndimage import binary_dilation, binary_erosion
        edge = valid & ~binary_erosion(valid, iterations=1)
        pick = edge & (rng.random(z.shape) < flying_px)
        z = np.where(pick, z + rng.uniform(100, 400, z.shape), z)
    out = np.where(valid, z, 0.0)
    return np.clip(np.round(out), 0, 65535).astype(np.uint16)


def uv_sphere(radius=1.0, n_lat=24, n_lon=48):
    lat = np.linspace(0, np.pi, n_lat + 1)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    v = [(0, -radius, 0)]
    for la in lat[1:-1]:
        for lo in lon:
            v.append((radius * np.sin(la) * np.cos(lo), -radius * np.cos(la), radius * np.sin(la) * np.sin(lo)))
    v.append((0, radius, 0))
    v = np.array(v)
    f = []
    for j in range(n_lon):
        f.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            f += [(a, c, b), (b, c, d)]
    last = len(v) - 1
    for j in range(n_lon):
        f.append((last, 1 + (n_lat - 2) * n_lon + (j + 1) % n_lon, 1 + (n_lat - 2) * n_lon + j))
    return v, np.array(f, dtype=np.int32)


def phantom_torso(scale=(180.0, 450.0, 110.0), n_lat=28, n_lon=56):
    """Ellipsoid 'torso' + two ellipsoid arms, an anisotropic, asymmetric test body (mm)."""
    v, f = uv_sphere(1.0, n_lat, n_lon)
    parts = [(v * np.array(scale), f)]
    for sx in (-1, 1):
        a = v * np.array([60, 260, 60]) + np.array([sx * 250, 0, 0])
        parts.append((a, f))
    verts, faces, off = [], [], 0
    for pv, pf in parts:
        verts.append(pv); faces.append(pf + off); off += len(pv)
    return np.vstack(verts), np.vstack(faces)


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
