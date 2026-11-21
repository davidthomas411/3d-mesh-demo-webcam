"""Where the camera is for every frame of a gantry recording, found from the depth alone.

The camera rides on the gantry, so in its own frame the room turns about a fixed line: the axis of the gantry. The large
planes of the room (floor, ceiling, side walls) are parallel to that axis and at right angles to one another, which gives:

  fixed parts    pixels whose depth never changes are the gantry itself (the treatment head): masked out everywhere.
  axis           the direction all those planes are parallel to.
  angle          the direction of the planes about the axis, modulo 90 degrees, unwrapped over time; 0 at frame 0.
  axis position  the distance of the floor from the camera at each angle: n(theta) . p + d(theta) = h for a point p on the
                 axis and the height h of the axis above the floor.

Room frame: origin on the axis (the point nearest the camera origin), Y along the axis, Z up (away from the floor),
X = Y x Z; millimetres. Along the axis the origin is arbitrary: it is NOT the isocentre until the cube or the beam axis fixes
it. Angle 0 is wherever the gantry stood at frame 0.
"""
import json
import os

import numpy as np


def planes(P, rng, k=3, iters=250, thr=12.0):
    """The k largest planes among the points P (n, 3): [(normal towards the camera, distance d with n.x + d = 0, pixels)]."""
    out, rest = [], P
    for _ in range(k):
        if len(rest) < 500:
            break
        best = None
        for _ in range(iters):
            a, b, c = rest[rng.integers(0, len(rest), 3)]
            n = np.cross(b - a, c - a); ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n /= ln
            m = np.abs((rest - a) @ n) < thr * (1 + np.abs(rest[:, 2]) / 3000)
            if best is None or m.sum() > best[0]:
                best = (m.sum(), m)
        m = best[1]; Q = rest[m]; c = Q.mean(0)
        w, v = np.linalg.eigh(np.cov((Q - c).T)); n = v[:, 0]
        if n @ c > 0:
            n = -n
        out.append((n, float(-n @ c), int(m.sum()))); rest = rest[~m]
    return out


class GantryGeometry:
    def __init__(self, path):
        g = np.load(path)
        self.frames, self.theta, self.a, self.p, self.h, self.sgn, self.nf0 = g["frames"], g["theta"], g["a"], g["p"], float(g["h"]), float(g["sgn"]), g["nf0"]
        self.fixed = g["fixed"]
        self.R0 = np.stack([np.cross(self.a, self.nf0), self.a, self.nf0])

    def angle(self, i):
        return float(np.interp(i, self.frames, self.theta))

    def to_room(self, P, i):
        """Camera-frame points of frame i -> room frame."""
        from scipy.spatial.transform import Rotation as Rot
        return Rot.from_rotvec(np.radians(-self.sgn * self.angle(i)) * self.a).apply(P - self.p) @ self.R0.T


def fixed_mask(rec):
    import warnings

    import cv2
    D = np.stack([rec.depth(i)[::2, ::2] for i in range(0, rec.n, 30)]).astype(np.float32)
    Dn = np.where(D > 0, D, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Dn, 0); mad = np.nanmedian(np.abs(Dn - med), 0)
    fixed = ((D > 0).mean(0) > 0.9) & (mad < 6) & (med < 1300)
    return cv2.dilate(cv2.resize(fixed.astype(np.uint8), rec.size), np.ones((25, 25), np.uint8)) > 0


def axis_of(res):
    N = np.array([n for i in res for n, d, k in res[i] if k > 1200])
    w, v = np.linalg.eigh(N.T @ N); a = v[:, 0]                   # the direction most normals are perpendicular to
    for _ in range(10):
        keep = np.abs(N @ a) < 0.08
        w, v = np.linalg.eigh(N[keep].T @ N[keep]); a = v[:, 0]
    return a


def angles_of(res, a):
    from scipy.signal import medfilt
    e1 = np.cross(a, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    fr = sorted(res); th = []
    for i in fr:
        c = sum(k * np.exp(4j * np.arctan2(n @ e2, n @ e1)) for n, d, k in res[i] if abs(n @ a) < 0.06 and k > 1200)
        th.append(np.angle(c) if abs(c) > 0 else np.nan)
    # Each sample gives the angle modulo 90 degrees. Follow it through time: a gantry turns slowly (a degree or two between
    # samples), so the branch nearest the last accepted angle is taken, and a sample that would mean a jump (planes of the
    # room hidden behind the patient, a couch edge mistaken for a wall) is left out and bridged.
    raw = np.degrees(np.array(th)) / 4
    out = np.full(len(fr), np.nan)
    last_i, last_t, rate = None, 0.0, 0.0
    for k, i in enumerate(fr):
        if not np.isfinite(raw[k]):
            continue
        if last_i is None:
            out[k], last_i, last_t = raw[k], i, raw[k]
            continue
        gap = i - last_i
        pred = last_t + rate * gap
        t = raw[k] + 90.0 * np.round((pred - raw[k]) / 90.0)
        if abs(t - pred) > min(40.0, 3.0 + 0.45 * gap):          # more than a gantry could have turned
            continue
        rate = 0.6 * rate + 0.4 * (t - last_t) / gap
        out[k], last_i, last_t = t, i, t
    ok = np.isfinite(out)
    th = np.interp(fr, np.array(fr)[ok], out[ok])
    return np.array(fr), medfilt(th - th[0], 5)


def calibrate(rec, out_dir, step=24, fine=6, floor=None):
    from scipy.spatial.transform import Rotation as Rot
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    fixed = fixed_mask(rec)
    print("fixed to the gantry: %.0f%% of the picture" % (100 * fixed.mean()), flush=True)

    def fit(i):
        P, _ = rec.points(rec.depth(i), mask=~fixed, step=6)
        return planes(P, rng)
    res = {i: fit(i) for i in range(0, rec.n, step)}
    a = axis_of(res)
    fr, th = angles_of(res, a)
    for k in np.where(np.abs(np.gradient(th)) > 0.3)[0]:          # sample finely wherever the gantry was moving
        for i in range(max(0, fr[k] - step), min(rec.n, fr[k] + step), fine):
            if i not in res:
                res[i] = fit(i)
    a = axis_of(res)
    fr, th = angles_of(res, a)
    # Which of the room's planes is the floor cannot be told from geometry alone. By default: of the planes parallel to the
    # axis at frame 0, the nearest (a camera at isocentre height is nearer the floor than a wall). --floor K overrides.
    home = [(n, d, k) for n, d, k in res[int(fr[0])] if abs(n @ a) < 0.06]
    print("planes parallel to the axis at frame 0 (normal, distance mm, pixels):")
    for k, (n, d, c) in enumerate(home):
        print("  %d: %s %6.0f %6d" % (k, np.round(n, 2).tolist(), d, c))
    fi = int(np.argmin([d for n, d, c in home])) if floor is None else int(floor)
    print("floor: plane", fi, flush=True)
    nf0 = home[fi][0].copy(); nf0 -= a * (a @ nf0); nf0 /= np.linalg.norm(nf0)
    best = None
    for sgn in (1.0, -1.0):
        rows = {}
        for i, t in zip(fr, th):
            nf = Rot.from_rotvec(np.radians(sgn * t) * a).apply(nf0)
            for n, d, k in res[int(i)]:
                if n @ nf > 0.995 and k > 600 and (int(i) not in rows or d > rows[int(i)][1]):
                    rows[int(i)] = (n, d, t)                      # of parallel planes (floor, couch top) the farther is the floor
        if best is None or len(rows) > len(best[1]):
            best = (sgn, rows)
    sgn, rows = best
    N = np.array([r[0] for r in rows.values()]); Dd = np.array([r[1] for r in rows.values()])
    e1 = np.cross(a, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    M = np.column_stack([N @ e1, N @ e2, -np.ones(len(N))])
    keep = np.ones(len(N), bool)
    for _ in range(4):
        x = np.linalg.lstsq(M[keep], -Dd[keep], rcond=None)[0]; r = M @ x + Dd
        keep = np.abs(r) < max(15, 2.5 * np.std(r[keep]))
    p = x[0] * e1 + x[1] * e2
    info = {"axis_in_camera": np.round(a, 4).tolist(), "axis_point_mm": np.round(p, 1).tolist(), "axis_height_above_floor_mm": round(float(x[2]), 1),
            "floor_fit_rms_mm": round(float(np.sqrt((r[keep] ** 2).mean())), 1), "floor_planes_used": int(keep.sum()),
            "angle_range_deg": [round(float(th.min()), 1), round(float(th.max()), 1)], "camera_to_axis_mm": round(float(np.linalg.norm(p)), 1)}
    np.savez_compressed(os.path.join(out_dir, "geometry.npz"), frames=fr, theta=th, a=a, p=p, h=x[2], sgn=sgn, nf0=nf0, fixed=fixed)
    json.dump(dict(info, frames=[int(i) for i in fr], theta_deg=[round(float(t), 2) for t in th]), open(os.path.join(out_dir, "geometry.json"), "w"))
    print(json.dumps(info, indent=1))
    return info


def fuse(rec, out_dir, name, frames, box, step=2, cube=False):
    """Every frame's points taken to the room frame; those inside box = (x0, x1, y0, y1, z0, z1) are kept, with their colours.
    Writes NAME.npz (X, C as BGR, T = gantry angle), NAME.ply and NAME.jpg (front, side and top views).
    cube=True: the camera's pose comes from the cube calibration (gantry_cube.py) at the depth-only angle, and the origin
    is the cube's centre."""
    import cv2
    geo = GantryGeometry(os.path.join(out_dir, "geometry.npz"))
    to_room = geo.to_room
    if cube:
        from gantry_cube import CubeGantry
        cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz"))
        from gantry_cube import frame_angles
        ang, src = frame_angles(out_dir)
        print("gantry angle from:", src)
        to_room = lambda P, i: cg.to_room(P, ang(i))
    pts, cols, ths = [], [], []
    for i in frames:
        z, im = rec.depth(i), rec.color(i)
        if z is None or im is None:
            continue
        P, (v, u) = rec.points(z, mask=~geo.fixed, step=step)
        X = to_room(P, i)
        m = (X[:, 0] > box[0]) & (X[:, 0] < box[1]) & (X[:, 1] > box[2]) & (X[:, 1] < box[3]) & (X[:, 2] > box[4]) & (X[:, 2] < box[5])
        pts.append(X[m].astype(np.float32)); cols.append(im[v[m], u[m]]); ths.append(np.full(int(m.sum()), geo.angle(i), np.float32))
    X, C, T = np.vstack(pts), np.vstack(cols), np.concatenate(ths)
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), X=X, C=C, T=T)
    header = "\n".join(["ply", "format binary_little_endian 1.0", "element vertex %d" % len(X), "property float x", "property float y",
                        "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header", ""])
    with open(os.path.join(out_dir, name + ".ply"), "wb") as f:
        f.write(header.encode())
        rows = np.empty(len(X), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rows["p"] = X; rows["c"] = C[:, ::-1]
        f.write(rows.tobytes())
    s = 600 / max(box[1] - box[0], box[3] - box[2], box[5] - box[4])

    def view(ax0, ax1, r0, r1):
        w, h = int((r0[1] - r0[0]) * s), int((r1[1] - r1[0]) * s)
        img = np.zeros((h, w, 3), np.uint8)
        uu = ((X[:, ax0] - r0[0]) * s).astype(int); vv = ((r1[1] - X[:, ax1]) * s).astype(int)
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        img[vv[ok], uu[ok]] = C[ok]
        return img
    views = [view(0, 2, box[0:2], box[4:6]), view(1, 2, box[2:4], box[4:6]), view(0, 1, box[0:2], box[2:4])]
    hh = max(v.shape[0] for v in views)
    cv2.imwrite(os.path.join(out_dir, name + ".jpg"), np.hstack([np.vstack([v, np.zeros((hh - v.shape[0], v.shape[1], 3), np.uint8)]) for v in views]))
    print("%s: %d points from %d frames" % (name, len(X), len(pts)))
    return X, C, T


def flatness(X, T, rng=None, above=None, faces=3):
    """How well a fused object with flat faces (the calibration cube) hangs together: per face, the rms thickness and how
    the face drifts with gantry angle. X: points of the object only."""
    rng = rng or np.random.default_rng(0)
    rest, out = np.arange(len(X)), []
    for _ in range(faces):
        R = X[rest]
        best = None
        for _ in range(400):
            a, b, c = R[rng.integers(0, len(R), 3)]
            n = np.cross(b - a, c - a); ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n /= ln
            m = np.abs((R - a) @ n) < 10
            if best is None or m.sum() > best[0]:
                best = (m.sum(), m)
        m = best[1]; P = R[m]; c = P.mean(0)
        w, v = np.linalg.eigh(np.cov((P - c).T)); n = v[:, 0]
        r = (P - c) @ n; t = T[rest][m]
        drift = {int(a0): round(float(r[(t >= a0) & (t < a0 + 30)].mean()), 1) for a0 in range(0, 180, 30) if ((t >= a0) & (t < a0 + 30)).sum() > 300}
        out.append({"points": int(len(P)), "normal": np.round(n, 2).tolist(), "thickness_rms_mm": round(float(r.std()), 1),
                    "edge_mm": round(float(np.sqrt(12 * w[2])), 0), "offset_by_angle_mm": drift})
        rest = rest[~m]
    ns = [np.array(f["normal"]) for f in out]
    ang = [round(float(np.degrees(np.arccos(min(1.0, abs(ns[i] @ ns[j]))))), 1) for i in range(len(ns)) for j in range(i + 1, len(ns))]
    return {"faces": out, "angles_between_faces_deg": ang}
