"""Simulated study: how should the depth camera correct the RGB model, if a region of interest must be right to under a millimetre?

Everything here is simulated, so the truth is known: a body of known shape, moved by a known amount, seen by a depth camera with
a stated error model, next to an "RGB model" with stated errors. See docs/DEPTH_STUDY.md for the questions, the assumptions
and the reading of the results.

    python tools/depth_study.py [--out docs/depth_study.json] [--trials 8] [--only e1,e2,...]

Frames: OpenCV camera (x right, y down, z forward), millimetres. The room frame is camera 0's frame.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
FAST = os.path.join(HERE, "..", "..", "Fast-SAM-3D-Body")
MHR_PATH = os.path.join(FAST, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")
REST = os.path.join(FAST, "mhr_rest.npz")

W, H = 640, 576                       # Femto Bolt, narrow field of view, unbinned
F = 505.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
KD = (F, F, W / 2, H / 2)
_v, _u = np.mgrid[0:H, 0:W]
RAYS = np.stack([(_u - K[0, 2]) / F, (_v - K[1, 2]) / F], -1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ rendering
def raster(verts, faces, kmax=10, chunk=6000):
    """Z-buffer (mm, inf where empty). Small triangles are filled all at once; the few large ones one by one."""
    z = verts[:, 2]
    uv = np.stack([F * verts[:, 0] / z + K[0, 2], F * verts[:, 1] / z + K[1, 2]], 1)
    p, zf = uv[faces], z[faces]
    x0 = np.maximum(np.floor(p[:, :, 0].min(1)), 0).astype(np.int64); x1 = np.minimum(np.ceil(p[:, :, 0].max(1)), W - 1).astype(np.int64)
    y0 = np.maximum(np.floor(p[:, :, 1].min(1)), 0).astype(np.int64); y1 = np.minimum(np.ceil(p[:, :, 1].max(1)), H - 1).astype(np.int64)
    d = (p[:, 1, 1] - p[:, 2, 1]) * (p[:, 0, 0] - p[:, 2, 0]) + (p[:, 2, 0] - p[:, 1, 0]) * (p[:, 0, 1] - p[:, 2, 1])
    ok = (zf > 1.0).all(1) & (x1 >= x0) & (y1 >= y0) & (np.abs(d) > 1e-9)
    small = ok & (x1 - x0 < kmax) & (y1 - y0 < kmax)
    zbuf = np.full(H * W, np.inf)
    oy, ox = np.mgrid[0:kmax, 0:kmax]
    ox, oy = ox.ravel()[None], oy.ravel()[None]
    ids = np.where(small)[0]
    for s in range(0, len(ids), chunk):
        i = ids[s:s + chunk]
        gx, gy = x0[i, None] + ox, y0[i, None] + oy
        pi, di = p[i], d[i, None]
        a = ((pi[:, 1, 1] - pi[:, 2, 1])[:, None] * (gx - pi[:, 2, 0:1]) + (pi[:, 2, 0] - pi[:, 1, 0])[:, None] * (gy - pi[:, 2, 1:2])) / di
        b = ((pi[:, 2, 1] - pi[:, 0, 1])[:, None] * (gx - pi[:, 2, 0:1]) + (pi[:, 0, 0] - pi[:, 2, 0])[:, None] * (gy - pi[:, 2, 1:2])) / di
        c = 1 - a - b
        m = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9) & (gx <= x1[i, None]) & (gy <= y1[i, None])
        iz = a / zf[i, 0:1] + b / zf[i, 1:2] + c / zf[i, 2:3]
        np.minimum.at(zbuf, (gy * W + gx)[m], (1.0 / iz)[m])
    zbuf = zbuf.reshape(H, W)
    for f in np.where(ok & ~small)[0]:
        pf = p[f]
        gx, gy = np.meshgrid(np.arange(x0[f], x1[f] + 1), np.arange(y0[f], y1[f] + 1))
        a = ((pf[1, 1] - pf[2, 1]) * (gx - pf[2, 0]) + (pf[2, 0] - pf[1, 0]) * (gy - pf[2, 1])) / d[f]
        b = ((pf[2, 1] - pf[0, 1]) * (gx - pf[2, 0]) + (pf[0, 0] - pf[2, 0]) * (gy - pf[2, 1])) / d[f]
        c = 1 - a - b
        zz = np.where((a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9), 1.0 / (a / zf[f, 0] + b / zf[f, 1] + c / zf[f, 2]), np.inf)
        sub = zbuf[y0[f]:y1[f] + 1, x0[f]:x1[f] + 1]
        np.minimum(sub, zz, out=sub)
    return zbuf


class Sensor:
    """What the depth camera adds to the true distance z (mm).

    random    sigma(z) = s0 + s2 * (z / 1 m)^2 per pixel and frame. The defaults (1.0, 0.17) are what this project's Bolt
              showed on the static background of its recordings: 1.4-1.6 mm at 1.5-2 m, 1.8 at 2-2.5 m, 2.3 at 2.5-3 m.
    scale     z * (1 + eps): a wrong distance scale
    offset    + b: a constant offset
    cyclic    + A sin(2 pi z / period): the periodic error of a phase measurement
    Then the camera rounds to whole millimetres (uint16), as the Bolt does."""
    def __init__(self, s0=1.0, s2=0.17, eps=0.0, offset=0.0, cyclic=0.0, period=500.0):
        self.s0, self.s2, self.eps, self.offset, self.cyclic, self.period = s0, s2, eps, offset, cyclic, period

    def sigma(self, z):
        return self.s0 + self.s2 * (z / 1000.0) ** 2

    def frames(self, z, n, rng):
        """n noisy frames (float, already rounded to mm; 0 = nothing) of the clean z-buffer."""
        ok = np.isfinite(z)
        zc = np.where(ok, z, 0.0)
        zs = zc * (1 + self.eps) + self.offset + self.cyclic * np.sin(2 * np.pi * zc / self.period)
        out = np.round(zs[None] + rng.normal(0, 1, (n,) + z.shape) * self.sigma(zc)[None])
        return np.where(ok[None], out, 0.0)


class Camera:
    """Room -> camera: Xc = R Xw + t."""
    def __init__(self, R=np.eye(3), t=np.zeros(3)):
        self.R, self.t = np.asarray(R, float), np.asarray(t, float)

    def to_cam(self, X):
        return X @ self.R.T + self.t

    def to_room(self, X):
        return (X - self.t) @ self.R

    @staticmethod
    def looking_at(target, axis, angle_deg, R_err=None, t_err=None):
        """Camera 0 turned about `axis` through `target` by angle (a second pod at the same distance)."""
        Rw = Rot.from_rotvec(np.radians(angle_deg) * np.asarray(axis, float)).as_matrix()      # where the camera sits, in the room
        R = Rw.T; t = target - R @ target
        if R_err is not None:
            R = R_err @ R; t = R_err @ t + t_err
        return Camera(R, t)


def backproject(depth, mask):
    v, u = np.where(mask)
    d = depth[v, u]
    return np.column_stack([RAYS[v, u, 0] * d, RAYS[v, u, 1] * d, d])


# ------------------------------------------------------------------------------------------------ rigid helpers
def make_T(rotvec_deg, trans, about):
    R = Rot.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = about - R @ about + trans
    return T


def apply(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def kabsch(A, B):
    """Rigid T with T A ~ B."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def shift_error(T_est, T_true, X):
    """How wrong the estimated motion is over the region's points X: (largest, at the centre, rotation in degrees)."""
    e = np.linalg.norm(apply(T_est, X) - apply(T_true, X), axis=1)
    c = X.mean(0)[None]
    return float(e.max()), float(np.linalg.norm(apply(T_est, c) - apply(T_true, c))), float(np.degrees(Rot.from_matrix(T_est[:3, :3] @ T_true[:3, :3].T).magnitude()))


# ------------------------------------------------------------------------------------------------ the reference-surface method
class ReferenceSurface:
    """SGRT-style: the region's surface as the depth camera(s) measured it at reference time, and the rigid motion that takes it
    onto the live measurement. No body model is in the measurement; the model only says which pixels belong to the region."""

    def __init__(self):
        self.P, self.N = [], []

    def add(self, depth_mean, mask, cam, z_typ, margin_mm=45.0, smooth_mm=5.0):
        """depth_mean: reference depth (float mm, averaged over the capture). mask: the region's pixels."""
        px = F / z_typ                                             # pixels per mm
        grown = binary_dilation(mask, iterations=int(np.ceil(margin_mm * px))) & (depth_mean > 0)
        # normals from a smoothed copy (the averaged frame still carries noise that would tilt them)
        sig = max(1.0, smooth_mm * px)
        valid = (depth_mean > 0).astype(float)
        zs = gaussian_filter(depth_mean * valid, sig) / np.maximum(gaussian_filter(valid, sig), 1e-6)
        X = np.dstack([RAYS[..., 0] * zs, RAYS[..., 1] * zs, zs])
        k = max(1, int(round(4.0 * px)))
        du = np.roll(X, -k, 1) - np.roll(X, k, 1); dv = np.roll(X, -k, 0) - np.roll(X, k, 0)
        n = np.cross(du, dv); n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)
        n[n[..., 2] > 0] *= -1
        inner = binary_dilation(~(depth_mean > 0), iterations=k + int(np.ceil(sig)))           # normals near the outline are not trusted
        grown &= ~inner
        v, u = np.where(grown)
        self.P.append(cam.to_room(backproject(depth_mean, grown))); self.N.append(n[v, u] @ cam.R)
        return int(grown.sum())

    def finish(self):
        self.P, self.N = np.vstack(self.P), np.vstack(self.N)
        self.tree = cKDTree(self.P)

    def register(self, Q, iters=15, reach_mm=25.0, cut_mm=6.0, init=None):
        """Q: live points of the region (room frame). init: a first guess of the motion (reference -> live), e.g. what the
        placed mesh says. -> T (4x4) with T(reference) = live, number of points used."""
        c = Q.mean(0)
        R, t = np.eye(3), np.zeros(3)                              # live -> reference, about c
        if init is not None:
            Ti = np.linalg.inv(init)
            R = Ti[:3, :3]; t = R @ c + Ti[:3, 3] - c
        n_used = 0
        for it in range(iters):
            Qt = (Q - c) @ R.T + c + t
            dist, j = self.tree.query(Qt, distance_upper_bound=reach_mm)
            ok = np.isfinite(dist)
            j = np.where(ok, j, 0)
            p, n = self.P[j], self.N[j]
            r = np.einsum("ij,ij->i", n, Qt - p)
            ok &= np.abs(r) < max(cut_mm, 3.0 * np.std(r[ok]) if ok.any() else cut_mm) if it > 2 else ok
            if ok.sum() < 50:
                return None, int(ok.sum())
            a, nn, rr = Qt[ok] - c, n[ok], r[ok]
            w = 1.0 / np.maximum(1.0, np.abs(rr) / 3.0)            # Huber, 3 mm
            J = np.column_stack([np.cross(a, nn), nn])
            Hm = (J * w[:, None]).T @ J; g = -J.T @ (w * rr)
            Hm[np.diag_indices(6)] += 1e-6 * np.trace(Hm) / 6
            dlt = np.linalg.solve(Hm, g)
            dR = Rot.from_rotvec(dlt[:3]).as_matrix()
            R, t = dR @ R, dR @ t + dlt[3:]
            n_used = int(ok.sum())
            if np.linalg.norm(dlt[:3]) < 1e-7 and np.linalg.norm(dlt[3:]) < 1e-4:
                break
        Tinv = np.eye(4); Tinv[:3, :3] = R; Tinv[:3, 3] = c - R @ c + t
        return np.linalg.inv(Tinv), n_used


# ------------------------------------------------------------------------------------------------ the scene
class Scene:
    def __init__(self):
        from bind_mhr_joints import MHR
        from depth.fit import PARTS, BodyFit
        from depth.synth import vertex_normals
        mhr = MHR(MHR_PATH)
        d = np.load(REST)
        self.faces = d["faces"]
        self.fit = BodyFit(self.faces, list(mhr.m.character_torch.skeleton.joint_names), mhr.w_joint, mhr.w_vert, mhr.w)
        self.parts = [p[0] for p in PARTS]
        V = d["verts"].astype(np.float64) * 1000.0
        self.model_rest = V                                        # the model's idea of this person's shape
        # the person's true shape: the model's, displaced along the normal by a smooth field (build, clothing): rms `shape_mm`
        n = vertex_normals(V, self.faces)
        g = np.random.default_rng(7)
        f = np.zeros(len(V))
        for _ in range(6):
            k = g.normal(0, 1, 3) * 2 * np.pi / g.uniform(250, 600); f += np.sin(V @ k + g.uniform(0, 6.28))
        self.shape_field = n * (f / f.std())[:, None]
        # sternum: the front of the chest on the midline (the body faces -z, towards the camera)
        cand = np.where((np.abs(V[:, 0]) < 25) & (np.abs(V[:, 1] + 1320) < 25))[0]
        self.sternum = int(cand[np.argmin(V[cand, 2])])
        self.chest_centre = V[self.fit.W[:, self.parts.index("chest")] > 0.5].mean(0)

    def region(self, radius_mm=120.0, offset_mm=(0.0, 0.0)):
        """Vertices of the front of the chest within radius of the sternum (optionally of a point offset from it)."""
        V = self.model_rest
        c = V[self.sternum] + np.array([offset_mm[0], offset_mm[1], 0.0])
        front = V[:, 2] < self.chest_centre[2]
        return np.where(front & (np.linalg.norm(V[:, :2] - c[:2], axis=1) < radius_mm) & (np.abs(V[:, 1] - c[1]) < radius_mm))[0]

    def home(self, V, dist_mm, tilt_deg):
        """Put the rest body in front of camera 0: the sternum on the axis at dist, the body tilted about the left-right axis
        (a camera that looks down along the body, as a ceiling pod does: the region then spans a range of depths)."""
        c = self.model_rest[self.sternum]
        R = Rot.from_euler("x", tilt_deg, degrees=True).as_matrix()
        return (V - c) @ R.T + np.array([0.0, 0.0, dist_mm])

    def truth_rest(self, shape_mm):
        return self.model_rest + shape_mm * self.shape_field

    def limbs(self, V, arms_deg=0.0, head_deg=0.0):
        """The rest body with both arms swung forward and the head turned (parts outside the chest region)."""
        R0 = self.model_rest
        for name, axis, deg in (("uparm_l", "y", arms_deg), ("uparm_r", "y", -arms_deg), ("head", "y", head_deg)):
            if not deg:
                continue
            g = self.parts.index(name)
            w = self.fit.Wc[:, g:g + 1]
            own = np.where(self.fit.W[:, g] > 0.5)[0]
            piv = R0[own[np.argmin(np.abs(R0[own, 0]))]] if name != "head" else R0[own[np.argmax(R0[own, 1])]]
            V = (1 - w) * V + w * ((V - piv) @ Rot.from_euler(axis, deg, degrees=True).as_matrix().T + piv)
        return V

    def model_output(self, T_true, dist_mm, tilt_deg, rng, size=1.15, lean_deg=0.0, arms_deg=0.0, jitter=True, true_arms_deg=0.0,
                     true_head_deg=0.0):
        """What the RGB model says for the body moved by T_true: its own shape, a little wrong in lean and limbs, too far away
        by `size` along every ray (a picture cannot tell size from distance), and trembling from frame to frame."""
        V = self.limbs(self.model_rest, arms_deg + true_arms_deg, true_head_deg)
        X = apply(T_true, self.home(V, dist_mm, tilt_deg))
        c = apply(T_true, self.home(self.chest_centre[None], dist_mm, tilt_deg))[0]
        if lean_deg:
            X = (X - c) @ Rot.from_euler("x", lean_deg, degrees=True).as_matrix().T + c
        s = size
        if jitter:
            s *= 1 + rng.normal(0, 0.003)
            X = X + np.array([rng.normal(0, 1.5), rng.normal(0, 1.5), 0.0]) * dist_mm / 1500.0      # about a pixel of the picture
            X = (X - c) @ Rot.from_rotvec(np.radians(rng.normal(0, 0.4, 3))).as_matrix().T + c
        return X * s

    def region_mask(self, X_roi, z_typ):
        """Pixels of the region, from where its (placed) model vertices fall in the depth image."""
        u = np.round(F * X_roi[:, 0] / X_roi[:, 2] + K[0, 2]).astype(int); v = np.round(F * X_roi[:, 1] / X_roi[:, 2] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        m = np.zeros((H, W), bool); m[v[ok], u[ok]] = True
        k = int(np.ceil(10.0 * F / z_typ)) + 1                    # vertices are up to a centimetre apart
        pad = np.pad(m, k)
        return binary_fill_holes(binary_closing(pad, iterations=k))[k:-k, k:-k]


def frame_of(depth_u16):
    from depth.fit import DepthFrame
    return DepthFrame(depth_u16.astype(np.uint16), RAYS, np.eye(3), np.zeros(3), KD, np.zeros(8))


def stats(x):
    x = np.asarray(x, float)
    return {"median": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3), "n": int(len(x))}


def random_shift(rng, mm=5.0, deg=1.5):
    return rng.uniform(-deg, deg, 3), rng.uniform(-mm, mm, 3)


# ------------------------------------------------------------------------------------------------ experiments
def e1_absolute(sc, trials):
    """Where does the fitted mesh end up, in 3-D, against the true body? By kind of model error and by fit stage."""
    out = {}
    dist, tilt = 1500.0, 30.0
    roi = sc.region()
    cases = [("size only", dict()), ("+ lean 5 deg", dict(lean_deg=5.0)), ("+ shape 8 mm", dict(lean_deg=5.0, shape=8.0)),
             ("+ arms 8 deg", dict(lean_deg=5.0, shape=8.0, arms_deg=8.0))]
    sensor = Sensor()
    for name, kw in cases:
        rows = {k: {"roi": [], "body": []} for k in ("one scale", "scale + rigid", "+ surface")}
        for trial in range(trials):
            rng = np.random.default_rng(100 + trial)
            Xt = sc.home(sc.truth_rest(kw.get("shape", 0.0)), dist, tilt)
            depth = sensor.frames(raster(Xt, sc.faces), 1, rng)[0]
            fr = frame_of(depth)
            Vm = sc.model_output(np.eye(4), dist, tilt, rng, lean_deg=kw.get("lean_deg", 0.0), arms_deg=kw.get("arms_deg", 0.0), jitter=False)
            seen = sc.fit.observe(Xt, fr)["ok"]
            # one scale about the camera
            a = sc.fit.part_scales(Vm, fr)[1]
            X1 = Vm * a["global_scale"]
            tr = {}
            X3, M, info = sc.fit.fit(Vm, fr, pose_iters=4, pose="rigid", track=tr)
            X2 = tr.get("X_pose", X3)
            for key, X in (("one scale", X1), ("scale + rigid", X2), ("+ surface", X3)):
                e = np.linalg.norm(X - Xt, axis=1)
                r = np.zeros(len(Xt), bool); r[roi] = True
                rows[key]["roi"].append(np.median(e[seen & r])); rows[key]["body"].append(np.median(e[seen]))
        out[name] = {k: {"roi_mm": round(float(np.mean(v["roi"])), 2), "body_mm": round(float(np.mean(v["body"])), 2)} for k, v in rows.items()}
        print("e1", name, out[name], flush=True)
    return out


def measure_shift(sc, dist, tilt, rng, sensor, T_true, roi, cams=None, model_kw=None, ref_frames=30, live_frames=(1,), shape=8.0,
                  live_roi=None, do_mesh=True, cam_err=None, others_move=(6.0, 5.0)):
    """One trial: reference at home, then the body moved by T_true. -> {method: (max, centre, rot)} over the region."""
    model_kw = model_kw or dict(lean_deg=3.0)
    cams = cams or [Camera()]
    seen_cams = cam_err or cams                                    # what the software believes the cameras' poses are
    # Between the reference and now the region moves rigidly by T_true; the rest of the person does what people do: the arms
    # swing a few degrees and the head turns. The RGB model sees that (with its own errors).
    arms, head = others_move
    X0 = sc.home(sc.truth_rest(shape), dist, tilt)
    X1 = apply(T_true, sc.home(sc.limbs(sc.truth_rest(shape), arms, head), dist, tilt))
    res, fits, masks, depths = {}, [], [], []
    ref = ReferenceSurface()
    live_roi = roi if live_roi is None else live_roi
    for step, (Xt, n_frames) in enumerate(((X0, ref_frames), (X1, max(live_frames)))):
        per_cam = []
        Vm = sc.model_output(np.eye(4) if step == 0 else T_true, dist, tilt, rng, true_arms_deg=arms * step, true_head_deg=head * step, **model_kw)
        for ci, cam in enumerate(cams):
            fr = sensor.frames(raster(cam.to_cam(Xt), sc.faces), n_frames, rng)
            per_cam.append(fr)
            if ci == 0:
                tr = {}
                Xs, M, info = sc.fit.fit(Vm, frame_of(fr[0]), pose_iters=4, pose="rigid", track=tr)
                fits.append((tr.get("X_pose", Xs), Xs, sc.fit.observe(Xs, frame_of(fr[0]))["ok"]))
        depths.append(per_cam)
    Xp0, Xs0, ok0 = fits[0]; Xp1, Xs1, ok1 = fits[1]
    truth_roi = X0[roi]
    T_mesh = kabsch(Xp0[roi], Xp1[roi])                           # the mesh's answer: also the first guess for the registration
    if do_mesh:
        res["mesh, placed as one"] = shift_error(T_mesh, T_true, truth_roi)
        both = np.intersect1d(roi[ok0[roi]], roi[ok1[roi]])
        if len(both) > 20:
            res["mesh + surface layer"] = shift_error(kabsch(Xs0[both], Xs1[both]), T_true, truth_roi)
    # reference surface: the region's pixels come from the placed mesh (in each camera), the measurement from the depth alone
    for ci, cam in enumerate(cams):
        believed = seen_cams[ci]
        m0 = sc.region_mask(believed.to_cam(Xp0[roi]), dist)
        ref.add(depths[0][ci].mean(0) * (depths[0][ci] > 0).all(0), m0, believed, dist)
    ref.finish()
    for nl in live_frames:
        Q = []
        for ci, cam in enumerate(cams):
            believed = seen_cams[ci]
            fr = depths[1][ci][:nl]
            d = fr.mean(0) * (fr > 0).all(0)
            m1 = sc.region_mask(believed.to_cam(Xp1[live_roi]), dist) & (d > 0)
            m1 &= ~binary_dilation(d == 0, iterations=2)
            Q.append(believed.to_room(backproject(d, m1)))
        t0 = time.perf_counter()
        T, n = ref.register(np.vstack(Q), init=T_mesh)
        ms = (time.perf_counter() - t0) * 1000
        if T is not None:
            res["reference surface, %d frame%s" % (nl, "" if nl == 1 else "s")] = shift_error(T, T_true, truth_roi) + (ms, n)
    return res


def collect(acc, res):
    for k, v in res.items():
        acc.setdefault(k, []).append(v)


def summarise(acc):
    out = {}
    for k, v in acc.items():
        v = np.array([x[:3] for x in v]); extra = [x[3:] for x in acc[k] if len(x) > 3]
        out[k] = {"max_mm": stats(v[:, 0]), "centre_mm": stats(v[:, 1]), "rot_deg": stats(v[:, 2])}
        if extra:
            out[k]["ms"] = round(float(np.median([e[0] for e in extra])), 1); out[k]["points"] = int(np.median([e[1] for e in extra]))
    return out


def e2_distance(sc, trials):
    """The clinical number: the region's shift from its reference, at each distance."""
    out = {}
    roi = sc.region()
    for dist in (600.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0):
        acc = {}
        for trial in range(trials):
            rng = np.random.default_rng(200 + trial)
            rv, tv = random_shift(rng)
            X0 = sc.home(sc.truth_rest(8.0), dist, 30.0)
            T = make_T(rv, tv, X0[roi].mean(0))
            collect(acc, measure_shift(sc, dist, 30.0, rng, Sensor(), T, roi, live_frames=(1, 3)))
        out[str(int(dist))] = summarise(acc)
        print("e2", dist, {k: (v["max_mm"]["median"], v["max_mm"]["p95"]) for k, v in out[str(int(dist))].items()}, flush=True)
    return out


def e3_systematic(sc, trials):
    """A camera whose distances are systematically wrong: absolute position against shift from a reference."""
    out = {}
    roi = sc.region()
    dist = 1500.0
    sensors = {"none": Sensor(), "scale 0.2%": Sensor(eps=0.002), "offset 5 mm": Sensor(offset=5.0),
               "cyclic 1.5 mm / 0.5 m": Sensor(cyclic=1.5), "all three": Sensor(eps=0.002, offset=5.0, cyclic=1.5)}
    for sname, sensor in sensors.items():
        row = {}
        # absolute: how far the measured surface is from the true one (noise averaged out)
        X0 = sc.home(sc.truth_rest(8.0), dist, 30.0)
        z = raster(X0, sc.faces)
        m = sc.region_mask(X0[roi], dist) & np.isfinite(z)
        fr = sensor.frames(z, 30, np.random.default_rng(1)).mean(0)
        row["absolute_mm"] = round(float(np.abs(fr[m] - z[m]).mean()), 2)
        for mag in (2.0, 10.0, 30.0, 100.0):
            acc = {}
            for trial in range(trials):
                rng = np.random.default_rng(300 + trial)
                dirn = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0]) * (1 if trial % 2 else -1)
                T = make_T(rng.uniform(-0.5, 0.5, 3), dirn * mag, X0[roi].mean(0))
                collect(acc, measure_shift(sc, dist, 30.0, rng, sensor, T, roi, live_frames=(3,), do_mesh=False))
            s = summarise(acc)
            row["shift %d mm" % mag] = s.get("reference surface, 3 frames", {}).get("max_mm")
        out[sname] = row
        print("e3", sname, row, flush=True)
    return out


def e4_region(sc, trials):
    """Does the region have to be found exactly (is a depth-based pose needed)? And how large must it be?"""
    out = {"offset": {}, "size": {}, "pose error": {}}
    dist = 1500.0
    base = sc.region()

    def run(roi, live_roi=None, model_kw=None, seed=400):
        acc = {}
        for trial in range(trials):
            rng = np.random.default_rng(seed + trial)
            rv, tv = random_shift(rng)
            X0 = sc.home(sc.truth_rest(8.0), dist, 30.0)
            T = make_T(rv, tv, X0[base].mean(0))
            r = measure_shift(sc, dist, 30.0, rng, Sensor(), T, roi, live_frames=(3,), do_mesh=False, live_roi=live_roi, model_kw=model_kw)
            # always judged over the same (base) region's points, so sizes can be compared
            collect(acc, r)
        return summarise(acc).get("reference surface, 3 frames")

    for off in (0.0, 10.0, 20.0, 40.0):
        out["offset"]["%d mm" % off] = run(base, live_roi=sc.region(offset_mm=(off * 0.6, off * 0.8)))
        print("e4 offset", off, out["offset"]["%d mm" % off]["max_mm"], flush=True)
    for rad in (40.0, 70.0, 120.0, 170.0):
        out["size"]["%d mm" % rad] = run(sc.region(radius_mm=rad))
        print("e4 size", rad, out["size"]["%d mm" % rad]["max_mm"], out["size"]["%d mm" % rad]["rot_deg"], flush=True)
    for lean in (0.0, 3.0, 8.0, 15.0):
        out["pose error"]["lean %d deg" % lean] = run(base, model_kw=dict(lean_deg=lean, arms_deg=lean))
        print("e4 lean", lean, out["pose error"]["lean %d deg" % lean]["max_mm"], flush=True)
    return out


def e5_cameras(sc, trials):
    """One camera or three pods: what the geometry of the cameras buys."""
    out = {}
    dist = 1500.0
    roi = sc.region()
    X0 = sc.home(sc.truth_rest(8.0), dist, 0.0)
    target = X0[roi].mean(0)
    long_axis = np.array([0.0, 1.0, 0.0])
    eg = np.random.default_rng(5)

    def off(mm, deg):
        return Rot.from_rotvec(np.radians(eg.normal(0, deg, 3))).as_matrix(), eg.normal(0, mm, 3)

    pods = [Camera(), Camera.looking_at(target, long_axis, 50.0), Camera.looking_at(target, long_axis, -50.0)]
    wrong_a = [Camera()] + [Camera.looking_at(target, long_axis, a, *off(0.3, 0.03)) for a in (50.0, -50.0)]
    wrong = [Camera()] + [Camera.looking_at(target, long_axis, a, *off(1.0, 0.1)) for a in (50.0, -50.0)]
    setups = {"one camera, in front": ([Camera()], None),
              "one camera, 50 deg to the side": ([pods[1]], None),
              "three pods (0, +50, -50 deg)": (pods, None),
              "three pods, poses known to 0.3 mm / 0.03 deg": (pods, wrong_a),
              "three pods, poses known to 1 mm / 0.1 deg": (pods, wrong)}
    for name, (cams, believed) in setups.items():
        acc = {}
        for trial in range(trials):
            rng = np.random.default_rng(500 + trial)
            rv, tv = random_shift(rng)
            T = make_T(rv, tv, target)
            if cams[0].t.any() or not np.allclose(cams[0].R, np.eye(3)):
                # the model works on camera 0's picture; for a single side camera, look at the scene from that camera
                Rc, tc = cams[0].R, cams[0].t
                Tc = np.eye(4); Tc[:3, :3] = Rc; Tc[:3, 3] = tc
                Tt = Tc @ T @ np.linalg.inv(Tc)
                r = measure_shift_view(sc, dist, rng, Tt, roi, Tc)
            else:
                r = measure_shift(sc, dist, 0.0, rng, Sensor(), T, roi, cams=cams, live_frames=(3,), do_mesh=False, cam_err=believed)
            collect(acc, r)
        out[name] = summarise(acc).get("reference surface, 3 frames")
        print("e5", name, {k: out[name][k] for k in ("max_mm", "centre_mm", "rot_deg")}, flush=True)
    return out


def measure_shift_view(sc, dist, rng, T_true, roi, Tc):
    """measure_shift for one camera that is not in front: the whole scene expressed in that camera's frame."""
    home = sc.home

    def home_side(V, d, tilt):
        return apply(Tc, home(V, d, tilt))
    sc.home = home_side
    try:
        return measure_shift(sc, dist, 0.0, rng, Sensor(), T_true, roi, live_frames=(3,), do_mesh=False)
    finally:
        sc.home = home


def e6_scale_only_vs_extent(sc, trials):
    """One number for the whole body: how its error over the region grows with the depth the region spans, when the model's
    lean is a little wrong."""
    out = {}
    roi = sc.region(radius_mm=170.0)
    sensor = Sensor()
    for tilt in (0.0, 20.0, 40.0, 60.0):
        row = {}
        for lean in (0.0, 3.0, 6.0):
            e1, e2, ext = [], [], []
            for trial in range(max(2, trials // 2)):
                rng = np.random.default_rng(600 + trial)
                Xt = sc.home(sc.truth_rest(0.0), 1500.0, tilt)
                fr = frame_of(sensor.frames(raster(Xt, sc.faces), 1, rng)[0])
                Vm = sc.model_output(np.eye(4), 1500.0, tilt, rng, lean_deg=lean, jitter=False)
                seen = sc.fit.observe(Xt, fr)["ok"][roi]
                a = sc.fit.part_scales(Vm, fr)[1]
                X1 = Vm * a["global_scale"]
                tr = {}
                sc.fit.fit(Vm, fr, pose_iters=4, pose="rigid", track=tr, surface=False)
                e1.append(np.percentile(np.linalg.norm(X1 - Xt, axis=1)[roi][seen], 95))
                e2.append(np.percentile(np.linalg.norm(tr["X_pose"] - Xt, axis=1)[roi][seen], 95))
                ext.append(np.ptp(Xt[roi][seen][:, 2]))
            row["lean %d deg" % lean] = {"one_scale_p95_mm": round(float(np.mean(e1)), 2), "scale_rigid_p95_mm": round(float(np.mean(e2)), 2)}
            row["depth_extent_mm"] = round(float(np.mean(ext)))
        out["tilt %d deg" % tilt] = row
        print("e6", tilt, row, flush=True)
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    d = sorted(res["e2"], key=float)
    for name, style in (("mesh, placed as one", "o-"), ("mesh + surface layer", "s-"), ("reference surface, 1 frame", "^-"), ("reference surface, 3 frames", "v-")):
        md = [res["e2"][k][name]["max_mm"]["median"] for k in d]; hi = [res["e2"][k][name]["max_mm"]["p95"] for k in d]
        x = [float(k) / 1000 for k in d]
        line, = ax[0].plot(x, md, style, label=name); ax[0].plot(x, hi, ":", color=line.get_color())
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--"); ax[0].set_xlabel("distance from the camera (m)"); ax[0].set_ylabel("largest error of the shift over the region (mm)")
    ax[0].set_title("Shift of a chest region from its reference\n(solid: median, dotted: 95th percentile)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 3)
    mags = [2, 10, 30, 100]
    for name, row in res["e3"].items():
        ax[1].plot(mags, [row["shift %d mm" % m]["median"] for m in mags], "o-", label="%s (surface off by %.1f mm)" % (name, row["absolute_mm"]))
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--"); ax[1].set_xscale("log"); ax[1].set_xlabel("how far the body is from its reference position (mm)")
    ax[1].set_ylabel("largest error of the shift over the region (mm)"); ax[1].set_title("A camera whose distances are systematically wrong\n(reference surface, 1.5 m)"); ax[1].legend(fontsize=8)
    sz = res["e4"]["size"]
    x = [2 * float(k.split()[0]) for k in sz]
    ax[2].plot(x, [v["max_mm"]["median"] for v in sz.values()], "o-", label="median"); ax[2].plot(x, [v["max_mm"]["p95"] for v in sz.values()], "o:", label="95th percentile")
    ax[2].axhline(1.0, color="k", lw=0.8, ls="--"); ax[2].set_xlabel("width of the region (mm)"); ax[2].set_ylabel("largest error of the shift over the region (mm)")
    ax[2].set_title("How large the region must be\n(reference surface, 1.5 m, 3 frames)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "depth_study.json"))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--figure", default="", help="only draw the figure from the saved results, to this file")
    a = ap.parse_args()
    if a.figure:
        figure(json.load(open(a.out)), a.figure)
        return
    sc = Scene()
    todo = {"e1": e1_absolute, "e6": e6_scale_only_vs_extent, "e2": e2_distance, "e3": e3_systematic, "e4": e4_region, "e5": e5_cameras}
    only = [x for x in a.only.split(",") if x]
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
    for k, fn in todo.items():
        if only and k not in only:
            continue
        t0 = time.time()
        res[k] = fn(sc, a.trials)
        print(k, "done in %.0f s" % (time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
