"""Gantry calibration from the aprilcube target on the couch.

The cube does not move while the gantry turns, so its pose in every picture gives the camera's pose in the cube's frame.
Those poses lie on a circle about the gantry's axis. Fitting that circle gives, in a frame fixed to a physical object:

  the axis (direction and position), the gantry angle of every frame, and the camera's pose at angle 0;
  how far the real camera path departs from a perfect rotation (sag, wobble), as a table against angle;
  a check of the depth-only calibration (gantry_geometry.py), whose angles the person's frames have to use, because
  the cube is not in the picture then.

Cube frame = room frame from here on: origin at the cube's centre, axes along its faces (see aprilcube_target.py).
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from aprilcube_target import Cube, CubeFinder


def track(rec, cube, frames, min_tags=4, max_err=2.0):
    """Cube pose per frame: {frame: dict(R, t, err, n, faces)} for frames with a trustworthy pose."""
    fd = CubeFinder(cube, rec.K)
    out, guess = {}, None
    for i in frames:
        im = rec.color(i)
        if im is None:
            continue
        ids, corners = fd.detect(im)
        r = fd.solve(ids, corners, guess) or fd.solve(ids, corners)
        if r is None or r["n"] < min_tags or r["err"] > max_err:
            guess = None
            continue
        out[int(i)] = r; guess = (r["R"], r["t"])
    return out


def refine_with_depth(rec, cube, i, R, t, fd=None, mask=None, sigma_mm=1.5, sigma_px=0.5, rounds=2):
    """Cube pose from the markers AND the depth. The markers fix how the cube is turned and where it is across the picture;
    the depth fixes how far away it is, which a 105 mm target seen from 0.8 m leaves weak. The orientation is NOT refined
    with the depth: on faces this small and this oblique the depth camera's planes are tilted by a few degrees, and letting
    them turn the cube put the gantry angle 5 degrees out (the room's planes and the markers agree with each other).
    Returns dict(R, t, depth_rms mm, px, points) or None."""
    import cv2
    from scipy.optimize import least_squares
    fd = fd or CubeFinder(cube, rec.K)
    ids, corners = fd.detect(rec.color(i))
    if len(ids) < 2:
        return None
    obj = np.concatenate([cube.corners[int(k)] for k in ids]); img = corners.reshape(-1, 2).astype(np.float64)
    P, _ = rec.points(rec.depth(i), mask=mask, step=1)
    P = P[np.linalg.norm(P - t, axis=1) < cube.dims.max() + 40]
    h = cube.dims / 2
    faces = cube.faces_mm()
    x = np.asarray(t, np.float64).copy(); Rm = np.asarray(R, np.float64)
    rng = np.random.default_rng(0)
    for rd in range(rounds):
        X = (P - x) @ Rm
        sel = []
        for name, cen, n in faces:
            ax = int(np.argmax(np.abs(n))); o = [a for a in range(3) if a != ax]
            d = (X - cen) @ n
            m = np.where((np.abs(d) < (16 if rd == 0 else 6)) & (np.abs(X[:, o[0]]) < h[o[0]] - 10) & (np.abs(X[:, o[1]]) < h[o[1]] - 10))[0]
            if len(m) < 200:
                continue
            if len(m) > 500:
                m = rng.choice(m, 500, replace=False)
            sel.append((P[m], cen, n))
        if not sel:
            return None
        Pd = np.concatenate([s[0] for s in sel]); Cn = np.concatenate([np.tile(s[1], (len(s[0]), 1)) for s in sel])
        Nn = np.concatenate([np.tile(s[2], (len(s[0]), 1)) for s in sel])
        wd = np.sqrt(len(img) * 2 / len(Pd)) * 2.0                # depth and markers weigh about the same overall

        def res(x):
            d = np.einsum("ni,ni->n", (Pd - x) @ Rm - Cn, Nn)
            q = obj @ Rm.T + x
            uv = np.c_[rec.K[0, 0] * q[:, 0] / q[:, 2] + rec.K[0, 2], rec.K[1, 1] * q[:, 1] / q[:, 2] + rec.K[1, 2]]
            return np.concatenate([wd * d / sigma_mm, ((uv - img) / sigma_px).ravel()])
        x = least_squares(res, x, loss="soft_l1", f_scale=2.0).x
    d = np.einsum("ni,ni->n", (Pd - x) @ Rm - Cn, Nn)
    q = obj @ Rm.T + x
    uv = np.c_[rec.K[0, 0] * q[:, 0] / q[:, 2] + rec.K[0, 2], rec.K[1, 1] * q[:, 1] / q[:, 2] + rec.K[1, 2]]
    return dict(R=Rm, t=x, depth_rms=float(np.sqrt((d ** 2).mean())), depth_bias=float(np.median(d)),
                err=float(np.linalg.norm(uv - img, axis=1).mean()), n=len(ids), depth_faces=len(sel),
                faces=sorted({cube.face_of[int(k)] for k in ids}))


class CubeGantry:
    """Camera pose in the cube frame as a function of gantry angle: X_cube = Rc(theta) P_cam + C(theta)."""

    def __init__(self, path):
        g = np.load(path)
        self.a, self.c, self.Rref, self.Cref = g["a"], g["c"], g["Rref"], g["Cref"]
        self.frames, self.theta = g["frames"], g["theta"]          # cube-session frames with the cube's own angle
        self.tab_t, self.tab_dC, self.tab_dr = g["tab_t"], g["tab_dC"], g["tab_dr"]   # measured departure from a pure rotation
        self.plane_gain, self.plane_off = float(g["plane_gain"]), float(g["plane_off"])
        self.room_R = g["room_R"]                                  # cube frame -> room axes (Y along the axis, Z up), origin kept

    def to_room(self, P, theta, table=True):
        """Camera points -> room axes with the origin at the cube's centre."""
        return self.to_cube(P, theta, table) @ self.room_R.T

    def pose(self, theta, table=True):
        Rt = Rot.from_rotvec(np.radians(theta) * self.a)
        Rc = Rt.as_matrix() @ self.Rref; C = self.c + Rt.apply(self.Cref - self.c)
        if table and len(self.tab_t) > 1:
            dr = np.array([np.interp(theta, self.tab_t, self.tab_dr[:, k]) for k in range(3)])
            dC = np.array([np.interp(theta, self.tab_t, self.tab_dC[:, k]) for k in range(3)])
            Rc = Rot.from_rotvec(dr).as_matrix() @ Rc; C = C + dC
        return Rc, C

    def to_cube(self, P, theta, table=True):
        Rc, C = self.pose(theta, table)
        return P @ Rc.T + C

    def angle_from_planes(self, plane_angle):
        """The depth-only angle of gantry_geometry, mapped to this calibration's angle."""
        return self.plane_gain * plane_angle + self.plane_off


def frame_angles(out_dir, source="best"):
    """frame -> gantry angle in the cube calibration's scale. The linac's own angle if a console video was matched to the
    recording (gantry_console.py sync), else the depth-only angle mapped onto the cube's. source: best | linac | planes."""
    from gantry_geometry import GantryGeometry
    lp = os.path.join(out_dir, "linac_angle.npz")
    if source != "planes" and os.path.exists(lp):
        L = np.load(lp)
        if "cube_theta" in L:
            th = L["cube_theta"]
            return (lambda i: float(th[int(i)])), "linac"
    if source == "linac":
        raise FileNotFoundError("no linac angle: run gantry_console.py sync")
    geo = GantryGeometry(os.path.join(out_dir, "geometry.npz")); cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz"))
    return (lambda i: cg.angle_from_planes(geo.angle(i))), "planes"


def fit_rotation(Rc, C, iters=8):
    """Poses (Rc_i, C_i) that should be one pose turned about a fixed axis: axis a, point c, reference pose, angles."""
    ref = len(Rc) // 2
    Rref = Rc[ref].copy()
    for _ in range(iters):
        rv = Rot.from_matrix(Rc @ Rref.T).as_rotvec()
        big = np.linalg.norm(rv, axis=1) > np.radians(8)
        w, v = np.linalg.eigh(rv[big].T @ rv[big]); a = v[:, 2]
        if (rv[big] @ a).sum() < 0:
            a = -a
        th = rv @ a
        Rt = Rot.from_rotvec(th[:, None] * a).as_matrix()
        M = np.einsum("nji,njk->ik", Rt, Rc)                      # mean of Rt_i^T Rc_i
        u, s, vt = np.linalg.svd(M); Rref = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    A = np.concatenate([np.concatenate([np.eye(3) - Rt[i], Rt[i]], 1) for i in range(len(Rc))])
    x = np.linalg.lstsq(A, C.ravel(), rcond=None)[0]
    c, Cref = x[:3], x[3:]
    return a, c, Rref, Cref, np.degrees(th)


def calibrate(rec, out_dir, frames, cube=None, use_depth=True):
    from gantry_geometry import GantryGeometry
    cube = cube or Cube()
    os.makedirs(out_dir, exist_ok=True)
    tr = track(rec, cube, frames)
    moved = []
    if use_depth:                                                 # the markers leave the distance weak; the depth fixes it
        fd = CubeFinder(cube, rec.K)
        gp = os.path.join(out_dir, "geometry.npz")
        mask = ~GantryGeometry(gp).fixed if os.path.exists(gp) else None
        for i in sorted(tr):
            r = refine_with_depth(rec, cube, i, tr[i]["R"], tr[i]["t"], fd, mask)
            if r is None or r["depth_faces"] < 2:
                del tr[i]
                continue
            moved.append(np.linalg.norm(r["t"] - tr[i]["t"])); r["markers_only_err"] = tr[i]["err"]; tr[i] = r
    fr = np.array(sorted(tr))
    good = np.array([len(tr[i]["faces"]) >= 2 for i in fr])       # one face alone leaves the tilt weak
    R = np.array([tr[i]["R"] for i in fr]); t = np.array([tr[i]["t"] for i in fr])
    Rc = np.transpose(R, (0, 2, 1)); C = -np.einsum("nij,nj->ni", Rc, t)
    a, c, Rref, Cref, _ = fit_rotation(Rc[good], C[good])
    # angle 0 = the gantry's position at the first frame; all frames' angles from their own rotation
    th = np.degrees(Rot.from_matrix(Rc @ Rref.T).as_rotvec() @ a)
    th0 = float(np.median(th[fr < fr[0] + 200])); th -= th0
    Rz = Rot.from_rotvec(np.radians(th0) * a)
    Rref = Rz.as_matrix() @ Rref; Cref = c + Rz.apply(Cref - c)
    if th[np.argmax(np.abs(th))] < 0:                             # count the sweep as positive
        a, th = -a, -th
    Rt = Rot.from_rotvec(np.radians(th)[:, None] * a)
    Cm = c + Rt.apply(Cref - c); Rm = Rt.as_matrix() @ Rref
    dC = C - Cm; dr = Rot.from_matrix(Rc @ np.transpose(Rm, (0, 2, 1))).as_rotvec()
    # the departure from a pure rotation, as a table against angle (medians over 10-degree bins of the good frames)
    edges = np.arange(np.floor(th.min() / 10) * 10, th.max() + 10, 10)
    tab = [(float(np.median(th[m])), np.median(dC[m], 0), np.median(dr[m], 0)) for lo in edges
           for m in [good & (th >= lo) & (th < lo + 10)] if m.sum() >= 3]
    tab_t = np.array([x[0] for x in tab]); tab_dC = np.array([x[1] for x in tab]); tab_dr = np.array([x[2] for x in tab])
    # the depth-only calibration, on the same frames
    info = {}
    gp = os.path.join(out_dir, "geometry.npz")
    gain, off = 1.0, 0.0
    room_R = np.eye(3)
    if os.path.exists(gp):
        geo = GantryGeometry(gp)
        y = a if (Rref @ geo.a) @ a > 0 else -a                    # room axes as in the depth-only calibration
        z = Rref @ geo.nf0; z -= y * (y @ z); z /= np.linalg.norm(z)
        room_R = np.stack([np.cross(y, z), y, z])
        first = [k for k in range(len(fr)) if fr[k] < fr[0] + 200]
        info["cube_centre_in_depth_only_room_frame_mm"] = np.round(np.median([geo.to_room(t[k][None], int(fr[k]))[0] for k in first], 0), 1).tolist()
        tp = np.array([geo.angle(i) for i in fr])
        gain, off = np.polyfit(tp[good], th[good], 1)
        res = th - (gain * tp + off)
        a_cam = np.median(np.einsum("nij,j->ni", R, a), 0); a_cam /= np.linalg.norm(a_cam)
        p_cam = np.median(np.einsum("nij,j->ni", R, c) + t, 0)    # a point of the axis, seen from the camera
        if a_cam @ geo.a < 0:
            a_cam = -a_cam
        off_ax = (p_cam - geo.p) - geo.a * ((p_cam - geo.p) @ geo.a)
        info["depth_only_check"] = {
            "axis_direction_differs_deg": round(float(np.degrees(np.arccos(min(1.0, abs(a_cam @ geo.a))))), 2),
            "axis_position_differs_mm": round(float(np.linalg.norm(off_ax)), 1),
            "angle_gain": round(float(gain), 4), "angle_offset_deg": round(float(off), 2),
            "angle_residual_deg_rms": round(float(np.sqrt(np.mean(res[good] ** 2))), 2),
            "angle_residual_deg_max": round(float(np.abs(res[good]).max()), 2)}
    dist_axis = np.linalg.norm(c - a * (c @ a))                   # cube centre (origin) to the axis
    Rt0 = Rot.from_rotvec(np.radians(th)[:, None] * a)
    iso0 = np.einsum("nij,nj->ni", Rt0.as_matrix() @ Rref, t) + c + Rt0.apply(Cref - c)   # where a pure rotation puts the cube's centre
    if use_depth:
        info["markers_and_depth"] = {"cube_moved_from_markers_only_mm_median": round(float(np.median(moved)), 1),
                                     "depth_on_faces_rms_mm_median": round(float(np.median([tr[i]["depth_rms"] for i in fr])), 2),
                                     "markers_px_median": round(float(np.median([tr[i]["err"] for i in fr])), 2),
                                     "markers_px_markers_only": round(float(np.median([tr[i]["markers_only_err"] for i in fr])), 2)}
    info.update({
        "cube": cube.config(), "frames_with_pose": int(len(fr)), "frames_two_faces_or_more": int(good.sum()),
        "reprojection_px_median": round(float(np.median([tr[i]["err"] for i in fr])), 2),
        "angle_range_deg": [round(float(th.min()), 1), round(float(th.max()), 1)],
        "axis_in_cube_frame": np.round(a, 4).tolist(),
        "axis_tilt_from_cube_axes_deg": [round(float(np.degrees(np.arccos(min(1.0, abs(a[k]))))), 2) for k in range(3)],
        "cube_centre_to_axis_mm": round(float(dist_axis), 1),
        "camera_to_axis_mm": round(float(np.linalg.norm((Cref - c) - a * ((Cref - c) @ a))), 1),
        "departure_from_pure_rotation": {
            "position_mm_rms": round(float(np.sqrt((dC[good] ** 2).sum(1).mean())), 2),
            "position_mm_max": round(float(np.linalg.norm(dC[good], axis=1).max()), 2),
            "orientation_deg_rms": round(float(np.degrees(np.sqrt((dr[good] ** 2).sum(1).mean()))), 3),
            "after_table_position_mm_rms": None},
        "cube_centre_placed_by_pure_rotation_mm": {"rms": round(float(np.sqrt((iso0[good] ** 2).sum(1).mean())), 2),
                                                   "max": round(float(np.linalg.norm(iso0[good], axis=1).max()), 2)}})
    np.savez_compressed(os.path.join(out_dir, "cube_geometry.npz"), a=a, c=c, Rref=Rref, Cref=Cref, frames=fr, theta=th, good=good,
                        R=R, t=t, tab_t=tab_t, tab_dC=tab_dC, tab_dr=tab_dr, plane_gain=gain, plane_off=off, room_R=room_R)
    cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz"))
    left = np.array([C[k] - cg.pose(th[k])[1] for k in np.where(good)[0]])
    info["departure_from_pure_rotation"]["after_table_position_mm_rms"] = round(float(np.sqrt((left ** 2).sum(1).mean())), 2)
    iso1 = np.array([cg.to_cube(t[k][None], th[k])[0] for k in np.where(good)[0]])
    info["cube_centre_placed_with_table_mm"] = {"rms": round(float(np.sqrt((iso1 ** 2).sum(1).mean())), 2),
                                                "max": round(float(np.linalg.norm(iso1, axis=1).max()), 2)}
    json.dump(info, open(os.path.join(out_dir, "cube_geometry.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    return info


def surface_error(X, cube, T=None, near=12.0):
    """Depth points in the cube frame against the cube's true faces: signed distance (outside positive) of the points
    lying over a face, per face, and by gantry angle."""
    h = cube.dims / 2
    out = {}
    allr = []
    for name, cen, n in cube.faces_mm():
        ax = int(np.argmax(np.abs(n)))
        d = (X - cen) @ n
        o = [k for k in range(3) if k != ax]
        m = (np.abs(d) < near) & (np.abs(X[:, o[0]]) < h[o[0]] - 8) & (np.abs(X[:, o[1]]) < h[o[1]] - 8)
        if m.sum() < 500:
            continue
        r = d[m]; allr.append(r)
        e = {"points": int(m.sum()), "bias_mm": round(float(np.median(r)), 2), "rms_about_bias_mm": round(float(np.std(r)), 2)}
        if T is not None:
            tt = T[m]
            e["bias_by_angle_mm"] = {int(a0): round(float(np.median(r[(tt >= a0) & (tt < a0 + 30)])), 2)
                                     for a0 in range(-30, 210, 30) if ((tt >= a0) & (tt < a0 + 30)).sum() > 300}
        out[name] = e
    r = np.concatenate(allr)
    out["all"] = {"points": int(len(r)), "bias_mm": round(float(np.median(r)), 2), "rms_mm": round(float(np.sqrt((r ** 2).mean())), 2),
                  "abs_p95_mm": round(float(np.percentile(np.abs(r), 95)), 2)}
    return out


def check(rec, out_dir, frames, cube=None, step=2):
    """Fuse the cube's depth in the cube frame three ways and compare with the cube's true faces:
    pose from the markers in that very frame; the fitted rotation at the cube's angle; the fitted rotation at the
    depth-only angle (what a frame without the cube gets)."""
    from gantry_geometry import GantryGeometry
    cube = cube or Cube()
    cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz")); geo = GantryGeometry(os.path.join(out_dir, "geometry.npz"))
    g = np.load(os.path.join(out_dir, "cube_geometry.npz"))
    have = {int(f): k for k, f in enumerate(g["frames"])}
    box = cube.dims.max() / 2 + 30
    acc = {"markers in the frame": [], "fitted rotation, cube angle": [], "fitted rotation, cube angle, no table": [],
           "fitted rotation, depth-only angle": []}
    lin = None
    if os.path.exists(os.path.join(out_dir, "linac_angle.npz")):
        lin = frame_angles(out_dir, "linac")[0]; acc["fitted rotation, linac angle"] = []
    ang = []
    for i in frames:
        if int(i) not in have:
            continue
        k = have[int(i)]
        P, _ = rec.points(rec.depth(i), mask=~geo.fixed, step=step)
        th = float(g["theta"][k])
        ways = {"markers in the frame": (P - g["t"][k]) @ g["R"][k],
                "fitted rotation, cube angle": cg.to_cube(P, th),
                "fitted rotation, cube angle, no table": cg.to_cube(P, th, table=False),
                "fitted rotation, depth-only angle": cg.to_cube(P, cg.angle_from_planes(geo.angle(i)))}
        if lin is not None:
            ways["fitted rotation, linac angle"] = cg.to_cube(P, lin(i))
        m = np.all(np.abs(ways["fitted rotation, cube angle"]) < box, 1)
        for name, X in ways.items():
            acc[name].append(X[m].astype(np.float32))
        ang.append(np.full(int(m.sum()), th, np.float32))
    T = np.concatenate(ang)
    res = {name: surface_error(np.vstack(v).astype(np.float64), cube, T) for name, v in acc.items()}
    json.dump(res, open(os.path.join(out_dir, "cube_check.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "cube_fused.npz"), X=np.vstack(acc["fitted rotation, cube angle"]), T=T)
    for name, r in res.items():
        print("%-40s all: %s" % (name, r["all"]))
        for f in r:
            if f != "all":
                print("     %s %s" % (f, r[f]))
    return res
