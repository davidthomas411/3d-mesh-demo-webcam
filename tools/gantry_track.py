"""Tracking a patient's motion while the gantry turns: one depth frame at a time, no stillness assumed during tracking.

Each live frame's depth, in a region of the body, is taken to the room frame with the calibrated camera pose for that
gantry angle and registered rigidly (six degrees of freedom, point-to-plane) onto a reference surface. Two kinds of
reference are compared:

  matched   the region as the SAME camera saw it from the SAME gantry angle during a reference sweep. View-dependent
            depth errors and calibration errors are in both and cancel; only the gantry angle has to repeat.
  fused     one surface of the region built from the whole reference sweep. Simpler, but every view-dependent error of
            the depth camera and of the calibration turns into apparent motion.

Test on a recording with two passes: the first pass is the reference, the second is "live". The person was asked to lie
still, so what is measured is the floor of the method (plus breathing and whatever the person really did), and a known
motion applied to the live points shows that motion is recovered.

    python tools/gantry_track.py FOLDER OUT_DIR --ref 4500:5004:3 --live 5010:5508:12
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gantry_recording import GantryRecording, parse_frames  # noqa: E402

REGIONS = {"head": (-110, 110, -470, -225, -20, 170), "chest": (-230, 230, -120, 240, 10, 210)}   # room frame, mm


def normals(P, k=14):
    tree = cKDTree(P)
    idx = tree.query(P, k)[1]
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    return v[:, :, 0], tree


def register(live, ref, nrm, tree, iters=20):
    """Rigid motion taking the reference position of the surface to where the live points are: returns (R, t) with
    live ~ R ref + t, by moving the live points back onto the reference (point-to-plane)."""
    R, t = np.eye(3), np.zeros(3)
    for it in range(iters):
        Q = live @ R.T + t
        d, j = tree.query(Q)
        gate = 25.0 if it < 5 else 10.0
        ok = d < gate
        if ok.sum() < 200:
            return None
        q, p, n = Q[ok], ref[j[ok]], nrm[j[ok]]
        r = np.einsum("ni,ni->n", q - p, n)
        ok2 = np.abs(r) < 3 * max(1.0, np.median(np.abs(r)) * 1.48)
        q, n, r = q[ok2], n[ok2], r[ok2]
        A = np.c_[np.cross(q, n), n]
        x = np.linalg.lstsq(A, -r, rcond=None)[0]
        dR = Rot.from_rotvec(x[:3]).as_matrix()
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x[:3]) < 1e-5 and np.linalg.norm(x[3:]) < 0.01:
            break
    # (R, t) takes live onto reference; the body's motion is the inverse
    return R.T, -R.T @ t, float(np.sqrt((r ** 2).mean())), int(len(r))


def shift_of(R, t, c):
    """Where the region's centre went, and by how much it turned."""
    return R @ c + t - c, float(np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec())))


def run(rec, out_dir, ref_frames, live_frames, step=2, n_live=4000, angles="best"):
    from gantry_cube import CubeGantry, frame_angles
    from gantry_geometry import GantryGeometry
    geo = GantryGeometry(os.path.join(out_dir, "geometry.npz")); cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz"))
    rng = np.random.default_rng(0)
    ang, src = frame_angles(out_dir, angles)
    print("gantry angle from:", src)

    def cloud(i):
        th = ang(i)
        P, _ = rec.points(rec.depth(i), mask=~geo.fixed, step=step)
        return cg.to_room(P, th), th

    def cut(X, box):
        return X[(X[:, 0] > box[0]) & (X[:, 0] < box[1]) & (X[:, 1] > box[2]) & (X[:, 1] < box[3]) & (X[:, 2] > box[4]) & (X[:, 2] < box[5])]
    refs = []
    for i in ref_frames:
        X, th = cloud(i)
        refs.append((th, {k: cut(X, b).astype(np.float32) for k, b in REGIONS.items()}))
    ref_th = np.array([r[0] for r in refs])
    fused = {}
    for k in REGIONS:
        A = np.vstack([r[1][k] for r in refs]).astype(np.float64)
        key = np.floor(A / 4.0).astype(np.int64)                  # one point per 4 mm cell: the mean of what fell in it
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        S = np.zeros((len(cnt), 3)); np.add.at(S, inv.ravel(), A)
        P = (S / cnt[:, None])[cnt >= 3]
        fused[k] = (P,) + normals(P)
    test = Rot.from_rotvec(np.radians([0.0, 1.0, 0.5])).as_matrix(), np.array([2.0, -3.0, 1.5])   # a known motion, about the centre
    rows = []
    for i in live_frames:
        X, th = cloud(i)
        j = int(np.argmin(np.abs(ref_th - th)))
        row = {"frame": int(i), "gantry_deg": round(float(th), 1), "ref_angle_gap_deg": round(float(abs(ref_th[j] - th)), 2)}
        for k, b in REGIONS.items():
            L = cut(X, b)
            if len(L) < 800:
                continue
            L = L[rng.choice(len(L), min(n_live, len(L)), replace=False)]
            c = L.mean(0)
            Pm = refs[j][1][k].astype(np.float64)
            out = {"points": int(len(L))}
            for name, (P, nrm, tree) in (("matched", (Pm,) + normals(Pm) if len(Pm) > 800 else (None, None, None)), ("fused", fused[k])):
                if P is None:
                    continue
                r0 = register(L, P, nrm, tree)
                if r0 is None:
                    continue
                d, a = shift_of(r0[0], r0[1], c)
                out[name] = {"shift_mm": np.round(d, 2).tolist(), "shift_abs_mm": round(float(np.linalg.norm(d)), 2), "turn_deg": round(a, 2), "fit_rms_mm": round(r0[2], 2)}
                Lm = (L - c) @ test[0].T + c + test[1]            # the same frame with a known motion applied
                r1 = register(Lm, P, nrm, tree)
                if r1 is not None:
                    Rrel = r1[0] @ r0[0].T; trel = r1[1] - Rrel @ r0[1]   # motion recovered on top of the frame's own
                    d1, a1 = shift_of(Rrel, trel, r0[0] @ c + r0[1])
                    out[name]["known_motion_error_mm"] = round(float(np.linalg.norm(d1 - test[1])), 2)
                    out[name]["known_turn_error_deg"] = round(float(np.degrees(np.linalg.norm(Rot.from_matrix(Rrel @ test[0].T).as_rotvec()))), 2)
            row[k] = out
        rows.append(row)
        print(i, row["gantry_deg"], {k: {m: (row[k][m]["shift_abs_mm"], row[k][m].get("known_motion_error_mm")) for m in ("matched", "fused") if m in row[k]}
                                      for k in REGIONS if k in row}, flush=True)
    summary = {}
    for k in REGIONS:
        for m in ("matched", "fused"):
            v = np.array([r[k][m]["shift_abs_mm"] for r in rows if k in r and m in r[k]])
            if not len(v):
                continue
            S = np.array([r[k][m]["shift_mm"] for r in rows if k in r and m in r[k]])
            e = np.array([r[k][m].get("known_motion_error_mm", np.nan) for r in rows if k in r and m in r[k]])
            a = np.array([r[k][m]["turn_deg"] for r in rows if k in r and m in r[k]])
            summary["%s, %s reference" % (k, m)] = {
                "frames": int(len(v)), "apparent_shift_mm_median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                "max": round(float(v.max()), 2), "per_axis_rms_mm_xyz": np.round(np.sqrt((S ** 2).mean(0)), 2).tolist(),
                "apparent_turn_deg_median": round(float(np.median(a)), 2),
                "known_motion_error_mm_median": round(float(np.nanmedian(e)), 2), "known_motion_error_mm_p95": round(float(np.nanpercentile(e, 95)), 2)}
    summary["gantry_angle_from"] = src
    json.dump({"regions_room_mm": REGIONS, "known_motion": {"shift_mm": test[1].tolist(), "turn_deg": [0.0, 1.0, 0.5]}, "summary": summary, "frames": rows},
              open(os.path.join(out_dir, "track_%s.json" % src), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rec = GantryRecording(sys.argv[1])
    run(rec, sys.argv[2], parse_frames(arg("--ref", "4500:5004:3"), rec.n), parse_frames(arg("--live", "5010:5508:12"), rec.n), angles=arg("--angles", "best"))
