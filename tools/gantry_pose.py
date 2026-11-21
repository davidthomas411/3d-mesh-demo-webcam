"""The pose model through a gantry sweep, judged in the room.

The person lies still while the camera goes round, so every frame's mesh, taken to the room frame with the calibrated
camera pose, should be the same mesh in the same place. How far they differ is the model's error as a function of viewing
angle, with nothing assumed about the model: orientation, position and what is left after both (pose and shape).
The fused point cloud of the person (fuse ... --cube) is the independent truth for the surface.

    python tools/gantry_pose.py FOLDER OUT_DIR [--frames a:b:step] [--rot 270] [--cloud person_iso] [--url ws://...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gantry_recording import GantryRecording, ModelClient, parse_frames, score_pose, turned  # noqa: E402


def unturn(u, v, deg, w0, h0):
    """Pixel of the turned picture -> pixel of the original (w0 x h0)."""
    if deg == 90:
        return v, h0 - 1 - u
    if deg == 270:
        return w0 - 1 - v, u
    if deg == 180:
        return w0 - 1 - u, h0 - 1 - v
    return u, v


def mesh_in_camera(meta, verts, scale, deg, rec):
    """The model's mesh as points in the real camera's frame (mm): each vertex keeps the pixel the model gave it and is put
    at the model's distance times the one scale the depth picture gave."""
    w0, h0 = rec.size
    wt, ht = (h0, w0) if deg in (90, 270) else (w0, h0)
    X = verts.astype(np.float64) + np.asarray(meta["pred_cam_t"], np.float64)
    f = float(meta["focal_length"])
    u = f * X[:, 0] / X[:, 2] + wt / 2; v = f * X[:, 1] / X[:, 2] + ht / 2
    u0, v0 = unturn(u, v, deg, w0, h0)
    Z = scale * X[:, 2] * 1000.0
    K = rec.K
    return np.c_[(u0 - K[0, 2]) / K[0, 0] * Z, (v0 - K[1, 2]) / K[1, 1] * Z, Z]


def kabsch(A, B):
    """Rigid motion taking A onto B: B ~ A R^T + t."""
    ca, cb = A.mean(0), B.mean(0)
    u, s, vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt).T
    return R, cb - ca @ R.T


def run(rec, out_dir, frames, url, deg=270, cloud="person_iso"):
    import cv2
    from scipy.spatial import cKDTree

    from gantry_cube import CubeGantry
    from gantry_geometry import GantryGeometry
    geo = GantryGeometry(os.path.join(out_dir, "geometry.npz")); cg = CubeGantry(os.path.join(out_dir, "cube_geometry.npz"))
    mc = ModelClient(url)
    M, th, fr = [], [], []
    for i in frames:
        img, z = rec.color(i), rec.depth(i)
        meta, verts = mc.infer(turned(img, deg))
        if meta is None or meta.get("no_person") or verts is None:
            continue
        sc, _ = score_pose(meta, verts, turned(z, deg))
        if sc["scale"] is None:
            continue
        t = cg.angle_from_planes(geo.angle(i))
        M.append(cg.to_room(mesh_in_camera(meta, verts, sc["scale"], deg, rec), t)); th.append(t); fr.append(int(i))
        print(i, round(t, 1), sc, flush=True)
    mc.close()
    M, th = np.array(M), np.array(th)
    C = np.load(os.path.join(out_dir, cloud + ".npz"))
    cl = C["X"][::40].astype(np.float64)
    tree_c = cKDTree(cl)
    # the part of the body the camera ever saw: vertices of the consensus mesh that lie near the cloud
    med = np.median(M, 0)
    seen = tree_c.query(med)[0] < 60
    for _ in range(2):                                            # consensus = median of the meshes after each is laid on it rigidly
        al = []
        for m in M:
            R, t = kabsch(m[seen], med[seen]); al.append(m @ R.T + t)
        med = np.median(np.array(al), 0); seen = tree_c.query(med)[0] < 60
    rows = []
    for m, t, i in zip(M, th, fr):
        R, tr = kabsch(m[seen], med[seen])
        left = np.linalg.norm(m[seen] @ R.T + tr - med[seen], axis=1)
        d_cloud = cKDTree(m).query(cl)[0]                         # the room's cloud to this frame's mesh
        rows.append({"frame": i, "gantry_deg": round(float(t), 1),
                     "orientation_off_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 1),
                     "position_off_mm": round(float(np.linalg.norm(m[seen].mean(0) - med[seen].mean(0))), 1),
                     "vertex_off_mm_median": round(float(np.median(np.linalg.norm(m[seen] - med[seen], axis=1))), 1),
                     "after_rigid_mm_median": round(float(np.median(left)), 1),
                     "cloud_to_mesh_mm_median": round(float(np.median(d_cloud)), 1),
                     "cloud_to_mesh_mm_p90": round(float(np.percentile(d_cloud, 90)), 1)})
    d_med = cKDTree(med).query(cl)[0]
    q = lambda k: [r[k] for r in rows]
    summary = {"frames": len(rows), "picture_turned_deg": deg, "vertices_in_view": int(seen.sum()),
               "orientation_off_deg": {"median": float(np.median(q("orientation_off_deg"))), "max": float(np.max(q("orientation_off_deg")))},
               "position_off_mm": {"median": float(np.median(q("position_off_mm"))), "max": float(np.max(q("position_off_mm")))},
               "vertex_off_mm": {"median": float(np.median(q("vertex_off_mm_median"))), "max": float(np.max(q("vertex_off_mm_median")))},
               "after_rigid_mm": {"median": float(np.median(q("after_rigid_mm_median"))), "max": float(np.max(q("after_rigid_mm_median")))},
               "cloud_to_mesh_mm": {"median": float(np.median(q("cloud_to_mesh_mm_median"))), "max": float(np.max(q("cloud_to_mesh_mm_median")))},
               "cloud_to_consensus_mesh_mm": {"median": round(float(np.median(d_med)), 1), "p90": round(float(np.percentile(d_med, 90)), 1)}}
    json.dump({"summary": summary, "frames": rows}, open(os.path.join(out_dir, "pose_room_rot%d.json" % deg), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "pose_room_rot%d.npz" % deg), M=M.astype(np.float32), theta=th, frames=np.array(fr), consensus=med, seen=seen)
    # picture: the cloud (grey) with every frame's mesh (coloured by gantry angle) and the consensus (white), three views
    box = [cl[:, k].min() for k in range(3)], [cl[:, k].max() for k in range(3)]
    s = 700 / max(box[1][k] - box[0][k] for k in range(3))
    views = []
    for a0, a1 in ((0, 2), (1, 2), (0, 1)):
        w, h = int((box[1][a0] - box[0][a0]) * s) + 40, int((box[1][a1] - box[0][a1]) * s) + 40
        img = np.zeros((h, w, 3), np.uint8)

        def put(P, col, step=1):
            uu = ((P[::step, a0] - box[0][a0]) * s).astype(int) + 20; vv = h - 20 - ((P[::step, a1] - box[0][a1]) * s).astype(int)
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = col
        put(cl, (90, 90, 90))
        for m, t in zip(M, th):
            c = cv2.applyColorMap(np.uint8([[int(255 * (t - th.min()) / max(1e-6, np.ptp(th)))]]), cv2.COLORMAP_TURBO)[0, 0]
            put(m[seen], [int(x) for x in c], 6)
        views.append(img)
    hh = max(v.shape[0] for v in views)
    cv2.imwrite(os.path.join(out_dir, "pose_room_rot%d.jpg" % deg), np.hstack([np.vstack([v, np.zeros((hh - v.shape[0], v.shape[1], 3), np.uint8)]) for v in views]))
    print(json.dumps(summary, indent=1))
    print("gantry  orient  position  vertex  after-rigid  cloud->mesh")
    for r in rows:
        print("%6.1f  %5.1f  %6.1f  %6.1f  %6.1f  %6.1f" % (r["gantry_deg"], r["orientation_off_deg"], r["position_off_mm"], r["vertex_off_mm_median"],
                                                         r["after_rigid_mm_median"], r["cloud_to_mesh_mm_median"]))
    return summary


if __name__ == "__main__":
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rec = GantryRecording(sys.argv[1])
    run(rec, sys.argv[2], parse_frames(arg("--frames", "4410:5560:36"), rec.n), arg("--url", "ws://127.0.0.1:8098/stream"),
        int(arg("--rot", 270)), arg("--cloud", "person_iso"))
