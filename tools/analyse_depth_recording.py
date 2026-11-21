"""Compare the body mesh with the Femto Bolt's depth in a recording made by the server (GET /record?frames=N&name=NAME).

For every frame: where the model put the mesh (distance of the trunk from the camera), where the depth camera measures the
surface the mesh covers, the single scale about the camera that reconciles them (depth/live.py) and how well the mesh then fits.
Writes a figure of a few frames: the picture with the mesh's outline, and a top view (x, z) of the depth points and the mesh
before and after the scale.

usage: python tools/analyse_depth_recording.py ../recordings/NAME.npz [out.png]
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from depth.anchor import depth_anchor  # noqa: E402
from depth.cloud import points_from_depth, to_colour_frame  # noqa: E402
from depth.live import depth_map_from_points, fast_visible  # noqa: E402


def load(path):
    z = np.load(path, allow_pickle=True)
    cal = json.loads(str(z["calibration"])) if "calibration" in z else None
    return z, cal


def frame_geometry(z, cal, i, hw=(180, 320)):
    """Mesh in the camera frame (mm), depth points in the colour frame (mm), depth map and K at hw."""
    P = (z["verts"][i].astype(np.float64) + z["cam_t"][i].astype(np.float64)) * 1000.0
    R, t, k = np.array(cal["R"]).reshape(3, 3), np.array(cal["t"]), cal["k"]
    pts = to_colour_frame(points_from_depth(z["depth"][i], z["rays"]), R, t)
    h, w = hw
    K = np.array([[k[0] * w, 0, k[2] * w], [0, k[1] * h, k[3] * h], [0, 0, 1.0]])
    return P, pts, depth_map_from_points(pts, K, hw), K


def main():
    path = sys.argv[1]
    z, cal = load(path)
    n = len(z["verts"])
    print(f"{os.path.basename(path)}: {n} frames, {(z['t'][-1] - z['t'][0]):.1f} s, depth in {int(z['has_depth'].sum())} of them")
    if cal is None:
        raise SystemExit("no depth in this recording")
    import cv2
    img0 = cv2.imdecode(z["jpeg"][0], cv2.IMREAD_COLOR)
    H, W = img0.shape[:2]
    faces = z["faces"]
    rows = []
    for i in range(n):
        if not z["has_depth"][i]:
            continue
        P, pts, dmap, K = frame_geometry(z, cal, i)
        vis = fast_visible(P, faces, K, dmap.shape)
        _, a = depth_anchor(P, dmap, K, vis, min_pts=200)
        Kused = z["K"][i]
        rows.append({"i": i, "mesh_z": float(np.median(P[vis][:, 2])) if vis.any() else np.nan, "cam_t": z["cam_t"][i],
                     "focal_used": float(Kused[0, 0]) if Kused[0, 0] > 0 else float(z["focal"][i]), "factory": bool(z["factory"][i]),
                     "scale": a.get("value", np.nan), "rms": a.get("rms_mm", np.nan), "spread": a.get("spread", np.nan), "n": a.get("n", 0),
                     "meas_z": float(np.median(P[vis][:, 2]) * a["value"]) if a.get("ok") else np.nan})
    ok = [r for r in rows if np.isfinite(r["scale"])]
    print(f"picture {W}x{H}; factory focal for this size would be {cal['k'][0] * W:.1f} px")
    print(f"focal used by the model: {np.median([r['focal_used'] for r in rows]):.1f} px (factory intrinsics in {sum(r['factory'] for r in rows)} of {len(rows)} frames)")
    if not ok:
        raise SystemExit("the mesh never overlapped valid depth")
    sc = np.array([r["scale"] for r in ok])
    print(f"mesh trunk distance (model): median {np.median([r['mesh_z'] for r in ok]):.0f} mm; measured: {np.median([r['meas_z'] for r in ok]):.0f} mm")
    print(f"scale that reconciles them: median {np.median(sc):.3f}, range {sc.min():.3f}-{sc.max():.3f}, frame-to-frame sd {np.std(np.diff(sc)):.4f}")
    print(f"fit after the scale: median {np.median([r['rms'] for r in ok]):.0f} mm rms (per-vertex ratio spread {np.median([r['spread'] for r in ok]):.3f}), "
          f"{np.median([r['n'] for r in ok]):.0f} vertices compared")
    if len(sys.argv) > 2:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = [ok[j]["i"] for j in np.linspace(0, len(ok) - 1, 3).astype(int)]
        fig, ax = plt.subplots(2, 3, figsize=(17, 10))
        for c, i in enumerate(pick):
            P, pts, dmap, K = frame_geometry(z, cal, i)
            r = next(q for q in ok if q["i"] == i)
            img = cv2.cvtColor(cv2.imdecode(z["jpeg"][i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            k = cal["k"]
            u = (k[0] * P[:, 0] / P[:, 2] + k[2]) * W; v = (k[1] * P[:, 1] / P[:, 2] + k[3]) * H
            ax[0, c].imshow(img); ax[0, c].plot(u[::9], v[::9], ".", ms=0.8, color="cyan", alpha=0.6)
            ax[0, c].set_title(f"frame {i}: mesh projected with the factory intrinsics"); ax[0, c].axis("off")
            band = np.abs(pts[:, 1] - np.median(P[:, 1])) < 250          # a slab through the trunk, seen from above
            mb = np.abs(P[:, 1] - np.median(P[:, 1])) < 250
            ax[1, c].plot(pts[band][::7, 0], pts[band][::7, 2], ".", ms=0.6, color="gray", label="depth camera")
            ax[1, c].plot(P[mb][::3, 0], P[mb][::3, 2], ".", ms=1.0, color="red", label="mesh (model)")
            S = P * r["scale"]
            ax[1, c].plot(S[mb][::3, 0], S[mb][::3, 2], ".", ms=1.0, color="green", label=f"mesh x{r['scale']:.3f}")
            ax[1, c].set_xlim(-1200, 1200); ax[1, c].set_ylim(0, 4500); ax[1, c].set_aspect("equal"); ax[1, c].invert_yaxis()
            ax[1, c].set_xlabel("x (mm)"); ax[1, c].set_ylabel("distance z (mm)"); ax[1, c].legend(markerscale=8, loc="lower right")
            ax[1, c].set_title(f"top view; fit after scale {r['rms']:.0f} mm rms")
        plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
        print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
