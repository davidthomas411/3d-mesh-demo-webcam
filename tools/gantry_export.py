"""A gantry recording with its tracking, made ready for the page (/gantry in the web app).

For every exported frame: one picture (the colour frame turned upright with the tracked regions tinted, and slices
through head and chest showing the live depth over the reference surface), and the frame's numbers (gantry angle, the six
shifts of each region against the reference). Written to a folder OUTSIDE the repository, because they are pictures of a
person; the server reads it from GANTRY_WEB (default ../recordings/gantry/web).

    python tools/gantry_export.py FOLDER OUT_DIR WEB_DIR [--frames 4380:5580:3] [--ref 4500:5004:3]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gantry_recording import GantryRecording, parse_frames, turned  # noqa: E402
from gantry_track import REGIONS, normals, register, shift_of  # noqa: E402

TINT = {"head": (255, 200, 0), "chest": (0, 200, 255)}            # BGR
SLICES = {"head": dict(y=(-365, -335), x=(-169, 169), z=(-25, 195), size=(540, 352)),     # 1.6 px per mm
          "chest": dict(y=(40, 70), x=(-250, 250), z=(-40, 286), size=(540, 352))}        # 1.08 px per mm


def cut(X, b):
    return (X[:, 0] > b[0]) & (X[:, 0] < b[1]) & (X[:, 1] > b[2]) & (X[:, 1] < b[3]) & (X[:, 2] > b[4]) & (X[:, 2] < b[5])


def main():
    import cv2

    from gantry_cube import CubeGantry, frame_angles
    from gantry_geometry import GantryGeometry
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rec = GantryRecording(sys.argv[1]); out, web = sys.argv[2], sys.argv[3]
    os.makedirs(os.path.join(web, "f"), exist_ok=True)
    frames = parse_frames(arg("--frames", "4380:5580:3"), rec.n); ref_frames = parse_frames(arg("--ref", "4500:5004:3"), rec.n)
    geo = GantryGeometry(os.path.join(out, "geometry.npz")); cg = CubeGantry(os.path.join(out, "cube_geometry.npz"))
    ang, src = frame_angles(out)
    lin = np.load(os.path.join(out, "linac_angle.npz"))["linac"] if src == "linac" else None
    rng = np.random.default_rng(0)

    def cloud(i, step):
        P, (v, u) = rec.points(rec.depth(i), mask=~geo.fixed, step=step)
        return cg.to_room(P, ang(i)), v, u
    # the reference: one surface per region from the reference sweep (4 mm cells), and its slices for the pictures
    acc = {k: [] for k in REGIONS}; sl_ref = {k: [] for k in SLICES}
    for i in ref_frames:
        X, _, _ = cloud(i, 2)
        for k, b in REGIONS.items():
            acc[k].append(X[cut(X, b)].astype(np.float32))
        for k, s in SLICES.items():
            m = (X[:, 1] > s["y"][0]) & (X[:, 1] < s["y"][1]) & (X[:, 0] > s["x"][0]) & (X[:, 0] < s["x"][1]) & (X[:, 2] > s["z"][0]) & (X[:, 2] < s["z"][1])
            sl_ref[k].append(X[m][:, [0, 2]].astype(np.float32))
    fused = {}
    for k in REGIONS:
        A = np.vstack(acc[k]).astype(np.float64)
        _, inv, cnt = np.unique(np.floor(A / 4.0).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        S = np.zeros((len(cnt), 3)); np.add.at(S, inv.ravel(), A)
        P = (S / cnt[:, None])[cnt >= 3]
        fused[k] = (P,) + normals(P)

    def panel(k, pts, colour, img=None):
        s = SLICES[k]; w, h = s["size"]
        if img is None:
            img = np.full((h, w, 3), 255, np.uint8)
            for gx in range(int(np.ceil(s["x"][0] / 50) * 50), int(s["x"][1]) + 1, 50):
                px = int((gx - s["x"][0]) / (s["x"][1] - s["x"][0]) * w); cv2.line(img, (px, 0), (px, h), (232, 232, 232), 1)
            for gz in range(int(np.ceil(s["z"][0] / 50) * 50), int(s["z"][1]) + 1, 50):
                py = int((s["z"][1] - gz) / (s["z"][1] - s["z"][0]) * h); cv2.line(img, (0, py), (w, py), (232, 232, 232), 1)
        if len(pts):
            px = ((pts[:, 0] - s["x"][0]) / (s["x"][1] - s["x"][0]) * w).astype(int); py = ((s["z"][1] - pts[:, 1]) / (s["z"][1] - s["z"][0]) * h).astype(int)
            ok = (px >= 1) & (px < w - 1) & (py >= 1) & (py < h - 1)
            for dx in (0, 1):
                for dy in (0, 1):
                    img[py[ok] + dy, px[ok] + dx] = colour
        return img
    base = {k: panel(k, np.vstack(sl_ref[k])[::3], (150, 150, 150)) for k in SLICES}
    rows = []
    t0 = None
    times = np.genfromtxt(os.path.join(sys.argv[1], "frames.csv"), delimiter=",", names=True, dtype=None, encoding=None)["host_us"] / 1e6
    for n, i in enumerate(frames):
        if n % 40 == 0:
            print("frame", i, flush=True)
        im = rec.color(i)
        X, v, u = cloud(i, 1)
        row = {"frame": int(i), "t": round(float(times[i] - times[frames[0]]), 3), "angle": round(float(ang(i)), 2),
               "linac": None if lin is None else round(float(lin[i]), 1), "img": "f/%05d.jpg" % i,
               "pass": "reference" if ref_frames[0] <= i <= ref_frames[-1] else "live"}
        layer = im.copy()
        for k, b in REGIONS.items():
            m = cut(X, b)
            layer[v[m], u[m]] = TINT[k]
            L = X[m]
            if len(L) < 800:
                row[k] = None
                continue
            L = L[rng.choice(len(L), min(4000, len(L)), replace=False)]
            r = register(L, *fused[k])
            if r is None:
                row[k] = None
                continue
            d, a = shift_of(r[0], r[1], L.mean(0))
            row[k] = {"shift": np.round(d, 2).tolist(), "size": round(float(np.linalg.norm(d)), 2), "turn": round(a, 2), "points": int(m.sum())}
        pic = turned(cv2.addWeighted(im, 0.62, layer, 0.38, 0), 180)      # head up
        pic = cv2.resize(pic, (1080, 608), interpolation=cv2.INTER_AREA)
        panels = []
        for k, s in SLICES.items():
            m = (X[:, 1] > s["y"][0]) & (X[:, 1] < s["y"][1]) & (X[:, 0] > s["x"][0]) & (X[:, 0] < s["x"][1]) & (X[:, 2] > s["z"][0]) & (X[:, 2] < s["z"][1])
            p = panel(k, X[m][:, [0, 2]], (40, 40, 220), base[k].copy())
            cv2.putText(p, "%s: slice across the body, seen from the feet (50 mm grid)" % k, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(p, "grey = reference sweep   red = this frame", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
            panels.append(p)
        cv2.imwrite(os.path.join(web, row["img"]), np.vstack([pic, np.hstack(panels)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        rows.append(row)
        if n % 40 == 0:
            print(i, row["angle"], {k: (row[k] or {}).get("size") for k in REGIONS}, flush=True)
    meta = {"recording": os.path.basename(os.path.normpath(sys.argv[1])), "angle_from": src, "regions_room_mm": REGIONS,
            "reference_frames": [int(ref_frames[0]), int(ref_frames[-1])],
            "note": "Each frame measured on its own against one surface built from the reference sweep. Frames inside the "
                    "reference sweep are part of that surface, so their shifts are small by construction."}
    json.dump({"meta": meta, "frames": rows}, open(os.path.join(web, "index.json"), "w"))
    print("wrote", len(rows), "frames to", web)


if __name__ == "__main__":
    main()
