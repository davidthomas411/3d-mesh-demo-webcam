"""Recordings from an Orbbec Femto Bolt riding on a rotating gantry (colour, depth aligned to colour, infrared).

A recording is a folder:
    session.json          what was recorded
    calibration.json      intrinsics of the colour and depth cameras, depth -> colour
    frames.csv            one row a frame, with host and device times
    color.mp4             1280 x 720
    depth.mkv             16-bit millimetres, lossless (FFV1), ALIGNED TO THE COLOUR CAMERA and undistorted
    ir.mkv                16-bit infrared (in the depth camera's own geometry)

Because the depth is aligned to the colour picture, a depth pixel (u, v) with value z is the 3-D point
((u - cx) z / fx, (v - cy) z / fy, z) in the colour camera's frame, with the colour intrinsics.

    python tools/gantry_recording.py FOLDER info
    python tools/gantry_recording.py FOLDER sheet OUT.jpg [--n 24]
    python tools/gantry_recording.py FOLDER calibrate OUT_DIR [--floor K]     gantry angle and axis per frame, from the depth
    python tools/gantry_recording.py FOLDER fuse OUT_DIR NAME --frames a:b:step --box x0,x1,y0,y1,z0,z1 [--flat-above Z]
    python tools/gantry_recording.py FOLDER pose OUT_DIR [--frames a:b:step] [--url ws://127.0.0.1:8098/stream]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class GantryRecording:
    def __init__(self, folder):
        import cv2
        self.folder = folder
        self.session = json.load(open(os.path.join(folder, "session.json")))
        self.cal = json.load(open(os.path.join(folder, "calibration.json")))
        c = self.cal["color"]["intrinsic"]
        self.K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        self.size = (c["width"], c["height"])
        self._color = cv2.VideoCapture(os.path.join(folder, "color.mp4"))
        self._depth = cv2.VideoCapture(os.path.join(folder, "depth.mkv"), cv2.CAP_FFMPEG, [cv2.CAP_PROP_CONVERT_RGB, 0])
        self.n = int(min(self._color.get(cv2.CAP_PROP_FRAME_COUNT), self._depth.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._at = {"c": -1, "d": -1}
        v, u = np.mgrid[0:self.size[1], 0:self.size[0]]
        self.rays = np.stack([(u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"]], -1).astype(np.float32)

    def _read(self, cap, key, i):
        import cv2
        if self._at[key] != i - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        self._at[key] = i if ok else -1
        return f if ok else None

    def color(self, i):
        """BGR, uint8 (720, 1280, 3)."""
        return self._read(self._color, "c", i)

    def depth(self, i):
        """uint16 mm (720, 1280), 0 = nothing, aligned to the colour picture."""
        z = self._read(self._depth, "d", i)
        return None if z is None else z.reshape(z.shape[0], z.shape[1])

    def points(self, z, mask=None, step=1):
        """(n, 3) mm in the colour camera's frame, and their pixel coordinates (v, u)."""
        m = z > 0 if mask is None else (mask & (z > 0))
        if step > 1:
            g = np.zeros_like(m); g[::step, ::step] = True; m &= g
        v, u = np.where(m)
        d = z[v, u].astype(np.float64)
        return np.column_stack([self.rays[v, u, 0] * d, self.rays[v, u, 1] * d, d]), (v, u)


def parse_frames(spec, n):
    a, b, s = (spec.split(":") + ["", "", ""])[:3]
    return range(int(a or 0), int(b or n), int(s or 1))


def cmd_sheet(rec, out, n=24):
    import cv2
    tiles = []
    for i in np.linspace(0, rec.n - 1, n).astype(int):
        im = cv2.resize(rec.color(int(i)), (320, 180)); cv2.putText(im, str(i), (5, 20), 0, 0.6, (0, 255, 255), 2)
        z = rec.depth(int(i))
        v = cv2.applyColorMap(np.clip(z.astype(np.float32) / 4000 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO); v[z == 0] = 0
        tiles.append(np.vstack([im, cv2.resize(v, (320, 180))]))
    cols = 6
    cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles) - cols + 1, cols)]))


class ModelClient:
    """The running demo server, used as the pose model: a picture in, the mesh and what the server says about it out."""
    NV = 18439

    def __init__(self, url):
        from simple_websocket import Client
        self.ws = Client.connect(url)
        self.ws.send(json.dumps({"type": "config", "action": "set_depth", "fusion": False, "bolt": False}))
        self.ready = False

    def infer(self, bgr, quality=90, timeout=60):
        import cv2
        jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self.ws.send(json.dumps({"capture_timestamp_ms": time.time() * 1000, "encode_ms": 0}))
        self.ws.send(jpg)
        meta, blob = None, None
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.ws.receive(timeout=timeout)
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                blob = bytes(msg)
                if meta is not None:
                    break
                continue
            m = json.loads(msg)
            if m.get("type") == "config" or m.get("fitted"):
                continue
            meta = m
            if m.get("no_person") or m.get("error") or m.get("skip_mesh"):
                break
        verts = None
        if blob is not None and len(blob) >= self.NV * 12:
            verts = np.frombuffer(blob[:self.NV * 12], "<f4").reshape(-1, 3)
        return meta, verts

    def close(self):
        self.ws.close()


ROT = {0: None, 90: "ROTATE_90_CLOCKWISE", 180: "ROTATE_180", 270: "ROTATE_90_COUNTERCLOCKWISE"}


def turned(img, deg):
    """The picture turned clockwise by deg (0, 90, 180, 270)."""
    import cv2
    return img if not deg else cv2.rotate(img, getattr(cv2, ROT[deg]))


def score_pose(meta, verts, z):
    """How well the model's mesh agrees with the depth picture it was not shown: project the mesh (the model's own focal
    length, principal point at the centre), bring it to the depth's distance with ONE scale, and look at what is left.
    -> dict(on_body: fraction of vertices that land on a depth pixel within 15 cm after the scale, median_mm, scale)."""
    h, w = z.shape
    X = verts.astype(np.float64) + np.asarray(meta["pred_cam_t"], np.float64)
    f = float(meta["focal_length"])
    u = np.round(f * X[:, 0] / X[:, 2] + w / 2).astype(int); v = np.round(f * X[:, 1] / X[:, 2] + h / 2).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (X[:, 2] > 0.05)
    d = np.zeros(len(X)); d[inside] = z[v[inside], u[inside]]
    ok = inside & (d > 0)
    if ok.sum() < 200:
        return {"on_body": 0.0, "median_mm": None, "scale": None, "inside": float(inside.mean())}, (u, v, inside)
    r = d[ok] / (X[ok, 2] * 1000.0)
    # the mesh has a back the camera cannot see: the scale is set by the nearest surface, so use a low quantile band
    s = float(np.median(r))
    e = d[ok] - s * X[ok, 2] * 1000.0
    front = e > -150
    return {"on_body": round(float((np.abs(e) < 150).sum() / len(X)), 3), "median_mm": round(float(np.median(np.abs(e[front]))), 1),
            "scale": round(s, 3), "inside": round(float(inside.mean()), 3)}, (u, v, inside)


def cmd_pose(rec, out_dir, frames, url, rotations=(0, 90, 180, 270)):
    """The pose model on the recording's pictures, each given at four rotations (a connection each, so that the server's
    tracking follows one orientation). Writes pose.json and one contact sheet per rotation."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    clients = {r: ModelClient(url) for r in rotations}
    rows, tiles = [], {r: [] for r in rotations}
    for i in frames:
        img, z = rec.color(i), rec.depth(i)
        if img is None or z is None:
            continue
        row = {"frame": int(i)}
        for r in rotations:
            im, zz = turned(img, r), turned(z, r)
            meta, verts = clients[r].infer(im)
            small = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4))
            if meta is None or meta.get("no_person") or verts is None:
                row[str(r)] = {"person": False}
            else:
                sc, (u, v, inside) = score_pose(meta, verts, zz)
                row[str(r)] = dict(sc, person=True, kp=np.round(np.asarray(meta["pred_keypoints_2d"])[:, :2], 1).tolist())
                layer = small.copy()
                layer[np.clip(v[inside] // 4, 0, small.shape[0] - 1), np.clip(u[inside] // 4, 0, small.shape[1] - 1)] = (0, 255, 0)
                small = cv2.addWeighted(small, 0.55, layer, 0.45, 0)
                for kx, ky in np.asarray(meta["pred_keypoints_2d"])[:, :2]:
                    cv2.circle(small, (int(kx) // 4, int(ky) // 4), 1, (0, 0, 255), -1)
                cv2.putText(small, "%s %s" % (sc["on_body"], sc["median_mm"]), (4, small.shape[0] - 6), 0, 0.45, (0, 255, 255), 1)
            cv2.putText(small, str(i), (4, 16), 0, 0.5, (0, 255, 255), 1)
            canvas = np.zeros((320, 320, 3), np.uint8); canvas[:small.shape[0], :small.shape[1]] = small
            tiles[r].append(canvas)
        rows.append(row)
        print(i, {r: (row[str(r)].get("on_body"), row[str(r)].get("median_mm")) for r in rotations}, flush=True)
    for c in clients.values():
        c.close()
    json.dump(rows, open(os.path.join(out_dir, "pose.json"), "w"))
    for r in rotations:
        t = tiles[r]
        cols = 8
        t += [np.zeros_like(t[0])] * ((-len(t)) % cols)
        cv2.imwrite(os.path.join(out_dir, "pose_rot%d.jpg" % r), np.vstack([np.hstack(t[k:k + cols]) for k in range(0, len(t), cols)]))


def main():
    folder, cmd = sys.argv[1], sys.argv[2]
    rec = GantryRecording(folder)
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    if cmd == "info":
        print(json.dumps({"frames": rec.n, "size": rec.size, "K": rec.K.tolist(), "seconds": rec.session.get("duration_s")}, indent=1))
    elif cmd == "sheet":
        cmd_sheet(rec, sys.argv[3], int(arg("--n", 24)))
    elif cmd == "pose":
        cmd_pose(rec, sys.argv[3], parse_frames(arg("--frames", "0::60"), rec.n), arg("--url", "ws://127.0.0.1:8098/stream"))
    elif cmd == "calibrate":
        import gantry_geometry as gg
        gg.calibrate(rec, sys.argv[3], floor=arg("--floor"))
    elif cmd == "fuse":
        import gantry_geometry as gg
        X, C, T = gg.fuse(rec, sys.argv[3], sys.argv[4], parse_frames(arg("--frames", "0::30"), rec.n), [float(x) for x in arg("--box").split(",")], int(arg("--step", 2)))
        if "--flat-above" in sys.argv:                            # a cube: how flat are its faces once fused?
            m = X[:, 2] > float(arg("--flat-above"))
            print(json.dumps(gg.flatness(X[m].astype(np.float64), T[m]), indent=1))
    elif cmd == "probe":
        mc = ModelClient(arg("--url", "ws://127.0.0.1:8098/stream"))
        meta, verts = mc.infer(rec.color(int(sys.argv[3])))
        print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 300 else str(type(v)) + " len " + str(len(v))) for k, v in (meta or {}).items()}, indent=1))
        print("verts", None if verts is None else verts.shape)
        mc.close()


if __name__ == "__main__":
    main()
