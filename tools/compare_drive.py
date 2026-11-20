"""Compare the two ways of driving the rig on real estimator output: segments fitted to skin patches vs segments taken from the
MHR joints the estimator predicts. Reports, per bone group, how much of the bone surface lies outside the live skin.

Input is a file (or a running server) with one mesh and its joints: float32 vertices (18439 x 3), then n x 3 joint positions and
n x 9 rotation-matrix entries, the layout of the /stream message and of GET /mesh?joints=1.

usage: python tools/compare_drive.py atlas/atlas faces.npy mesh_with_joints.bin [more.bin ...]
       python tools/compare_drive.py atlas/atlas faces.npy --image person.jpg [--url ws://127.0.0.1:8097/stream] [--save out.bin]"""
import argparse, collections, json, os, sys, time
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anatomy.runtime import Atlas  # noqa: E402

NV = 18439
BONE_GROUPS = ("skull", "spine", "ribcage", "shoulder", "pelvis_leg")


def split(buf):
    a = np.frombuffer(buf, "<f4").astype(np.float64)
    v = a[:NV * 3].reshape(NV, 3)
    rest = a[NV * 3:]
    if not len(rest):
        return v, None
    n = len(rest) // 12
    return v, (rest[:n * 3].reshape(n, 3), rest[n * 3:].reshape(n, 3, 3))


def from_server(url, image, frames=45):
    import cv2
    from simple_websocket import Client
    img = cv2.resize(cv2.imread(image), (640, 480))
    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    ws = Client.connect(url)
    last = None
    for _ in range(frames):                                   # past the focal calibration
        ws.send(json.dumps({"capture_timestamp_ms": time.time() * 1000, "encode_ms": 0}))
        ws.send(jpeg)
        meta = json.loads(ws.receive())
        if meta.get("error") or meta.get("no_person"):
            continue
        last = ws.receive()
    ws.close()
    if last is None:
        raise SystemExit("the server found no person in the image")
    return last


def outside(mesh, posed, atlas):
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    per = {}
    for s in atlas.structures:
        if s["group"] not in BONE_GROUPS or s["bind"] == "soft":
            continue
        p = posed[s["name"]]
        ins = mesh.contains(p)
        worst = trimesh.proximity.closest_point(mesh, p[~ins])[1].max() * 1000 if (~ins).any() else 0.0
        a = agg[s["group"]]; a[0] += len(p); a[1] += int((~ins).sum()); a[2] = max(a[2], worst)
        per[s["name"]] = (100 * (~ins).mean(), worst)
    return agg, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas"); ap.add_argument("faces"); ap.add_argument("files", nargs="*")
    ap.add_argument("--image"); ap.add_argument("--url", default="ws://127.0.0.1:8097/stream"); ap.add_argument("--save")
    a = ap.parse_args()
    atlas = Atlas(a.atlas)
    faces = np.load(a.faces).reshape(-1, 3)
    inputs = [(f, open(f, "rb").read()) for f in a.files]
    if a.image:
        buf = from_server(a.url, a.image)
        if a.save:
            open(a.save, "wb").write(buf)
        inputs.append((a.image, buf))
    for name, buf in inputs:
        v, joints = split(buf)
        if joints is None:
            print(f"{name}: no joints in this capture"); continue
        mesh = trimesh.Trimesh(v, faces, process=False)
        m = atlas.meta["mhr"]
        print(f"\n{name}")
        for bone, lb in m["limbs"].items():
            n = np.linalg.norm(joints[0][lb["child"]] - joints[0][lb["joint"]])
            print(f"  {bone:14s} joint length {n * 1000:5.0f} mm (neutral body {lb['rest_len'] * 1000:.0f} mm, x{n / lb['rest_len']:.3f})")
        res = {}
        for tag, mj in (("skin patches", None), ("MHR joints", joints)):
            res[tag] = outside(mesh, atlas.pose(v, spine="physio", mhr_joints=mj), atlas)
        print(f"  {'bone group':12s} {'skin patches':>26s} {'MHR joints':>26s}")
        for g in BONE_GROUPS:
            row = []
            for tag in ("skin patches", "MHR joints"):
                n, o, w = res[tag][0][g]
                row.append(f"{100 * o / max(n, 1):5.1f}% out, worst {w:5.1f} mm")
            print(f"  {g:12s} {row[0]:>26s} {row[1]:>26s}")
        worst = sorted(res["MHR joints"][1].items(), key=lambda kv: -kv[1][0])[:5]
        print("  most outside with MHR joints:", ", ".join(f"{k} {p:.0f}%/{w:.0f} mm (skin-driven {res['skin patches'][1][k][0]:.0f}%)" for k, (p, w) in worst))


if __name__ == "__main__":
    main()
