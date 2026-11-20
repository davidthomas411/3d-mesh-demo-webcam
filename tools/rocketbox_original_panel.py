"""Draw the ORIGINAL Rocketbox motion (the skinned Rocketbox avatar plus its Bip01 skeleton) at the same frames as a retargeted clip, and paste it next to the
rig/CT frames. Views use the same axes as the CT panels (front: person-left on the right of the image, as a viewer facing the person sees it; side: face to the left).

usage: python tools/rocketbox_original_panel.py AVATAR_ENGINE.glb CLIP_FRAMES_DIR OUT_DIR [--stride 3]
AVATAR_ENGINE.glb is the `engine/<clip>/<clip>.glb` file of codepawl/rocketbox-motions-glb (uncompressed avatar + clip); CLIP_FRAMES_DIR holds the NNN.png frames
written by experiments/render_clip.py; OUT_DIR gets the same frames with the original motion on the left.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0, os.path.dirname(__file__))
from rocketbox_to_clip import load_glb, accessor, unit

DRAW_BONES = [("Bip01 Pelvis", "Bip01 Spine"), ("Bip01 Spine", "Bip01 Spine1"), ("Bip01 Spine1", "Bip01 Spine2"), ("Bip01 Spine2", "Bip01 Neck"), ("Bip01 Neck", "Bip01 Head"),
              ("Bip01 Neck", "Bip01 L Clavicle"), ("Bip01 L Clavicle", "Bip01 L UpperArm"), ("Bip01 L UpperArm", "Bip01 L Forearm"), ("Bip01 L Forearm", "Bip01 L Hand"),
              ("Bip01 Neck", "Bip01 R Clavicle"), ("Bip01 R Clavicle", "Bip01 R UpperArm"), ("Bip01 R UpperArm", "Bip01 R Forearm"), ("Bip01 R Forearm", "Bip01 R Hand"),
              ("Bip01 Pelvis", "Bip01 L Thigh"), ("Bip01 L Thigh", "Bip01 L Calf"), ("Bip01 L Calf", "Bip01 L Foot"),
              ("Bip01 Pelvis", "Bip01 R Thigh"), ("Bip01 R Thigh", "Bip01 R Calf"), ("Bip01 R Calf", "Bip01 R Foot")]


CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_any(js, binc, i):
    """Any glTF accessor, honouring byteStride; normalised integer data is returned as float in 0..1."""
    a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]; dt = np.dtype(CT[a["componentType"]]); nc = NC[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0); stride = bv.get("byteStride") or dt.itemsize * nc
    raw = np.frombuffer(binc, np.uint8, count=(a["count"] - 1) * stride + dt.itemsize * nc, offset=off)
    out = np.stack([np.frombuffer(raw[k * stride:k * stride + dt.itemsize * nc].tobytes(), dt) for k in range(a["count"])]) if False else \
        np.lib.stride_tricks.as_strided(raw, shape=(a["count"], dt.itemsize * nc), strides=(stride, 1)).copy().view(dt).reshape(a["count"], nc)
    if a.get("normalized") and dt.kind in "ui":
        out = out / float(np.iinfo(dt).max)
    return out


def read_avatar(path, stride):
    js, binc = load_glb(path); nodes = js["nodes"]; parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []): parent[c] = i
    anim = js["animations"][0]; ch = {}
    for c in anim["channels"]:
        s = anim["samplers"][c["sampler"]]
        ch[(c["target"]["node"], c["target"]["path"])] = (accessor(js, binc, s["input"]), accessor(js, binc, s["output"]))
    dur = max(float(t.max()) for t, _ in ch.values()); times = np.arange(0, dur + 1e-6, 1 / 30.0)[::stride]; T = len(times)
    def track(i, path_, default):
        if (i, path_) not in ch: return np.tile(default, (T, 1))
        t, v = ch[(i, path_)]
        if len(t) == 1: return np.tile(v[0], (T, 1))
        if path_ == "rotation": return Slerp(t, Rot.from_quat(v))(np.clip(times, t[0], t[-1])).as_quat()
        return np.stack([np.interp(times, t, v[:, k]) for k in range(3)], 1)
    order, seen = [], set()
    def visit(i):
        if i in seen: return
        if i in parent: visit(parent[i])
        seen.add(i); order.append(i)
    for i in range(len(nodes)): visit(i)
    Wm = {}
    for i in order:
        nd = nodes[i]; M = np.tile(np.eye(4), (T, 1, 1))
        M[:, :3, :3] = Rot.from_quat(track(i, "rotation", np.array(nd.get("rotation", [0, 0, 0, 1]), np.float32))).as_matrix() * np.array(nd.get("scale", [1, 1, 1]), np.float32)[None, None, :]
        M[:, :3, 3] = track(i, "translation", np.array(nd.get("translation", [0, 0, 0]), np.float32))
        Wm[i] = M if i not in parent else Wm[parent[i]] @ M
    skin = js["skins"][0]; ibm = accessor_any(js, binc, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)       # glTF matrices are column-major
    V, J, Wt, F = [], [], [], []; base = 0
    for m in js["meshes"]:
        for p in m["primitives"]:
            at = p["attributes"]; V.append(accessor_any(js, binc, at["POSITION"]).astype(float))
            J.append(accessor_any(js, binc, at["JOINTS_0"]).astype(int)); Wt.append(accessor_any(js, binc, at["WEIGHTS_0"]).astype(float))
            F.append(accessor_any(js, binc, p["indices"]).reshape(-1, 3).astype(int) + base); base += len(V[-1])
    V = np.vstack(V); J = np.vstack(J); Wt = np.vstack(Wt); Wt = Wt / np.maximum(Wt.sum(1, keepdims=True), 1e-9); F = np.vstack(F)
    jn = skin["joints"]; Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
    verts = np.zeros((T, len(V), 3))
    for t in range(T):
        jm = np.stack([Wm[j][t] @ ibm[k] for k, j in enumerate(jn)])               # (80, 4, 4) joint matrices
        acc = np.zeros((len(V), 4))
        for k in range(4): acc += Wt[:, k:k + 1] * np.einsum("nij,nj->ni", jm[J[:, k]], Vh)
        verts[t] = acc[:, :3]
    byname = {nd["name"]: i for i, nd in enumerate(nodes)}
    bones = [(a, b) for a, b in DRAW_BONES if a in byname and b in byname]
    joints = {n: np.stack([Wm[byname[n]][t][:3, 3] for t in range(T)]) for n in {x for ab in bones for x in ab}}
    return verts, F, bones, joints, times


def to_mhr_axes(verts, joints):
    """Character frame from frame 0 (hips line, trunk up), as in tools/rocketbox_to_clip.py -> MHR axes (x person-left, y down, z back)."""
    mid = 0.5 * (joints["Bip01 L Thigh"][0] + joints["Bip01 R Thigh"][0])
    L = unit(joints["Bip01 L Thigh"][0] - joints["Bip01 R Thigh"][0]); U = unit(joints["Bip01 Neck"][0] - mid); U = unit(U - (U @ L) * L); F = np.cross(L, U)
    A = np.stack([L, -U, -F])
    return verts @ A.T, {k: v @ A.T for k, v in joints.items()}


def view(P, which):
    return np.stack([P[..., 0], -P[..., 1]], -1) if which == "front" else np.stack([P[..., 2], -P[..., 1]], -1)


def main():
    a = sys.argv[1:]
    glb, frames_dir, out = a[0], a[1], a[2]; stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 3
    os.makedirs(out, exist_ok=True)
    verts, F, bones, joints, times = read_avatar(glb, stride)
    verts, joints = to_mhr_axes(verts, joints)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    assert len(files) == len(times), f"clip has {len(files)} frames, original motion has {len(times)}"
    PW, PH = 300, 560
    lim = {w: (view(verts, w).reshape(-1, 2).min(0) - 0.08, view(verts, w).reshape(-1, 2).max(0) + 0.08) for w in ("front", "side")}
    light = np.array([-0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    for t, f in enumerate(files):
        im0 = Image.open(os.path.join(frames_dir, f)); H = im0.height
        panel = Image.new("RGB", (2 * PW + 6, H), (0, 0, 0)); d = ImageDraw.Draw(panel)
        d.text((8, 6), "ORIGINAL motion (what drives the rig)", fill=(255, 220, 120))
        d.text((8, 24), "Microsoft Rocketbox mocap, MIT: avatar skinned by its animation, red = Bip01 skeleton", fill=(170, 200, 230))
        d.text((8, 44), f"front", fill=(235, 235, 235)); d.text((PW + 14, 44), "side (face left)", fill=(235, 235, 235))
        for k, w in enumerate(("front", "side")):
            fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            lo, hi = lim[w]; ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); fig.patch.set_facecolor("black")
            tri = verts[t][F]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            sh = 0.35 + 0.65 * np.abs(nrm @ light); depth = tri[..., 2 if w == "front" else 0].mean(1)
            P2 = view(tri, w); order = np.argsort(-depth if w == "front" else depth)
            ax.add_collection(PolyCollection(P2[order], facecolors=np.stack([sh] * 3, 1)[order] * 0.8, edgecolors="none"))
            for A_, B_ in bones:
                s_ = view(np.stack([joints[A_][t], joints[B_][t]]), w); ax.plot(s_[:, 0], s_[:, 1], "-", color="#ff4d4d", lw=1.6)
            fig.canvas.draw(); img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
            panel.paste(img, (k * (PW + 6), 64))
        full = Image.new("RGB", (panel.width + im0.width, H)); full.paste(panel, (0, 0)); full.paste(im0, (panel.width, 0)); full.save(os.path.join(out, f))
    print("wrote", len(files), "frames to", out)


if __name__ == "__main__":
    main()
