"""Retarget a Microsoft Rocketbox motion (GLB, 121-joint Bip01 skeleton, MIT licence; the codepawl/rocketbox-motions-glb dataset on Hugging Face)
onto MHR model parameters, as a clip file for this rig (tools/mhr_clip.py).

Method: forward kinematics of the Biped gives joint positions per frame; for a set of bone and body-axis DIRECTIONS (trunk, hip line, shoulder line,
upper arms, forearms, thighs, shins, head up and eye line) the 62 non-hand MHR body-pose parameters are fitted so the MHR70 keypoints give the
same directions (unit vectors, so skeleton proportions do not matter). The Biped's own pelvis motion, root translation and finger/face motion are not
transferred, and spine twist/bend is only constrained through the shoulder, hip and neck keypoints (the MHR keypoints have no mid-spine points); a small
prior spreads what is left. Not a motion-capture-grade retargeting: it is for driving the rig with real human motion.

usage: python tools/rocketbox_to_clip.py CLIP.glb OUT.npz --mhr-onnx mhr-body.onnx [--stride 3]
"""
import sys, json, struct
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

NAMES = {"pelvis": "Bip01 Pelvis", "neck": "Bip01 Neck", "head": "Bip01 Head", "lcl": "Bip01 L Clavicle", "rcl": "Bip01 R Clavicle",
         "lua": "Bip01 L UpperArm", "rua": "Bip01 R UpperArm", "lfa": "Bip01 L Forearm", "rfa": "Bip01 R Forearm", "lh": "Bip01 L Hand", "rh": "Bip01 R Hand",
         "lth": "Bip01 L Thigh", "rth": "Bip01 R Thigh", "lca": "Bip01 L Calf", "rca": "Bip01 R Calf", "lft": "Bip01 L Foot", "rft": "Bip01 R Foot",
         "leye": "Bip01 LEye", "reye": "Bip01 REye"}
# MHR70 keypoint indices (SAM 3D Body): nose 0, eyes 1/2, shoulders 5/6, elbows 7/8, hips 9/10, knees 11/12, ankles 13/14, right wrist 41, left wrist 62, neck 69
K = dict(nose=0, leye=1, reye=2, lsh=5, rsh=6, lel=7, rel=8, lhip=9, rhip=10, lkn=11, rkn=12, lan=13, ran=14, rwr=41, lwr=62, neck=69)
BODY0, NBODY = 6, 62                    # model_params[6:68] = body-pose indices 0..61 (non-hand); 62..115 are hand parameters, never fitted


def load_glb(path):
    b = open(path, "rb").read(); n = struct.unpack("<I", b[8:12])[0]; off = 12; js = binc = None
    while off < n:
        l, t = struct.unpack("<II", b[off:off + 8]); d = b[off + 8:off + 8 + l]; off += 8 + l
        if t == 0x4E4F534A: js = json.loads(d)
        elif t == 0x004E4942: binc = d
    return js, binc


def accessor(js, binc, i):
    a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
    nc = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    arr = np.frombuffer(binc, np.float32, a["count"] * nc, bv.get("byteOffset", 0) + a.get("byteOffset", 0))
    return arr.reshape(a["count"], nc) if nc > 1 else arr


def joint_positions(path, stride=3, fps=30.0):
    """-> {key: (T,3)} world positions of the named Biped joints on a 30 fps grid, plus the time axis."""
    js, binc = load_glb(path); nodes = js["nodes"]; parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []): parent[c] = i
    anim = js["animations"][0]; ch = {}
    for c in anim["channels"]:
        s = anim["samplers"][c["sampler"]]
        ch[(c["target"]["node"], c["target"]["path"])] = (accessor(js, binc, s["input"]), accessor(js, binc, s["output"]))
    dur = max(float(t.max()) for t, _ in ch.values()); times = np.arange(0, dur + 1e-6, 1.0 / fps)[::stride]
    def track(i, path_, default):
        if (i, path_) not in ch: return np.tile(default, (len(times), 1))
        t, v = ch[(i, path_)]
        if len(t) == 1: return np.tile(v[0], (len(times), 1))
        if path_ == "rotation":
            return Slerp(t, Rot.from_quat(v))(np.clip(times, t[0], t[-1])).as_quat()
        return np.stack([np.interp(times, t, v[:, k]) for k in range(3)], 1)
    order = []; seen = set()
    def visit(i):
        if i in seen: return
        if i in parent: visit(parent[i])
        seen.add(i); order.append(i)
    for i in range(len(nodes)): visit(i)
    Wm = {}
    T = len(times)
    for i in order:
        nd = nodes[i]
        tr = track(i, "translation", np.array(nd.get("translation", [0, 0, 0]), np.float32))
        q = track(i, "rotation", np.array(nd.get("rotation", [0, 0, 0, 1]), np.float32))
        sc = np.array(nd.get("scale", [1, 1, 1]), np.float32)
        M = np.tile(np.eye(4), (T, 1, 1)); M[:, :3, :3] = Rot.from_quat(q).as_matrix() * sc[None, None, :]; M[:, :3, 3] = tr
        Wm[i] = M if i not in parent else Wm[parent[i]] @ M
    byname = {nd["name"]: i for i, nd in enumerate(nodes)}
    miss = [v for v in NAMES.values() if v not in byname]
    if miss: raise SystemExit(f"skeleton lacks joints {miss}")
    return {k: Wm[byname[v]][:, :3, 3] for k, v in NAMES.items()}, times


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def biped_directions(P):
    """Direction targets in MHR axes (x person-left, y down, z backward) from Biped joint tracks; the character frame comes from frame 0."""
    mid = lambda a, b: 0.5 * (P[a] + P[b])
    L = unit((P["lth"] - P["rth"])[0]); U = unit((P["neck"] - mid("lth", "rth"))[0]); U = unit(U - (U @ L) * L); F = np.cross(L, U)
    A = np.stack([L, -U, -F])                                     # rows: MHR x, y, z axes expressed in the Biped world frame
    d = lambda a, b: unit((a - b) @ A.T)
    D = {"trunk": d(P["neck"], mid("lth", "rth")), "hips": d(P["lth"], P["rth"]), "shoulders": d(P["lua"], P["rua"]),
         "clavL": d(P["lua"], P["lcl"]), "clavR": d(P["rua"], P["rcl"]),
         "uarmL": d(P["lfa"], P["lua"]), "farmL": d(P["lh"], P["lfa"]), "uarmR": d(P["rfa"], P["rua"]), "farmR": d(P["rh"], P["rfa"]),
         "thighL": d(P["lca"], P["lth"]), "shinL": d(P["lft"], P["lca"]), "thighR": d(P["rca"], P["rth"]), "shinR": d(P["rft"], P["rca"]),
         "headUp": d(0.5 * (P["leye"] + P["reye"]), P["neck"]), "eyes": d(P["leye"], P["reye"])}
    return D


def mhr_directions(J):
    mid = lambda a, b: 0.5 * (J[a] + J[b])
    d = lambda a, b: unit(a - b)
    return {"trunk": d(J[K["neck"]], mid(K["lhip"], K["rhip"])), "hips": d(J[K["lhip"]], J[K["rhip"]]), "shoulders": d(J[K["lsh"]], J[K["rsh"]]),
            "clavL": d(J[K["lsh"]], J[K["neck"]]), "clavR": d(J[K["rsh"]], J[K["neck"]]),
            "uarmL": d(J[K["lel"]], J[K["lsh"]]), "farmL": d(J[K["lwr"]], J[K["lel"]]), "uarmR": d(J[K["rel"]], J[K["rsh"]]), "farmR": d(J[K["rwr"]], J[K["rel"]]),
            "thighL": d(J[K["lkn"]], J[K["lhip"]]), "shinL": d(J[K["lan"]], J[K["lkn"]]), "thighR": d(J[K["rkn"]], J[K["rhip"]]), "shinR": d(J[K["ran"]], J[K["rkn"]]),
            "headUp": d(mid(K["leye"], K["reye"]), J[K["neck"]]), "eyes": d(J[K["leye"]], J[K["reye"]])}


WEIGHT = {"trunk": 3.0, "hips": 1.0, "shoulders": 2.0, "clavL": 1.0, "clavR": 1.0, "uarmL": 2.0, "farmL": 1.5, "uarmR": 2.0, "farmR": 1.5,
          "thighL": 0.5, "shinL": 0.3, "thighR": 0.5, "shinR": 0.3, "headUp": 2.0, "eyes": 1.5}


def lm_fit(resid, x0, iters=30, lo=-2.5, hi=2.5, h=3e-3):
    """Levenberg-Marquardt with central-difference Jacobian. (scipy's least_squares stalled at poor local minima here: the ONNX model runs in float32 and
    forward differences were too noisy; this converges to ~0.3 deg on a test frame.)"""
    x = np.clip(np.asarray(x0, float), lo, hi); cost = 0.5 * (resid(x) ** 2).sum(); lam = 1e-2; n = len(x); E = np.eye(n)
    for _ in range(iters):
        r = resid(x); Jm = np.stack([(resid(x + h * E[i]) - resid(x - h * E[i])) / (2 * h) for i in range(n)], 1); A = Jm.T @ Jm; g = Jm.T @ r
        improved = False
        while lam < 1e6:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-6), g); x2 = np.clip(x + step, lo, hi); c2 = 0.5 * (resid(x2) ** 2).sum()
            if c2 < cost:
                improved = cost - c2 > 1e-7 * max(cost, 1e-9); x, cost = x2, c2; lam = max(lam / 3, 1e-7); break
            lam *= 4
        if not improved:
            break
    return x


def main():
    a = sys.argv[1:]
    glb, out = a[0], a[1]
    onnx = a[a.index("--mhr-onnx") + 1]; stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 3
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]); sh = np.zeros((1, 45), np.float32)
    P, times = joint_positions(glb, stride)
    D = biped_directions(P); keys = list(D)
    def mhr_joints(theta):
        p = np.zeros((1, 204), np.float32); p[0, BODY0:BODY0 + NBODY] = theta
        return sess.run(None, {"mhr_params": p, "shape_params": sh})[0][0]
    Tn = len(times); thetas = np.zeros((Tn, NBODY)); errs = np.zeros((Tn, len(keys)))
    prev = np.zeros(NBODY)
    for t in range(Tn):
        tgt = {k: D[k][t] for k in keys}
        def resid(th):
            m = mhr_directions(mhr_joints(th))
            r = [WEIGHT[k] * (m[k] - tgt[k]) for k in keys]
            return np.concatenate(r + [0.05 * th, 0.15 * (th - prev)])
        sol = lm_fit(resid, prev)
        thetas[t] = sol; prev = sol
        m = mhr_directions(mhr_joints(sol))
        errs[t] = [np.degrees(np.arccos(np.clip((m[k] * tgt[k]).sum(), -1, 1))) for k in keys]
        print(f"frame {t + 1}/{Tn}: mean direction error {errs[t].mean():.1f} deg, worst {keys[int(errs[t].argmax())]} {errs[t].max():.1f}", flush=True)
    P204 = np.zeros((Tn, 204), np.float32); P204[:, BODY0:BODY0 + NBODY] = thetas
    np.savez(out, mhr_model_params=P204, fps=30.0 / stride, direction_error_deg=errs, direction_keys=np.array(keys),
             source=f"Microsoft Rocketbox motion {glb.split('/')[-1]} (MIT) retargeted by direction fitting")
    print("wrote", out, P204.shape, "mean error over clip", round(float(errs.mean()), 1), "deg")


if __name__ == "__main__":
    main()
