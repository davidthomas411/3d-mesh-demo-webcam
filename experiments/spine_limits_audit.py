"""Do the posed per-level vertebra rotations stay inside physiological range-of-motion limits?
Relative rotation of each vertebra with respect to the one below (in the lower vertebra's frame), decomposed into axial (about MHR y),
lateral bend (about z) and flexion/extension (about x). Limits are ASSUMED conservative one-direction maxima in the range of the classic
in-vitro/in-vivo tables (White & Panjabi 1990; Pan et al. 2018 review) - not retrieved or verified in this session.
usage: ATLAS=.. REST=.. ONNX=.. python experiments/spine_limits_audit.py out.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from scipy.spatial.transform import Rotation as Rot
from benchmark_versions import VERSIONS, PerStructure, posed_for, motion_params
from anatomy.runtime import Atlas, kabsch

LIM = {"L": (13.0, 9.0, 5.0), "T": (8.0, 8.0, 9.0), "C": (20.0, 11.0, 9.0)}      # (flexion, lateral, axial) deg, one direction per level


def limits(upper):
    r = upper.replace("vertebrae_", "")
    f, l, a = LIM[r[0]]
    return (f, l, 45.0 if r == "C1" else a)


if __name__ == "__main__":
    import onnxruntime as ort
    atlas = Atlas(os.environ["ATLAS"]); rest = np.load(os.environ["REST"])["verts"].astype(float)
    sess = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"]); ps = PerStructure(atlas, rest)
    by = {s["name"]: s["pos"] for s in atlas.structures}
    ch = atlas.meta["chain"]
    res = {}
    for v in ("V3", "V4"):
        worst = {}
        for ph in (0, 2, 4):
            for u in np.linspace(0, 1, 9):
                sk = sess.run(None, {"mhr_params": motion_params(ph, u)[1], "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
                P = posed_for(v, atlas, sk, ps)
                Rs = {n: kabsch(by[n], P[n])[0] for n in ch}
                for lo, up in zip(ch[:-1], ch[1:]):
                    rel = Rot.from_matrix(Rs[lo].T @ Rs[up]).as_rotvec()              # lower-frame relative rotation (rest frames identical)
                    ax, lat, fl = np.degrees(np.abs(rel[1])), np.degrees(np.abs(rel[2])), np.degrees(np.abs(rel[0]))
                    fm, lm, am = limits(up)
                    k = f"{lo.replace('vertebrae_', '')}-{up.replace('vertebrae_', '')}"
                    w = worst.setdefault(k, {"flex": 0, "lat": 0, "axial": 0, "over": 0})
                    w["flex"], w["lat"], w["axial"] = max(w["flex"], fl), max(w["lat"], lat), max(w["axial"], ax)
                    w["over"] = max(w["over"], fl - fm, lat - lm, ax - am)
        viol = {k: {a: round(float(b), 1) for a, b in w.items()} for k, w in worst.items() if w["over"] > 0}
        res[v] = {"n_levels": len(worst), "n_over_limit": len(viol), "over": viol,
                  "max_deg": {a: round(max(w[a] for w in worst.values()), 1) for a in ("flex", "lat", "axial")}}
        print(v, "levels over limit:", len(viol), "of", len(worst), "| max per-level deg", res[v]["max_deg"], flush=True)
        for k, w in sorted(viol.items(), key=lambda kv: -kv[1]["over"])[:5]:
            print("   ", k, w)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
