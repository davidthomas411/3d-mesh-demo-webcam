"""Does the rig's bone motion agree with the MHR skin that drives it? (Atlas space, no CT needed.)
For every MHR skin vertex within NEAR mm of a bone at rest, move it with that bone's rigid motion (rest -> posed atlas vertices) and compare with where the posed
skin puts it. Skin lying on bone should move with it; the residual, per bone group, is the rig's inconsistency with its own input.
usage: ATLAS=.. REST=.. ONNX=.. python experiments/rig_skin_consistency.py CLIP.npz [frame frame ...] [--spine physio|skin]"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from anatomy.runtime import Atlas
from anatomy import atlas_build as ab
from mhr_clip import load_clip, prepare

NEAR = 0.025


def kabsch_rt(r, p):
    _, Q, t = ab.umeyama(r, p, with_scale=False)
    return Q, t


def group(n):
    if n == "skull": return "skull"
    if n.startswith("vertebrae_C"): return "cervical"
    if n.startswith("vertebrae_"): return "spine T/L"
    if n.startswith("rib"): return "ribs"
    if n in ("sternum",): return "sternum"
    if n.startswith(("hip", "sacrum")): return "pelvis"
    return None


def measure(atlas, mrest, skin, spine="physio", joints=True):
    posed = atlas.pose(skin, spine=spine, joints=joints)
    names = [s["name"] for s in atlas.structures if s["bind"] != "soft" and group(s["name"])]
    rest = {s["name"]: s["pos"] for s in atlas.structures}
    allr = np.vstack([rest[n] for n in names]); own = np.concatenate([[k] * len(rest[n]) for k, n in enumerate(names)])
    d, i = cKDTree(allr).query(mrest); near = d < NEAR
    out = {}
    for k, n in enumerate(names):
        m = near & (own[i] == k)
        if not m.any(): continue
        Q, t = kabsch_rt(rest[n], posed[n])
        out.setdefault(group(n), []).append(np.linalg.norm(mrest[m] @ Q.T + t - skin[m], axis=1))
    return {g: float(np.median(np.concatenate(v))) * 1000 for g, v in out.items()}, {g: int(sum(len(x) for x in v)) for g, v in out.items()}


if __name__ == "__main__":
    import onnxruntime as ort
    a = [x for x in sys.argv[2:] if not x.startswith("--")]
    arr, fps, src = load_clip(sys.argv[1]); P = prepare(arr)
    frames = [int(x) for x in a] or [0, len(P) // 3, len(P) // 2, 2 * len(P) // 3]
    spine = sys.argv[sys.argv.index("--spine") + 1] if "--spine" in sys.argv else "physio"
    atlas = Atlas(os.environ["ATLAS"]); mrest = np.load(os.environ["REST"])["verts"].astype(float)
    sess = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    tot = []
    for f in frames:
        skin = sess.run(None, {"mhr_params": P[f][None].astype(np.float32), "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
        med, n = measure(atlas, mrest, skin, spine)
        tot.append(med); print(f"frame {f:3d}: median skin-vs-bone residual (mm):", {g: round(v, 1) for g, v in med.items()})
    print("mean over frames:", {g: round(float(np.mean([t[g] for t in tot if g in t])), 1) for g in tot[0]})
