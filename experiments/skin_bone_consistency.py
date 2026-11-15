"""Do the warp's landmark sets agree? Take each skin landmark within 20 mm of a bone and move it with that bone's RIGID motion; compare with where the skin
landmark actually goes (the posed MHR skin). Skin lying on bone (scalp, sternum, spine) should move with it: a large residual means the bone landmarks and the
skin landmarks contradict each other and the interpolated soft-tissue warp has to tear. (Comparing displacement vectors instead is wrong for rotating bodies.)
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/skin_bone_consistency.py CLIP.npz FRAME"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
from mhr_clip import load_clip, prepare
from anatomy import atlas_build as ab


def old_positions(W, n, p):
    s, R, t = W.T[n]; return ((p - t) @ R) / s


if __name__ == "__main__":
    arr, fps, src = load_clip(sys.argv[1]); P = prepare(arr)
    bv.setup(); W = bv.W
    skin = bv.SESS.run(None, {"mhr_params": P[int(sys.argv[2])][None].astype(np.float32), "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = bv.posed_for("V4", W.atlas, skin, bv.PS)
    s_w, R_w, _ = W.W
    Ys = W.skin_ct + (skin[W.skin_nn] - W.mhr_rest[W.skin_nn]) @ R_w / s_w
    names = [n for n in W.names if ab.is_bone(n) and not n.startswith(("humerus", "scapula", "clavicula", "femur"))]
    Xall = np.vstack([W.v_ct[n] for n in names]); own = np.concatenate([[k] * len(W.v_ct[n]) for k, n in enumerate(names)])
    d, i = cKDTree(Xall).query(W.skin_ct); near = d < 20
    for label, fn in (("bone motion via the global registration W (new)", lambda n: W.bone_ct_positions(n, posed[n])),
                      ("bone motion via the bone's own similarity (old)", lambda n: old_positions(W, n, posed[n]))):
        res = np.full(len(Ys), np.nan)
        for k, n in enumerate(names):
            m = near & (own[i] == k)
            if not m.any(): continue
            _, Rb, tb = ab.umeyama(W.v_ct[n], fn(n), with_scale=False)
            res[m] = np.linalg.norm(Ys[m] - (W.skin_ct[m] @ Rb.T + tb), axis=1)
        r = res[near]
        grp = lambda n: "skull" if n == "skull" else "cervical" if n.startswith("vertebrae_C") else "spine T/L" if n.startswith("vertebrae_") else "ribs" if n.startswith("rib") else "sternum/cartilage" if n in ("sternum", "costal_cartilages") else "pelvis" if n.startswith(("hip", "sacrum")) else "other"
        by_g = {}
        for k, n in enumerate(names):
            m = near & (own[i] == k)
            if m.any(): by_g.setdefault(grp(n), []).append(res[m])
        print("   by bone group (median mm):", {g: round(float(np.median(np.concatenate(v))), 1) for g, v in by_g.items()}, "n:", {g: int(sum(len(x) for x in v)) for g, v in by_g.items()})
        print(f"{label}: skin within 20 mm of bone n={int(near.sum())}: residual median {np.median(r):.1f} mm, p90 {np.percentile(r, 90):.1f} mm, p99 {np.percentile(r, 99):.1f} mm")
