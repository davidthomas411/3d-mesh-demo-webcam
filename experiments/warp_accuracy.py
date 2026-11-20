"""How accurate is the soft-tissue warp? Held-out landmarks: the field is built from every 2nd vertex of every structure; the OTHER vertices (odd indices) are
mapped back through the field and compared with their true source positions in the CT. Median / 90th percentile error in mm per structure group, for the
chosen field method. usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/warp_accuracy.py CLIP.npz FRAME [method ...]"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
from mhr_clip import load_clip, prepare
from anatomy import atlas_build as ab


def src_at(fld, g):
    c = ((g - fld["origin"]) / fld["grid"]).T
    u = np.stack([ndi.map_coordinates(fld["u"][..., a], c, order=1, mode="nearest") for a in range(3)], 1)
    return g - u


if __name__ == "__main__":
    arr, fps, srcname = load_clip(sys.argv[1]); P = prepare(arr)
    frame = int(sys.argv[2]); methods = sys.argv[3:] or ["shepard"]
    bv.setup(); W = bv.W
    skin = bv.SESS.run(None, {"mhr_params": P[frame][None].astype(np.float32), "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = bv.posed_for("V4", W.atlas, skin, bv.PS)
    for m in methods:
        am = float(m.split("@")[1]) if "@" in m else 0.0; mm = m.split("@")[0]
        fld = W.field(posed, skin, method=mm.replace("-nosoft", ""), skin_half=True, soft="-nosoft" not in mm, attach_mm=am)
        errs = {}
        s_w, R_w, t_w = W.W
        names_ = [n for n in W.names if n in posed and n in W.by]
        Rall = W.local_rotations(np.vstack([W.by[n]["pos"] for n in names_]), np.vstack([posed[n] for n in names_])); Rl, o_ = {}, 0
        for n in names_:
            Rl[n] = Rall[o_:o_ + len(W.by[n]["pos"])]; o_ += len(W.by[n]["pos"])
        for n in W.names:
            if ab.is_bone(n) or n in ("brain",) or n.startswith(("humerus", "scapula", "clavicula", "femur")) or n not in posed:
                continue
            xs = W.v_ct[n][1::2]; p = posed[n][1::2]; rest = W.by[n]["pos"][1::2]
            q = (xs @ R_w.T) * s_w + t_w
            ys = (((p + np.einsum("nij,nj->ni", Rl[n][1::2], q - rest)) - t_w) @ R_w) / s_w if os.environ.get("CTW_RT_SOFT", "0") == "1" else xs + ((p - rest) @ R_w) / s_w   # same carry as the field's landmarks
            errs[n] = np.linalg.norm(src_at(fld, ys) - xs, axis=1)
        s_w, R_w, _ = W.W
        xs_ = W.skin_ct[1::2]; ys_ = W.skin_targets(skin, fld["bones"], am)[1::2]; es = np.linalg.norm(src_at(fld, ys_) - xs_, axis=1)
        zc = xs_[:, 2]; zr = np.percentile(zc, [0, 100]); band = lambda a, b: (zc >= zr[0] + a * (zr[1] - zr[0])) & (zc <= zr[0] + b * (zr[1] - zr[0]))
        print(f"[{m}] held-out SKIN points ({len(es)}): median {np.median(es):.1f} mm p90 {np.percentile(es, 90):.1f} p99 {np.percentile(es, 99):.1f};",
              "by height (feet..head thirds, z):", [round(float(np.median(es[band(a, b)])), 1) for a, b in ((0, .33), (.33, .66), (.66, 1))],
              " top 10% (head/neck):", round(float(np.median(es[band(.9, 1)])), 1), "p90", round(float(np.percentile(es[band(.9, 1)], 90)), 1))
        allv = np.concatenate(list(errs.values()))
        worst = sorted(errs.items(), key=lambda kv: -np.median(kv[1]))[:6]
        print(f"[{m}] frame {frame}: soft held-out landmarks {len(allv)}  median {np.median(allv):.1f} mm  p90 {np.percentile(allv, 90):.1f} mm  p99 {np.percentile(allv, 99):.1f} mm")
        print("   worst structures (median mm):", {n: round(float(np.median(e)), 1) for n, e in worst})
