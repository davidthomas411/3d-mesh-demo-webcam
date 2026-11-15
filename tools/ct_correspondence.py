"""CT-space <-> atlas correspondence for deforming the atlas subject's REAL CT with the rig.

Regenerates the CT-space structure meshes exactly as anatomy.atlas_build.build() does (deterministic, same
vertex order), then recovers, per structure, the similarity that takes CT-space vertices to the atlas rest
vertices, plus the global CT->rest similarity of the registration.

usage: PYTHONPATH=. python tools/ct_correspondence.py SUBJECT_DIR MHR_NPZ ATLAS_PREFIX OUT.npz
"""
import sys
import numpy as np
from anatomy import atlas_build as ab
from anatomy.runtime import Atlas

subject, mhr_npz, atlas_prefix, out = sys.argv[1:5]
z = np.load(mhr_npz)
mhr_v, mhr_j = z["verts"].astype(float), z["joints"].astype(float)
ct, masks = ab.load_subject(subject)
skin_v, _ = ab.extract_skin(ct)
warp, _ = ab.register_to_mhr(skin_v, masks, ct.affine, mhr_v, mhr_j, verbose=False)
vol = {n: m.sum() for n, (m, _) in masks.items()}
wts = {n: max(v, 1) ** (2 / 3) for n, v in vol.items()}
tot = sum(wts.values())
atlas = Atlas(atlas_prefix)
by = {s["name"]: s for s in atlas.structures}
res = {"names": [], "W": np.array([warp.s, 0, 0])}
bad = 0
for name in sorted(masks):
    mesh = ab.mask_to_mesh(masks[name], ct.affine, max(120, int(140000 * wts[name] / tot)))
    if mesh is None or name not in by:
        continue
    v_ct, _ = mesh
    rest = by[name]["pos"]
    if len(v_ct) != len(rest):
        bad += 1
        print("VERTEX COUNT MISMATCH", name, len(v_ct), len(rest))
        continue
    s, R, t = ab.umeyama(v_ct, rest, with_scale=True)
    err = np.linalg.norm(s * v_ct @ R.T + t - rest, axis=1)
    res["names"].append(name)
    res[f"v_{name}"] = v_ct
    res[f"T_{name}"] = np.concatenate([[s], R.reshape(-1), t])
    res[f"e_{name}"] = np.array([err.mean(), err.max()])
    if not ab.is_bone(name):
        pass
print("structures matched:", len(res["names"]), " mismatches:", bad)
for n in ("skull", "vertebrae_L3", "rib_left_5", "sternum", "scapula_left", "humerus_left", "liver"):
    if n in res["names"]:
        print(f"  {n:14s} similarity-fit residual mean/max (mm): {1000*res['e_'+n][0]:.2f} / {1000*res['e_'+n][1]:.2f}")
R = warp.R; ss = warp.s
# CT skin surface tied to the MHR skin: each CT skin vertex -> its image in the rest frame -> nearest MHR vertex
from scipy.spatial import cKDTree
skin_rest = warp(skin_v)
skin_nn = cKDTree(mhr_v).query(skin_rest)[1]
res["skin_ct"] = skin_v; res["skin_nn"] = skin_nn.astype(np.int32)
print("CT skin vertices:", len(skin_v), " median distance to the MHR skin after registration (mm):", round(1000 * float(np.median(np.linalg.norm(skin_rest - mhr_v[skin_nn], axis=1))), 1))
np.savez_compressed(out, names=np.array(res.pop("names")), Wsim=np.concatenate([[ss], R.reshape(-1), warp.t]),
                    **{k: v for k, v in res.items() if k != "W"})
print("global CT->rest similarity: scale", round(ss * 1000, 4), "(mm->m x1000) saved to", out)
