"""Rig validation on REAL MHR articulations (needs mhr-body.onnx; not run in CI).

usage: PYTHONPATH=. python tests/validate_rig.py ATLAS_V2_PREFIX ATLAS_V1_PREFIX_OR_- mhr-body.onnx mhr_rest.npz [outdir]
Metrics per pose (lower is better), on ~2500 atlas sample points:
  depth_dev : mean |change in distance-below-skin| vs rest, mm  (does anatomy keep its depth under the skin?)
  outside   : fraction of points that end up outside the posed skin, minus the rest-pose fraction
  bone_rig  : worst bone shape change (max pairwise-distance change within a bone), mm
  spine_rel : largest relative rotation between neighbouring vertebrae, deg (smooth distribution of bend/twist)
"""
import sys, json, numpy as np
import onnxruntime as ort, trimesh
from anatomy.runtime import Atlas, kabsch

v2, v1, onnx_path, rest_npz = sys.argv[1:5]
outdir = sys.argv[5] if len(sys.argv) > 5 else None
atlas = Atlas(v2)
z = np.load(rest_npz); rest_v, faces = z["verts"].astype(np.float64), z["faces"]
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
def mhr(params):
    p = np.zeros((1, 204), np.float32)
    for i, v in params.items(): p[0, i] = v
    return sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(np.float64)

# ---- old method (per-structure rigid fit on its own nearest-skin patch), built on the SAME rest geometry
from scipy.spatial import cKDTree
_tree = cKDTree(rest_v); old = []
for s in atlas.structures:
    r_ = np.random.default_rng(0); pick = r_.choice(len(s["pos"]), min(600, len(s["pos"])), replace=False)
    _, nn = _tree.query(s["pos"][pick], k=3); ids, cnt = np.unique(nn.ravel(), return_counts=True)
    o = np.argsort(-cnt)[:192]
    old.append({"name": s["name"], "pos": s["pos"], "idx": ids[o], "w": cnt[o].astype(float), "rest": rest_v[ids[o]]})
def old_pose(skin):
    out = {}
    for s in old:
        R, cs, cd = kabsch(s["rest"], skin[s["idx"]], s["w"]); out[s["name"]] = (s["pos"] - cs) @ R.T + cd
    return out

skin_rest = trimesh.Trimesh(rest_v, faces, process=False)
rng = np.random.default_rng(0)
names = atlas.names()
S = {}
for s in atlas.structures:
    S[s["name"]] = rng.choice(len(s["pos"]), min(40, len(s["pos"])), replace=False)
rest_pts = np.vstack([atlas.structures[i]["pos"][S[n]] for i, n in enumerate(names)])
lab = np.concatenate([[n] * len(S[n]) for n in names])

def depth_inside(mesh, pts):
    """Signed depth below the skin (+ = inside) by the normal-direction test (robust to the
    non-watertight MHR mesh, unlike ray-parity contains())."""
    d_all, ins_all = [], []
    for i in range(0, len(pts), 1500):
        c, d, tri = trimesh.proximity.closest_point(mesh, pts[i:i + 1500])
        sgn = np.einsum("ij,ij->i", pts[i:i + 1500] - c, mesh.face_normals[tri]) < 0
        d_all.append(np.where(sgn, d, -d)); ins_all.append(sgn)
    return np.concatenate(d_all), np.concatenate(ins_all)

d0, in0 = depth_inside(skin_rest, rest_pts)
print(f"rest: {100 * (~in0).mean():.1f}% of sample points outside the MHR skin; median depth {1000*np.median(d0[in0]):.0f} mm")

def evaluate(posed_dict, skin_posed):
    pts = np.vstack([posed_dict[n][S[n]] for n in names])
    mesh = trimesh.Trimesh(skin_posed, faces, process=False)
    d1, in1 = depth_inside(mesh, pts)
    newly = (~in1) & in0
    worst = sorted(((np.sum(newly & (lab == n)), n) for n in set(lab[newly])), reverse=True)[:3]
    return {"depth_dev": 1000 * np.abs(d1 - d0).mean(), "outside": 100 * ((~in1).mean() - (~in0).mean()),
            "worst": [(n, int(c)) for c, n in worst]}

def bone_rigidity(posed):
    worst = 0.0
    for s in atlas.structures:
        if s["bind"] != "bone": continue
        a, b = s["pos"][::7], posed[s["name"]][::7]
        da = np.linalg.norm(a[:, None] - a[None], axis=-1); db = np.linalg.norm(b[:, None] - b[None], axis=-1)
        worst = max(worst, np.abs(da - db).max())
    return 1000 * worst

def link_gaps(posed):
    """max change (mm) of vertebra-vertebra spacing and rib-to-vertebra distance: a coherent
    skeleton keeps these constant; independent per-structure fits do not."""
    cen = lambda P, n: P[n].mean(0)
    rest = {s["name"]: s["pos"] for s in atlas.structures}
    ch = [f"vertebrae_{x}" for x in ["C3","C4","C5","C6","C7","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5"]]
    worst = 0.0
    for a_, b_ in zip(ch[:-1], ch[1:]):
        if a_ in rest and b_ in rest:
            worst = max(worst, abs(np.linalg.norm(cen(posed, a_) - cen(posed, b_)) - np.linalg.norm(cen(rest, a_) - cen(rest, b_))))
    for side in ("left", "right"):
        for k in range(1, 13):
            r, v = f"rib_{side}_{k}", f"vertebrae_T{k}"
            if r in rest and v in rest:
                worst = max(worst, abs(np.linalg.norm(cen(posed, r) - cen(posed, v)) - np.linalg.norm(cen(rest, r) - cen(rest, v))))
    return 1000 * worst


def spine_rel(posed):
    from anatomy.runtime import rot_to_quat
    chain = [f"vertebrae_{x}" for x in ["C1","C2","C3","C4","C5","C6","C7","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5"]]
    rots = {}
    for s in atlas.structures:
        if s["name"] in chain:
            R, _, _ = kabsch(s["pos"], posed[s["name"]]); rots[s["name"]] = R
    ang = []
    for a, b in zip(chain[:-1], chain[1:]):
        if a in rots and b in rots:
            Rrel = rots[a] @ rots[b].T; ang.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    return max(ang) if ang else float("nan")

poses = {
    "spine_bend_A": {8: 0.35, 14: 0.25},
    "spine_bend_B": {10: 0.35, 16: 0.25},
    "spine_mix":    {8: 0.3, 10: -0.3, 14: 0.2, 19: 0.3},
    "neck_head":    {24: 0.4, 25: 0.5, 26: -0.4},
    "r_arm":        {31: 0.7, 32: -0.5},
    "root_rot":     {3: 0.3, 5: -0.25},
    "all_mix":      {8: 0.25, 16: 0.25, 19: 0.2, 25: 0.3, 31: 0.5, 3: 0.15},
}
print(f"{'pose':14s} {'method':6s} depth_dev  outside  link_gap  spine_rel")
for pname, prm in poses.items():
    skin = mhr(prm)
    if outdir: skin.astype("<f4").tofile(f"{outdir}/{pname}.bin")
    new = atlas.pose(skin); m = evaluate(new, skin)
    print(f"{pname:14s} {'rig':6s} {m['depth_dev']:7.1f}mm {m['outside']:+7.1f}% {link_gaps(new):7.1f}mm {spine_rel(new):7.1f}deg  worst-outside {m['worst']}")
    if True:
        po = old_pose(skin); m = evaluate(po, skin)
        print(f"{'':14s} {'old':6s} {m['depth_dev']:7.1f}mm {m['outside']:+7.1f}% {link_gaps(po):7.1f}mm {spine_rel(po):7.1f}deg  worst-outside {m['worst']}")
