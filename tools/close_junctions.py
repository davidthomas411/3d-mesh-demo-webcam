"""Re-join the articulated bone groups at the spine's group boundaries by ONE joint solve, without pushing bones out through the skin.

atlas_build places each group (head_neck, thorax, lumbar, pelvis) with its own rigid transform fitted to sit inside the avatar's skin; the
junction vertebrae then end up far from where the CT subject's own anatomy puts them (broken neck: C7 to T1 gap 9 mm at rest and 22 mm
between centroids beyond the CT's; L1 to T12 gap 23 mm). The avatar's proportions differ from the CT subject's, so the mismatch has to be shared
between the skin fit and the joints.

Unknowns: a rigid correction (6 parameters, about its own centroid) for each of lumbar, thorax, head_neck (pelvis is the fixed reference).
Residuals:
  junction  centroid of each upper junction vertebra vs the CT's own offset from the lower one (CT offset rotated by the lower bone's new rotation)
  kink      relative rotation of the two junction bones vs the CT's (small weight: supine CT vs standing avatar differ in curvature)
  skin      core bones of each group staying their class margin below the avatar skin
  move      small penalty on the size of each correction
Structures bound to a group (the brain on the skull) and the CT correspondence file follow the same transforms.

usage: PYTHONPATH=. python tools/close_junctions.py ATLAS_PREFIX CORR.npz MHR_NPZ OUT_ATLAS_PREFIX OUT_CORR.npz [--wj W]
"""
import re, sys
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from anatomy import atlas_build as ab, rig

ORDER = ["lumbar", "thorax", "head_neck"]
JUNC = {"lumbar": ("sacrum", "vertebrae_L5", None), "thorax": ("vertebrae_L1", "vertebrae_T12", "lumbar"), "head_neck": ("vertebrae_T1", "vertebrae_C7", "thorax")}
CORE = {"lumbar": r"^vertebrae_L\d$", "thorax": r"^(vertebrae_T\d+|rib_|sternum|costal_cartilages)", "head_neck": r"^(skull|vertebrae_C\d)$"}


def gap(a, b):
    return float(cKDTree(a).query(b)[0].min())


def main(atlas_prefix, corr_in, mhr_npz, atlas_out, corr_out, wj=1.0):
    structs, meta = ab.load_meshes(atlas_prefix)
    by = {s["name"]: s for s in structs}
    corr = dict(np.load(corr_in, allow_pickle=True))
    z = np.load(mhr_npz); skin = trimesh.Trimesh(z["verts"].astype(float), z["faces"], process=False)
    bind = {e["name"]: e for e in meta["structures"]}
    rng = np.random.default_rng(0)
    members = {}
    for g in ORDER:
        m = {n for n in by if ab.is_bone(n) and rig.bone_group(n) == g}
        m |= {n for n, e in bind.items() if e.get("bind") == "frame" and e.get("frame") in m}
        members[g] = m
    Pg, Mg, piv = {}, {}, {}
    for g in ORDER:
        core = [n for n in by if re.match(CORE[g], n)]
        P = np.vstack([by[n]["verts"] for n in core])
        mg = np.concatenate([np.full(len(by[n]["verts"]), ab.MIN_DEPTH_MM.get(ab.group_of(n), ab.MIN_DEPTH_MM["default"]) / 1000.0) for n in core])
        pick = rng.choice(len(P), min(700, len(P)), replace=False)
        Pg[g], Mg[g] = P[pick], np.minimum(mg[pick], 0.006)
        piv[g] = np.vstack([by[n]["verts"] for n in members[g] if n in by]).mean(0)
    cen = {n: by[n]["verts"].mean(0) for n in by if n in ("sacrum", "vertebrae_L5", "vertebrae_L1", "vertebrae_T12", "vertebrae_T1", "vertebrae_C7")}
    Rct = {n: Rot.from_matrix(corr["T_" + n][1:10].reshape(3, 3)) for n in cen}
    sct = {n: corr["T_" + n][0] for n in cen}
    cct = {n: corr["v_" + n].mean(0) for n in cen}

    def tf(x, g, X):
        k = ORDER.index(g); return Rot.from_rotvec(x[6 * k:6 * k + 3]).apply(X - piv[g]) + piv[g] + x[6 * k + 3:6 * k + 6]

    def rot_of(x, g):
        return Rot.identity() if g is None else Rot.from_rotvec(x[6 * ORDER.index(g):6 * ORDER.index(g) + 3])

    def group_of_vertebra(n):
        return {"sacrum": None, "vertebrae_L5": "lumbar", "vertebrae_L1": "lumbar", "vertebrae_T12": "thorax", "vertebrae_T1": "thorax", "vertebrae_C7": "head_neck"}[n]

    def cen_new(x, n):
        g = group_of_vertebra(n)
        return cen[n] if g is None else tf(x, g, cen[n][None])[0]

    def resid(x):
        out = []
        for g in ORDER:
            lo, up, _ = JUNC[g]
            if g == "lumbar":
                lo_g = None
            else:
                lo_g = group_of_vertebra(lo)
            Rl = rot_of(x, lo_g) * Rct[lo]
            want = cen_new(x, lo) + sct[lo] * Rl.apply(cct[up] - cct[lo])
            out.append(wj * (cen_new(x, up) - want) / 0.002)
            Ru = rot_of(x, g) * Rct[up]
            out.append(0.5 * (Ru * Rl.inv()).as_rotvec() / np.radians(8.0))
            depth, _ = ab._skin_depth(skin, tf(x, g, Pg[g]))
            out.append(0.3 * np.maximum(0.0, Mg[g] - depth) / 0.003)
            k = ORDER.index(g)
            out.append(0.05 * x[6 * k + 3:6 * k + 6] / 0.01); out.append(0.05 * x[6 * k:6 * k + 3] / np.radians(5.0))
        return np.concatenate(out)

    before = {g: 100 * float((ab._skin_depth(skin, Pg[g])[0] < 0).mean()) for g in ORDER}
    gb = {g: 1000 * gap(by[JUNC[g][0]]["verts"], by[JUNC[g][1]]["verts"]) for g in ORDER}
    sol = least_squares(resid, np.zeros(18), diff_step=1e-3, max_nfev=50)
    x = sol.x
    print(f"{'junction':16s} {'CT gap':>7s} {'before':>7s} {'after':>7s}   core bones outside skin   move / rotation")
    for g in ORDER:
        k = ORDER.index(g); rv, tv = x[6 * k:6 * k + 3], x[6 * k + 3:6 * k + 6]
        Rc = Rot.from_rotvec(rv).as_matrix(); tc = piv[g] - Rc @ piv[g] + tv
        for n in members[g]:
            by[n]["verts"] = by[n]["verts"] @ Rc.T + tc
            if "T_" + n in corr:
                t = corr["T_" + n].copy(); R = t[1:10].reshape(3, 3)
                t[1:10] = (Rc @ R).reshape(-1); t[10:13] = Rc @ t[10:13] + tc
                corr["T_" + n] = t
    for g in ORDER:
        lo, up, _ = JUNC[g]; k = ORDER.index(g)
        outside = 100 * float((ab._skin_depth(skin, tf(x, g, Pg[g]))[0] < 0).mean())
        print(f"{lo[-3:]:>4s}|{up[-3:]:<4s} {g:9s} {gap(corr['v_' + lo], corr['v_' + up]):7.1f} {gb[g]:7.1f} {1000 * gap(by[lo]['verts'], by[up]['verts']):7.1f}"
              f"   {before[g]:5.1f}% -> {outside:5.1f}%   {1000 * np.linalg.norm(x[6 * k + 3:6 * k + 6]):5.1f} mm / {np.degrees(np.linalg.norm(x[6 * k:6 * k + 3])):4.1f} deg")
    ab.write_rigged_atlas(structs, mhr_npz, atlas_out, {"source": meta.get("source"), "landmarks_mm": meta.get("landmarks_mm")}, enforce=False)
    np.savez_compressed(corr_out, **corr)
    print("wrote", atlas_out, corr_out)


if __name__ == "__main__":
    a = sys.argv[1:]
    wj = float(a[a.index("--wj") + 1]) if "--wj" in a else 1.0
    main(*a[:5], wj=wj)
