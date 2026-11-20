"""Audit of the premise 'no gaps between bones, no bones out of joints'.
For every pair of bones that touch (or nearly touch) at rest, measure how far the pair opens (gap grows) or interpenetrates
(overlap grows) in the posed model, relative to rest, across three motions and seven poses each, for model stages V1-V4.
Signed distance of A's surface points to B: positive = outside B, negative = inside B (sign from B's nearest-vertex normal).
usage: ATLAS=.. REST=.. ONNX=.. python experiments/joint_audit.py out.json"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from benchmark_versions import VERSIONS, PerStructure, posed_for, motion_params
from anatomy.runtime import Atlas
from anatomy import atlas_build as ab

NEAR_MM = 6.0       # pairs closer than this at rest count as articulating/contacting
TOL_MM = 3.0        # change beyond this is reported as a violation


def normals(P, F):
    n = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


import trimesh
from trimesh.proximity import signed_distance


def signed(PA, mesh_b, PB, NB, treeB):
    """Signed distance of points PA to mesh B (positive outside). Exact for watertight B; for open meshes (scapula, costal
    cartilages) it falls back to the nearest-vertex-normal sign test, which is only indicative."""
    if mesh_b is not None:
        return -signed_distance(mesh_b, PA)             # trimesh: positive inside
    d, j = treeB.query(PA)
    s = np.sign(((PA - PB[j]) * NB[j]).sum(1))
    return d * np.where(s == 0, 1, s)


def pair_stat(PA, mesh_b, PB, NB, tB, idx):
    sd = signed(PA[idx], mesh_b, PB, NB, tB)
    return float(np.abs(sd).min()), float(max(0.0, -sd.min()))          # gap (mm), overlap depth (mm)


if __name__ == "__main__":
    out = sys.argv[1]
    import onnxruntime as ort
    atlas = Atlas(os.environ["ATLAS"]); rest = np.load(os.environ["REST"])["verts"].astype(float)
    sess = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    ps = PerStructure(atlas, rest)
    by = {s["name"]: s for s in atlas.structures}
    bones = [n for n in by if ab.is_bone(n)]
    R = {n: by[n]["pos"] * 1000.0 for n in bones}
    F = {n: np.asarray(by[n]["faces"]).reshape(-1, 3) for n in bones}
    N0 = {n: normals(R[n], F[n]) for n in bones}
    trees = {n: cKDTree(R[n]) for n in bones}
    pairs = []
    for i, a in enumerate(bones):
        for b in bones[i + 1:]:
            d, ia = trees[b].query(R[a]);
            consecutive = {a, b} <= set(atlas.meta["chain"]) and abs(atlas.meta["chain"].index(a) - atlas.meta["chain"].index(b)) == 1
            if d.min() < NEAR_MM or consecutive or {a, b} == {"skull", "vertebrae_C1"}:   # adjacent vertebrae and skull|C1 always count
                idx = np.flatnonzero(d < max(NEAR_MM, d.min() + 3.0) + 15.0)              # contact patch of a (stays the same vertices as the model moves)
                pairs.append((a, b, idx, np.flatnonzero(trees[a].query(R[b])[0] < NEAR_MM + 15.0)))
    print("touching bone pairs at rest:", len(pairs), flush=True)
    def mk(P, n):
        m = trimesh.Trimesh(P, F[n], process=False)
        return m if m.is_watertight else None
    exact = {n: mk(R[n], n) is not None for n in bones}
    rest_stat = {(a, b): pair_stat(R[a], mk(R[b], b), R[b], N0[b], trees[b], ia) for a, b, ia, ib in pairs}
    skin_of = lambda p: sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    res = {}
    for v, name, _ in VERSIONS[1:]:
        worst = {}
        for ph, ts in ((0, np.linspace(0, 1, 7)), (2, np.linspace(0, 1, 7)), (4, np.linspace(0, 1, 7))):
            for u in ts:
                posed = posed_for(v, atlas, skin_of(motion_params(ph, u)[1]), ps)
                P = {n: posed[n] * 1000.0 for n in bones}
                NP = {}
                for a, b, ia, ib in pairs:
                    if b not in NP:
                        NP[b] = (normals(P[b], F[b]), cKDTree(P[b]), mk(P[b], b))
                    gap, ov = pair_stat(P[a], NP[b][2], P[b], NP[b][0], NP[b][1], ia)
                    g0, o0 = rest_stat[(a, b)]
                    k = (a, b)
                    w = worst.setdefault(k, {"open_mm": 0.0, "overlap_mm": 0.0})
                    w["open_mm"] = max(w["open_mm"], gap - g0 if ov == 0 and o0 == 0 else 0.0)
                    w["overlap_mm"] = max(w["overlap_mm"], ov - o0)
        rows = [{"pair": f"{a} | {b}", "exact": exact[b], **{k: round(x, 1) for k, x in w.items()}} for (a, b), w in worst.items()]
        res[v] = {"name": name, "n_pairs": len(rows),
                  "n_open": sum(r["open_mm"] > TOL_MM for r in rows), "n_overlap": sum(r["overlap_mm"] > TOL_MM for r in rows),
                  "worst_open": sorted(rows, key=lambda r: -r["open_mm"])[:6], "worst_overlap": sorted(rows, key=lambda r: -r["overlap_mm"])[:6]}
        print(f"{v} {name}: pairs {len(rows)} | opened > {TOL_MM:.0f} mm: {res[v]['n_open']} | overlapping > {TOL_MM:.0f} mm: {res[v]['n_overlap']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
