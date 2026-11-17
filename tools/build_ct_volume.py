"""Bring the atlas subject's REAL CT into the rig's rest frame, as a volume the page can deform live.

The fitted atlas (atlas/atlas.*) holds the CT subject's structures as surfaces, moved and resized to sit in the MHR body. Their
vertices are in the same order as meshes regenerated from the CT segmentations, so every atlas vertex has a known place in the CT.
Those pairs give a dense map from the rest frame back to the CT:

    for a rest-frame point y, take the nearest paired vertices y_i (rest) <-> x_i (CT); each proposes x_i + A_s (y - y_i), where A_s
    is the linear part of the best affine map rest -> CT of the structure the vertex belongs to; the proposals are averaged with
    Gaussian weights over the 24 nearest pairs.

It is close to exact at the paired vertices and smooth in between (structures the fit moved differently are blended over a couple of centimetres). The CT is resampled through it on a regular rest-frame grid and written
as half-float Hounsfield units; the page then only has to undo the live pose (anatomy/ctvol.js).

What it is not: tissue far from any segmented structure (subcutaneous fat, skin) is carried by the nearest structures; the arms are
not in this CT; and it is the CT of the atlas subject, not of the person in front of the camera.

usage: python tools/build_ct_volume.py SUBJECT_DIR atlas/atlas atlas/ct_volume [--voxel-mm 2.5]"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anatomy import atlas_build as ab  # noqa: E402
from anatomy.runtime import Atlas  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject"); ap.add_argument("atlas"); ap.add_argument("out")
    ap.add_argument("--voxel-mm", type=float, default=2.5)
    ap.add_argument("--margin-mm", type=float, default=70.0)
    a = ap.parse_args()

    atlas = Atlas(a.atlas)
    by = {s["name"]: s for s in atlas.structures}
    ct, masks = ab.load_subject(a.subject)
    hu = np.asarray(ct.dataobj).astype(np.float32)
    A = ab.CT_TO_MHR @ ct.affine[:3, :3]; b = ab.CT_TO_MHR @ ct.affine[:3, 3]      # voxel index -> mm in MHR axes
    Ainv = np.linalg.inv(A)

    # ---- paired vertices: CT (mm) <-> fitted atlas rest (m), structure by structure
    vol = {n: m.sum() for n, (m, _) in masks.items()}
    wts = {n: max(v, 1) ** (2 / 3) for n, v in vol.items()}
    tot = sum(wts.values())
    X, Y, Lin = [], [], []
    used, skipped = 0, []
    for name in sorted(masks):
        if name not in by:
            continue
        mesh = ab.mask_to_mesh(masks[name], ct.affine, max(120, int(140000 * wts[name] / tot)))
        if mesh is None or len(mesh[0]) != len(by[name]["pos"]):
            skipped.append(name)
            continue
        x, y = mesh[0], by[name]["pos"]
        # affine rest -> CT for this structure (least squares); its linear part carries the neighbourhood
        Yh = np.hstack([y - y.mean(0), np.ones((len(y), 1))])
        M, *_ = np.linalg.lstsq(Yh, x - x.mean(0), rcond=None)
        L = M[:3].T
        step = max(1, len(x) // 600)
        X.append(x[::step]); Y.append(y[::step]); Lin.append(np.repeat(L[None], len(x[::step]), 0))
        used += 1
    X, Y, Lin = np.vstack(X), np.vstack(Y), np.vstack(Lin)
    print(f"structures paired with the CT: {used} ({len(X)} vertex pairs); not paired: {skipped if skipped else 'none'}")

    # ---- rest-frame grid around the atlas
    v = a.voxel_mm / 1000.0
    P = np.vstack([s["pos"] for s in atlas.structures])
    lo, hi = P.min(0) - a.margin_mm / 1000, P.max(0) + a.margin_mm / 1000
    shape = np.ceil((hi - lo) / v).astype(int)
    print(f"rest-frame grid {shape.tolist()} at {a.voxel_mm} mm ({shape.prod() / 1e6:.1f} M voxels)")
    tree = cKDTree(Y)
    out = np.full(shape, -1000.0, np.float32)
    gx = lo[0] + (np.arange(shape[0]) + 0.5) * v
    gy = lo[1] + (np.arange(shape[1]) + 0.5) * v
    gz = lo[2] + (np.arange(shape[2]) + 0.5) * v
    K = 24
    for i in range(shape[0]):                                   # one sagittal slab at a time
        pts = np.stack(np.meshgrid([gx[i]], gy, gz, indexing="ij"), -1).reshape(-1, 3)
        d, idx = tree.query(pts, k=K)
        # Gaussian weights over a neighbourhood that widens away from the structures: neighbouring structures that the skeleton fit
        # moved differently are blended over a couple of centimetres instead of meeting at a tear
        sig = np.maximum(1.5 * d[:, :1], 0.018)
        w = np.exp(-0.5 * (d / sig) ** 2) + 1e-9; w /= w.sum(1, keepdims=True)
        prop = X[idx] + np.einsum("nkab,nkb->nka", Lin[idx], pts[:, None, :] - Y[idx])     # each neighbour's proposal (mm)
        x_mm = (w[..., None] * prop).sum(1)
        vox = (x_mm - b) @ Ainv.T
        val = ndi.map_coordinates(hu, vox.T, order=1, mode="constant", cval=-1000.0)
        val[d[:, 0] > 0.09] = -1000.0                           # nothing segmented within 9 cm: outside the scanned body
        out[i] = val.reshape(shape[1], shape[2])
        if i % 40 == 0:
            print(f"  slab {i}/{shape[0]}", flush=True)
    out = np.clip(out, -1024, 3071)

    # ---- how the rig moves the volume: soft tissue follows the two spine-chain frames bracketing its height (anatomy/rig.py)
    chain = atlas.meta.get("chain", [])
    lv = atlas.meta["spine_levels"]
    chain_y = [float(np.asarray(by[n]["centroid"])[1]) for n in chain]
    levels = [float(lv["pelvis"])] + chain_y + [chain_y[-1] - 0.045 if chain_y else float(lv["C2"]) - 0.03]
    meta = {"shape": shape.tolist(), "origin": [float(x) for x in lo], "voxel": v, "units": "HU, float16, x fastest",
            "levels_y": levels, "hip_y": float(lv["hip_y"]),
            "frames": "pelvis segment, vertebrae bottom to top, head segment, uparm_l, uparm_r, thigh_l, thigh_r",
            "source": atlas.meta["source"], "paired_structures": used}
    json.dump(meta, open(a.out + ".json", "w"), separators=(",", ":"))
    out.transpose(2, 1, 0).astype(np.float16).tofile(a.out + ".bin")
    body = out > -300
    print(f"wrote {a.out}.json/.bin ({os.path.getsize(a.out + '.bin') / 1e6:.1f} MB); body voxels {100 * body.mean():.0f}%, HU median inside {np.median(out[body]):.0f}")


if __name__ == "__main__":
    main()
