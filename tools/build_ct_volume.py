"""Bring the atlas subject's REAL CT into the rig's rest frame, as a volume the page can deform live.

The CT has to be laid into the rest pose of the body model before the rig can move it. The fitted atlas (atlas/atlas.*) holds the
CT subject's structures as surfaces placed in that body, with vertices in the same order as meshes regenerated from the CT
segmentations, so every atlas vertex has a known place in the CT. Those pairs define the map from the rest frame back to the CT.

The map is deliberately SMOOTH: one affine map per height level along the body, fitted to the pairs near that level and blended
between levels. The image is then the original CT, gently stretched to the body, with nothing torn. (A first version followed every
pair locally, vertex by vertex. Because the skeleton fit moved neighbouring structures differently, that tore and smeared the image:
the offline renders on the project page looked far better because they keep the CT in its own frame and only apply motion to it.)

The price of smoothness: the CT's own bones do not land exactly on the atlas bone meshes (reported below per group), and the limb
bones the fit re-seated (humerus, scapula, clavicle, femur) are left out of the fit altogether.

usage: python tools/build_ct_volume.py SUBJECT_DIR atlas/atlas atlas/ct_volume [--voxel-mm 2.0]"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anatomy import atlas_build as ab  # noqa: E402
from anatomy.runtime import Atlas  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject"); ap.add_argument("atlas"); ap.add_argument("out")
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--level-mm", type=float, default=50.0, help="spacing of the height levels of the smooth map")
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
    X, Y, G = [], [], []
    used, skipped = 0, []
    RESEATED = ("humerus_", "scapula_", "clavicula_", "femur_")      # placed on the body model's joints, not where the CT has them
    for name in sorted(masks):
        if name not in by:
            continue
        mesh = ab.mask_to_mesh(masks[name], ct.affine, max(120, int(140000 * wts[name] / tot)))
        if mesh is None or len(mesh[0]) != len(by[name]["pos"]):
            skipped.append(name)
            continue
        x, y = mesh[0], by[name]["pos"]
        step = max(1, len(x) // 600)
        X.append(x[::step]); Y.append(y[::step]); G.append(np.full(len(x[::step]), by[name]["group"]))
        used += 1
    X, Y, G = np.vstack(X), np.vstack(Y), np.concatenate(G)
    fit = ~np.isin(G, ["shoulder"]) & ~np.array([False] * len(G))
    names_all = np.concatenate([np.full(len(by[n]["pos"][::max(1, len(by[n]["pos"]) // 600)]), n) for n in sorted(masks) if n in by and n not in skipped])
    fit = np.array([not str(n).startswith(RESEATED) for n in names_all])
    print(f"structures paired with the CT: {used} ({len(X)} vertex pairs, {int(fit.sum())} used for the map); not paired: {skipped if skipped else 'none'}")

    # ---- smooth map rest -> CT: an affine per height level, blended along the body
    sig = a.level_mm / 1000.0
    y_levels = np.arange(Y[fit, 1].min(), Y[fit, 1].max() + sig, sig)
    Yh = np.hstack([Y, np.ones((len(Y), 1))])
    Mg, *_ = np.linalg.lstsq(Yh[fit], X[fit], rcond=None)             # global affine: what sparse levels fall back to
    Ms = []
    for yl in y_levels:
        w = np.exp(-0.5 * ((Y[fit, 1] - yl) / sig) ** 2)
        sw = np.sqrt(w)[:, None]
        lam = 0.02 * w.sum() + 1e-6                                   # mild pull towards the global map
        Aw = np.vstack([Yh[fit] * sw, np.sqrt(lam) * np.eye(4)])
        Bw = np.vstack([X[fit] * sw, np.sqrt(lam) * Mg])
        M, *_ = np.linalg.lstsq(Aw, Bw, rcond=None)
        Ms.append(M)
    Ms = np.array(Ms)                                                # (L, 4, 3)

    def to_ct_mm(pts):
        wl = np.exp(-0.5 * ((pts[:, 1:2] - y_levels[None]) / sig) ** 2) + 1e-12
        wl /= wl.sum(1, keepdims=True)
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        return np.einsum("nl,nla->na", wl, np.einsum("nb,lba->nla", ph, Ms))

    res = np.linalg.norm(to_ct_mm(Y) - X, axis=1)
    print("how far the CT's own structures are from the atlas meshes under the smooth map (median / 95%, mm):")
    for g in sorted(set(G)):
        m = G == g
        print(f"    {g:11s} {np.median(res[m]):5.1f} / {np.percentile(res[m], 95):5.1f}")

    # ---- the scanned body only: the couch and anything else outside the largest connected body is air
    body = ndi.binary_opening(hu > -350, iterations=2)
    lab, n_lab = ndi.label(body)
    if n_lab > 1:
        body = lab == (1 + int(np.argmax(ndi.sum(body, lab, range(1, n_lab + 1)))))
    body = ndi.binary_dilation(body, iterations=2)
    hu = np.where(body, hu, -1000.0).astype(np.float32)

    # ---- rest-frame grid around the atlas
    v = a.voxel_mm / 1000.0
    lo, hi = Y[fit].min(0) - a.margin_mm / 1000, Y[fit].max(0) + a.margin_mm / 1000
    shape = np.ceil((hi - lo) / v).astype(int)
    print(f"rest-frame grid {shape.tolist()} at {a.voxel_mm} mm ({shape.prod() / 1e6:.1f} M voxels)")
    out = np.full(shape, -1000.0, np.float32)
    gx = lo[0] + (np.arange(shape[0]) + 0.5) * v
    gy = lo[1] + (np.arange(shape[1]) + 0.5) * v
    gz = lo[2] + (np.arange(shape[2]) + 0.5) * v
    for i in range(shape[0]):                                   # one sagittal slab at a time
        pts = np.stack(np.meshgrid([gx[i]], gy, gz, indexing="ij"), -1).reshape(-1, 3)
        vox = (to_ct_mm(pts) - b) @ Ainv.T
        out[i] = ndi.map_coordinates(hu, vox.T, order=1, mode="constant", cval=-1000.0).reshape(shape[1], shape[2])
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
