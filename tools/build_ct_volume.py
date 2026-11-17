"""Bring the atlas subject's REAL CT into the rig's rest frame, as a volume the page can deform live, and make the atlas the
same CT.

The CT has to be laid into the rest pose of the body model before the rig can move it. The atlas as built from the CT
(atlas/ct_subject.*) holds the CT subject's structures in that frame, with vertices in the same order as meshes regenerated from
the CT segmentations, so every atlas vertex has a known place in the CT.

The CT is UN-POSED with the rig's own model: one similarity per spine-chain frame (pelvis, each vertebra, head), fitted to that
bone's pairs and blended by height exactly as the rig skins soft tissue. The trunk keeps the CT's own size (one scale); the head is
placed and sized (one uniform scale) onto the skull of the atlas fitted to the body model, so the neck bends to bring the scan's
head onto the model's head. Bones stay rigid and nothing is sheared. Three other maps were tried and dropped: following every
vertex pair tore the image (the atlas build moved neighbouring structures differently); an affine per height level sheared it; a
single similarity left the head where the scan's pillow put it, outside the model's head.

--align-atlas OUT rewrites the atlas so that it IS this CT: every structure's vertices are taken from the CT and carried into the
rest frame by the same map, and the joint centres follow their bones. The CT slice, the CT-bone overlay and the DRR are
then one coherent CT. The cost: the CT's own shapes and proportions are kept, so they are not fitted to the body model (the skull
does not fill the head, the humeri are where the scan had them); fitting bones to the body is the SKEL skeleton's job.

--pet-out PREFIX writes a PET volume through the same map: a real PET registered to the CT (--pet FILE, SUV), or, without one, a
SIMULATED uptake map made from the segmentation (typical FDG SUV per organ; physiological only, no lesions).

usage: python tools/build_ct_volume.py SUBJECT_DIR atlas/ct_subject atlas/ct_volume --fitted atlas/fitted --align-atlas atlas/atlas --pet-out atlas/ct_pet
       python tools/bind_mhr_joints.py atlas/atlas mhr_model.pt        (the aligned atlas needs its joint binding again)"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anatomy import atlas_build as ab  # noqa: E402
from anatomy.runtime import Atlas  # noqa: E402


def affine_fit(src, dst):
    """Least-squares affine dst ~ [src, 1] M; returns (M, linear part acting on column vectors)."""
    M, *_ = np.linalg.lstsq(np.hstack([src, np.ones((len(src), 1))]), dst, rcond=None)
    return M, M[:3].T


def align_atlas(src_prefix, out_prefix, ct_verts, to_rest, to_ct_mm):
    """Rewrite the atlas with every structure taken from the CT through the map. Joint centres, hinge axes and the rib-cage
    ellipsoid follow the bones they belong to (by the affine that takes each bone's old vertices to its new ones)."""
    import shutil
    from anatomy.runtime import ellipsoid_gap
    if os.path.abspath(src_prefix) != os.path.abspath(out_prefix):
        for ext in (".json", ".bin"):
            shutil.copyfile(src_prefix + ext, out_prefix + ext)
    meta = json.load(open(out_prefix + ".json"))
    blob = bytearray(open(out_prefix + ".bin", "rb").read())
    by = {e["name"]: e for e in meta["structures"]}
    old, new = {}, {}
    for name, e in by.items():
        old[name] = np.frombuffer(blob, np.float32, e["n_verts"] * 3, e["pos"]).reshape(-1, 3).astype(np.float64)
        new[name] = to_rest(ct_verts[name]) if name in ct_verts else old[name]
        blob[e["pos"]:e["pos"] + e["n_verts"] * 12] = new[name].astype("<f4").tobytes()
        e["centroid"] = [round(float(v), 5) for v in new[name].mean(0)]
    back = max(float(np.linalg.norm(to_ct_mm(new[n]) - ct_verts[n], axis=1).max()) for n in ct_verts)
    moved = np.concatenate([np.linalg.norm(new[n] - old[n], axis=1) for n in ct_verts]) * 1000
    print(f"atlas aligned to the CT: structures moved by median {np.median(moved):.1f} mm (95% {np.percentile(moved, 95):.1f} mm); "
          f"map inverted to within {back:.3f} mm")
    ribs = [n for n in by if n.startswith("rib_")]
    M_ribs, L_ribs = affine_fit(np.vstack([old[n] for n in ribs]), np.vstack([new[n] for n in ribs]))
    for j in meta.get("joints", []):
        ref = j["child"] if j["child"] in old else j["parent"]
        M, L = affine_fit(old[ref], new[ref])
        j["center"] = [round(float(v), 5) for v in np.r_[j["center"], 1.0] @ M]
        if "axis" in j:
            ax = L @ np.asarray(j["axis"]); j["axis"] = [round(float(v), 5) for v in ax / np.linalg.norm(ax)]
        if "ellipsoid" in j:                                   # rib-cage ellipsoid the scapula glides on
            e = j["ellipsoid"]
            e["center"] = [round(float(v), 5) for v in np.r_[e["center"], 1.0] @ M_ribs]
            e["radii"] = [round(float(v), 5) for v in np.asarray(e["radii"]) * np.abs(np.diag(L_ribs))]
            if "patch" in j:
                g, _ = ellipsoid_gap(new[j["child"]][np.asarray(j["patch"])], np.asarray(e["center"]), np.asarray(e["radii"]))
                j["gap0"], j["gapmin0"] = round(float(g.mean()), 5), round(float(g.min()), 5)
    if meta.get("mhr"):
        meta["mhr"]["limb_seated"] = False                     # the humeri are where the CT has them, not on the MHR joints
    meta["ct_aligned"] = True
    meta.setdefault("history", []).append("tools/build_ct_volume.py --align-atlas: structures taken from the CT through the smooth rest map")
    json.dump(meta, open(out_prefix + ".json", "w"), separators=(",", ":"))
    open(out_prefix + ".bin", "wb").write(bytes(blob))
    print(f"wrote {out_prefix}.json/.bin")


# Typical FDG uptake (SUV) per TotalSegmentator structure, for the SIMULATED PET (--pet-out without --pet). Physiological uptake
# only: no lesions are invented.
SIM_SUV = [("brain", 8.0), ("heart", 4.0), ("liver", 2.3), ("spleen", 1.9), ("kidney_", 3.2), ("urinary_bladder", 14.0),
           ("vertebrae_", 1.6), ("sacrum", 1.5), ("hip_", 1.3), ("sternum", 1.4), ("lung_", 0.4), ("stomach", 1.6), ("colon", 1.4),
           ("small_bowel", 1.5), ("thyroid_gland", 1.6), ("aorta", 1.5), ("autochthon_", 0.8), ("gluteus_", 0.7), ("iliopsoas_", 0.8),
           ("pancreas", 1.5), ("esophagus", 1.4), ("prostate", 1.8)]


def simulated_suv(hu, masks):
    """SUV on the CT grid from the segmentation: a soft-tissue background and a typical value per organ, blurred to PET resolution."""
    suv = np.where(hu > -300, 0.55, 0.0).astype(np.float32)
    for prefix, value in SIM_SUV:
        for name, (m, lo) in masks.items():
            if name.startswith(prefix):
                sl = tuple(slice(l, l + n) for l, n in zip(lo, m.shape))
                suv[sl][m] = value
    return suv


def skel_spine_targets(prefix):
    """Where the SKEL skeleton fitted to the body model has each vertebra, the pelvis and the skull (rest frame, metres).
    SKEL's cervical and lumbar vertebrae are separate structures; its thoracic vertebrae are the twelve midline pieces of the
    one rigid thorax mesh (the sternum pieces are in front, the ribs to the sides)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    sk = {s["name"]: s for s in Atlas(prefix).structures}
    tgt = {f"vertebrae_C{i}": sk[f"skel_cervical_{i}"]["pos"].mean(0) for i in range(1, 8)}
    tgt.update({f"vertebrae_L{i}": sk[f"skel_lumbar_{i}"]["pos"].mean(0) for i in range(1, 6)})
    th = sk["skel_thorax"]; P = th["pos"]; F = np.asarray(th["faces"]).reshape(-1, 3)
    n, lab = connected_components(coo_matrix((np.ones(2 * len(F)), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])), shape=(len(P), len(P))), directed=False)
    spine_z = np.mean([tgt["vertebrae_C7"][2], tgt["vertebrae_L1"][2]])
    cent = [P[lab == i].mean(0) for i in range(n)]
    mid = sorted([c for c in cent if abs(c[0] - tgt["vertebrae_C7"][0]) < 0.02 and abs(c[2] - spine_z) < 0.05], key=lambda c: c[1])
    if len(mid) != 12:
        raise SystemExit(f"expected 12 thoracic vertebrae in the SKEL thorax, found {len(mid)}")
    tgt.update({f"vertebrae_T{i + 1}": c for i, c in enumerate(mid)})
    tgt["pelvis"] = sk["skel_pelvis"]["pos"].mean(0)
    tgt["skull"] = (sk["skel_skull"]["pos"].mean(0), float(np.ptp(sk["skel_skull"]["pos"], 0).mean()))
    return tgt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject"); ap.add_argument("atlas"); ap.add_argument("out")
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--margin-mm", type=float, default=95.0)
    ap.add_argument("--fitted", default=None, help="atlas fitted to the body model (tools/fit_skeleton_to_mhr.py): where its skull is, is where the CT head goes")
    ap.add_argument("--skel", default=None, help="SKEL atlas (atlas/skel): register the CT's spine, pelvis and head onto SKEL's, vertebra by vertebra")
    ap.add_argument("--pet-out", default=None, help="also write a PET volume (uint8 SUV) in the same rest frame, e.g. atlas/ct_pet")
    ap.add_argument("--pet", default=None, help="PET NIfTI in SUV, registered to the CT (a real PET-CT). Without it the PET is SIMULATED from the segmentation")
    ap.add_argument("--pet-voxel-mm", type=float, default=4.0)
    ap.add_argument("--align-atlas", default=None, help="write an atlas whose structures are the CT's, placed by this map")
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
    ct_verts = {}                                               # every vertex of every paired structure, in the CT (mm)
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
        ct_verts[name] = x
        step = max(1, len(x) // 600)
        X.append(x[::step]); Y.append(y[::step]); G.append(np.full(len(x[::step]), by[name]["group"]))
        used += 1
    X, Y, G = np.vstack(X), np.vstack(Y), np.concatenate(G)
    fit = ~np.isin(G, ["shoulder"]) & ~np.array([False] * len(G))
    names_all = np.concatenate([np.full(len(by[n]["pos"][::max(1, len(by[n]["pos"]) // 600)]), n) for n in sorted(masks) if n in by and n not in skipped])
    BAD = ("brain",)                                               # malformed in this atlas (265 mm tall)
    fit = np.array([not str(n).startswith(RESEATED + BAD) for n in names_all])
    print(f"structures paired with the CT: {used} ({len(X)} vertex pairs, {int(fit.sum())} used for the map); not paired: {skipped if skipped else 'none'}")

    # ---- the map rest -> CT. The scan was taken in its own posture (supine, neck and head where the pillow put them), so the CT is
    # UN-POSED with the rig's own model: one similarity per spine-chain frame (pelvis, every vertebra, head), each fitted to that
    # bone's vertex pairs, blended between the two frames bracketing a point's height, exactly as the rig skins soft tissue
    # (anatomy/rig.py: soft_frame_weights) and as the live shader un-poses it. Bones stay rigid, nothing is sheared, and the head is
    # brought onto the body model's head by bending the neck.
    chain = atlas.meta.get("chain", [])
    lv = atlas.meta["spine_levels"]
    names_fit = names_all

    def umeyama(src, dst, scale=None):
        """dst ~ k R src + t (row vectors: dst = k src @ R.T + t). scale=None fits k, otherwise it is fixed."""
        sc, dc = src.mean(0), dst.mean(0)
        Uu, sv, Vt = np.linalg.svd((dst - dc).T @ (src - sc))
        Dd = np.diag([1, 1, np.sign(np.linalg.det(Uu @ Vt))])
        R = Uu @ Dd @ Vt
        kk = float((sv * np.diag(Dd)).sum() / ((src - sc) ** 2).sum()) if scale is None else scale
        return kk, R, dc - kk * sc @ R.T

    def pairs_of(names, Ysrc=None):
        m = np.isin(names_fit, names)
        return (Y if Ysrc is None else Ysrc)[m], X[m]

    trunk = np.isin(G, ["spine", "ribcage", "pelvis_leg"]) & fit
    k_body, _, _ = umeyama(Y[trunk], X[trunk])                       # one size for the whole trunk (mm per m)
    # ---- articulated registration onto the SKEL spine. The scan's own spinal curve (supine, a pillow under the head) is not the
    # curve of the body model: against SKEL fitted to the same body, T1 was 40 mm forward and L5 44 mm back. Each vertebra's
    # target is therefore moved onto SKEL's vertebra (centroid to centroid); the frames below are then fitted to those targets, so
    # each vertebra moves rigidly, takes its tilt from its neighbours, and soft tissue is interpolated between them. Bones are
    # not bent, which a free-form registration would do.
    Y_ct = Y
    pelvis_shift = np.zeros(3); skel_head = None
    if a.skel:
        tgt = skel_spine_targets(a.skel)
        Y = Y.copy(); moved = []
        for n in chain:
            if n in tgt:
                d = tgt[n] - np.asarray(by[n]["pos"]).mean(0); Y[names_fit == n] += d; moved.append(np.linalg.norm(d) * 1000)
        pel = [n for n in ("hip_left", "hip_right", "sacrum") if n in by]
        pelvis_shift = tgt["pelvis"] - np.vstack([by[n]["pos"] for n in pel]).mean(0)
        for n in pel + ["vertebrae_S1"]:
            Y[names_fit == n] += pelvis_shift
        skel_head = tgt["skull"]
        print(f"registered onto the SKEL spine: {len(moved)} vertebrae moved by {min(moved):.0f}-{max(moved):.0f} mm (median {np.median(moved):.0f}), "
              f"pelvis by {np.linalg.norm(pelvis_shift) * 1000:.0f} mm")
    frames = []                                                      # (k, R, t) per frame: pelvis, vertebrae bottom to top, head
    ys, xs = pairs_of(["hip_left", "hip_right", "sacrum", "vertebrae_S1"])
    frames.append(umeyama(ys, xs, k_body))
    for i, n in enumerate(chain):
        ys, xs = pairs_of(chain[max(0, i - 1):i + 2])                # the vertebra and its neighbours: a stable local frame
        frames.append(umeyama(ys, xs, k_body))
    # head: the skull as placed and sized to fill the body model's head (the fitted atlas), by ONE uniform scale (no stretching)
    Yhead = None
    if a.fitted:
        fa = Atlas(a.fitted); fby = {st["name"]: st for st in fa.structures}
        Yhead = Y.copy()
        m = names_fit == "skull"
        Yhead[m] = fby["skull"]["pos"][::max(1, len(fby["skull"]["pos"]) // 600)]
        if skel_head is not None:                                    # centred and sized as SKEL's skull, orientation as fitted
            fp = fby["skull"]["pos"]; c0 = fp.mean(0)
            Yhead[m] = skel_head[0] + (Yhead[m] - c0) * min(1.0, skel_head[1] / float(np.ptp(fp, 0).mean()))
    ys, xs = pairs_of(["skull"], Yhead)
    frames.append(umeyama(ys, xs))
    chain_y = [float(Y[names_fit == n][:, 1].mean()) for n in chain]      # rest heights of the frames (after the registration)
    levels = np.array([float(lv["pelvis"]) + float(pelvis_shift[1])] + chain_y + [chain_y[-1] - 0.045])
    if np.any(np.diff(levels) >= 0):
        raise SystemExit("chain levels are not strictly ascending in height after the registration")
    Ms = np.array([np.vstack([kk * R.T, t]) for kk, R, t in frames])  # (L, 4, 3): x = [y, 1] M
    Mg = Ms[len(Ms) // 2]
    print(f"CT un-posed with {len(frames)} chain frames: trunk scale {k_body / 1000:.3f}, head scale {frames[-1][0] / 1000:.3f}")

    def level_weights(yv):
        """Linear weights of the two chain frames bracketing each height (levels decrease upward: y is down)."""
        hi = np.clip(np.searchsorted(-levels, -yv, side="left"), 1, len(levels) - 1)
        lo = hi - 1
        t = np.clip((levels[lo] - yv) / np.maximum(levels[lo] - levels[hi], 1e-6), 0, 1)
        W = np.zeros((len(yv), len(levels)))
        r = np.arange(len(yv)); W[r, lo] = 1 - t; W[r, hi] += t
        return W

    def to_ct_mm(pts):
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        return np.einsum("nl,nla->na", level_weights(pts[:, 1]), np.einsum("nb,lba->nla", ph, Ms))

    res = np.linalg.norm(to_ct_mm(Y_ct) - X, axis=1)
    print("how far the CT's own structures are from the source atlas meshes under this placement (median / 95%, mm):")
    for g in sorted(set(G)):
        m = G == g
        print(f"    {g:11s} {np.median(res[m]):5.1f} / {np.percentile(res[m], 95):5.1f}")

    def to_rest(x_mm):
        """Inverse of to_ct_mm: the map is close to affine, so re-solving with the blend at the current estimate converges fast."""
        y = (x_mm - Mg[3]) @ np.linalg.inv(Mg[:3])
        for _ in range(30):
            M = np.einsum("nl,lba->nba", level_weights(y[:, 1]), Ms)   # (n, 4, 3): x = [y, 1] M
            y = np.einsum("na,nab->nb", x_mm - M[:, 3], np.linalg.inv(M[:, :3]))
        return y

    if a.align_atlas:
        align_atlas(a.atlas, a.align_atlas, ct_verts, to_rest, to_ct_mm)

    hu_raw = hu
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
    levels_out = [float(v) for v in levels]
    meta = {"shape": shape.tolist(), "origin": [float(x) for x in lo], "voxel": v, "units": "HU, float16, x fastest",
            "levels_y": levels_out, "hip_y": float(lv["hip_y"]),
            "frames": "pelvis segment, vertebrae bottom to top, head segment, uparm_l, uparm_r, thigh_l, thigh_r",
            "source": atlas.meta["source"], "paired_structures": used}
    json.dump(meta, open(a.out + ".json", "w"), separators=(",", ":"))
    out.transpose(2, 1, 0).astype(np.float16).tofile(a.out + ".bin")
    if a.pet_out:
        # PET goes through exactly the same map as the CT, on a coarser grid (PET resolution is ~5 mm anyway)
        if a.pet:
            import nibabel as nib
            pet = nib.load(a.pet); suv = np.asarray(pet.dataobj).astype(np.float32)
            Ap = ab.CT_TO_MHR @ pet.affine[:3, :3]; bp = ab.CT_TO_MHR @ pet.affine[:3, 3]
            kind = "measured: " + os.path.basename(a.pet)
        else:
            sp = float(np.abs(np.linalg.det(ct.affine[:3, :3])) ** (1 / 3))
            suv = ndi.gaussian_filter(np.where(body, simulated_suv(hu_raw, masks), 0.0), 2.5 / sp)
            Ap, bp = A, b
            kind = "SIMULATED from the CT segmentation (typical FDG uptake per organ, no lesions); not a measurement"
        Apinv = np.linalg.inv(Ap)
        pv = a.pet_voxel_mm / 1000.0
        pshape = np.ceil((hi - lo) / pv).astype(int)
        pout = np.zeros(pshape, np.float32)
        px, py, pz = (lo[i] + (np.arange(pshape[i]) + 0.5) * pv for i in range(3))
        for i in range(pshape[0]):
            pts = np.stack(np.meshgrid([px[i]], py, pz, indexing="ij"), -1).reshape(-1, 3)
            vox = (to_ct_mm(pts) - bp) @ Apinv.T
            pout[i] = ndi.map_coordinates(suv, vox.T, order=1, mode="constant", cval=0.0).reshape(pshape[1], pshape[2])
        scale = 16.0                                             # uint8 = SUV * 16 (up to SUV 15.9)
        pmeta = {"shape": pshape.tolist(), "origin": [float(x) for x in lo], "voxel": pv, "suv_scale": scale, "suv_window": 7.0,
                 "units": "SUV * 16, uint8, x fastest", "kind": kind, "simulated": not a.pet}
        json.dump(pmeta, open(a.pet_out + ".json", "w"), separators=(",", ":"))
        np.clip(pout.transpose(2, 1, 0) * scale, 0, 255).astype(np.uint8).tofile(a.pet_out + ".bin")
        print(f"wrote {a.pet_out}.json/.bin ({os.path.getsize(a.pet_out + '.bin') / 1e6:.1f} MB): PET {kind}; SUV max {pout.max():.1f}")
    body = out > -300
    print(f"wrote {a.out}.json/.bin ({os.path.getsize(a.out + '.bin') / 1e6:.1f} MB); body voxels {100 * body.mean():.0f}%, HU median inside {np.median(out[body]):.0f}")


if __name__ == "__main__":
    main()
