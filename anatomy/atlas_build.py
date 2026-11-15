"""Build a TotalSegmentator-class anatomy atlas bound to the MHR (SAM 3D Body) skin mesh.

Input : one TotalSegmentator-dataset subject (ct.nii.gz + segmentations/*.nii.gz, ground truth
        or model output) and an MHR rest mesh (+ joints) from mhr-body.onnx with zero params.
Output: atlas.json + atlas.bin consumed by anatomy/anatomy.js and anatomy/runtime.py.

Frames: MHR rest = metres, y down, +x person-left, front = -z. CT = RAS mm.
"""
import glob, json, os
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

from .classes import group_of, GROUP_COLOR, UNCERTAINTY_MM, display_name

CT_TO_MHR = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)   # RAS -> (left, down, back)


# ---------------------------------------------------------------- loading
def load_subject(subject_dir):
    import nibabel as nib
    ct = nib.load(os.path.join(subject_dir, "ct.nii.gz"))
    aff = ct.affine
    masks = {}
    for f in sorted(glob.glob(os.path.join(subject_dir, "segmentations", "*.nii.gz"))):
        m = np.asanyarray(nib.load(f).dataobj) > 0
        if not m.any():
            continue
        idx = np.argwhere(m)
        lo, hi = idx.min(0), idx.max(0) + 1
        masks[os.path.basename(f)[:-7]] = (m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], lo)
    return ct, masks


def vox_to_mhr_mm(ijk, affine):
    ras = nib_apply(affine, ijk)
    return ras @ CT_TO_MHR.T          # mm, MHR axes


def nib_apply(aff, pts):
    return pts @ aff[:3, :3].T + aff[:3, 3]


def mask_points(entry, affine):
    m, lo = entry
    return vox_to_mhr_mm(np.argwhere(m) + lo, affine)


def mask_surface_points(entry, affine):
    m, lo = entry
    shell = m & ~ndi.binary_erosion(m)
    return vox_to_mhr_mm(np.argwhere(shell) + lo, affine)


# ---------------------------------------------------------------- skin
def extract_skin(ct, step=2, thresh=-300, open_r=4):
    """Body surface mesh (mm, MHR axes) from the CT by threshold + opening (removes couch/cables)."""
    from skimage import measure
    data = np.asanyarray(ct.dataobj)[::step, ::step, ::step]
    body = data > thresh
    body = ndi.binary_opening(body, structure=ndi.generate_binary_structure(3, 1), iterations=open_r)
    lab, n = ndi.label(body)
    if n == 0:
        raise RuntimeError("no body found")
    sizes = ndi.sum(body, lab, range(1, n + 1))
    body = lab == (1 + int(np.argmax(sizes)))
    for k in range(body.shape[2]):
        body[:, :, k] = ndi.binary_fill_holes(body[:, :, k])
    body = ndi.gaussian_filter(body.astype(np.float32), 1.0)
    pad = np.pad(body, 1)
    v, f, _, _ = measure.marching_cubes(pad, 0.5)
    v = (v - 1) * step
    return vox_to_mhr_mm(v, ct.affine), f.astype(np.int32)


# ---------------------------------------------------------------- landmarks
def _sphere_fit(p):
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 0)))


def joint_head_center(surface_pts, top_axis_sign=-1, r_range=(17, 30), slab=70, iters=500, seed=0):
    """Femoral/humeral head centre: RANSAC sphere fit on bone-surface points of the superior slab
    (MHR y is down, so superior = -y). Returns (centre_mm, radius_mm, inlier_fraction)."""
    rng = np.random.default_rng(seed)
    y = surface_pts[:, 1] * top_axis_sign
    surf = surface_pts[y > y.max() - slab]
    best, best_in = None, -1
    for _ in range(iters):
        c, r = _sphere_fit(surf[rng.choice(len(surf), 4, replace=False)])
        if not (r_range[0] <= r <= r_range[1]):
            continue
        inl = int((np.abs(np.linalg.norm(surf - c, axis=1) - r) < 1.5).sum())
        if inl > best_in:
            best, best_in = (c, r), inl
    if best is None:                                   # fall back: centroid of the superior cap
        return surf[surf[:, 1] * top_axis_sign > y.max() - 25].mean(0), 0.0, 0.0
    return best[0], best[1], best_in / len(surf)


def ct_landmarks(masks, affine):
    pts = lambda n: mask_points(masks[n], affine) if n in masks else None
    srf = lambda n: mask_surface_points(masks[n], affine) if n in masks else None
    out = {}
    for side, key in (("left", "l"), ("right", "r")):
        fem, hum = srf(f"femur_{side}"), srf(f"humerus_{side}")
        if fem is not None:
            out[f"{key}Hip"] = joint_head_center(fem)[0]
        if hum is not None:
            out[f"{key}Shoulder"] = joint_head_center(hum)[0]
    c7 = pts("vertebrae_C7")
    if c7 is not None:
        out["neck"] = c7.mean(0)
    return out


MHR_KP = {"lShoulder": 5, "rShoulder": 6, "lHip": 9, "rHip": 10, "neck": 69}


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d])
    R = U @ E @ Vt
    s = (sig * np.array([1, 1, d])).sum() / ((S ** 2).sum() / len(src)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------- registration
class Warp:
    """CT-space (mm, MHR axes) -> MHR rest space (m). Similarity followed by RBF residual fields."""
    def __init__(self, s, R, t):
        self.s, self.R, self.t = s, R, t
        self.fields = []        # list of RBFInterpolator on already-warped coordinates

    def __call__(self, x):
        y = self.s * (x @ self.R.T) + self.t
        for f in self.fields:
            y = y + f(y)
        return y


def register_to_mhr(skin_v_mm, masks, affine, mhr_v, mhr_joints, rounds=(8.0, 3.0, 1.0), n_ctrl=900,
                    verbose=True):
    lm = ct_landmarks(masks, affine)
    names = [k for k in lm if k in MHR_KP]
    src = np.array([lm[k] for k in names]) / 1000.0
    dst = np.array([mhr_joints[MHR_KP[k]] for k in names])
    s, R, t = umeyama(src, dst)
    warp = Warp(s / 1000.0, R, t)                 # similarity fitted metres->metres; CT input is mm
    if verbose:
        err = np.linalg.norm(warp(src * 1000.0) - dst, axis=1)
        print(f"landmarks {names}  similarity scale={s:.3f}  residual (mm) {np.round(err * 1000, 1)}")

    # torso band + lateral limit from the CT's own ribcage so arms never drive the warp
    ys = dst[:, 1]
    y_lo = dst[[names.index(k) for k in names if k.endswith("Shoulder")], 1].mean() if any(k.endswith("Shoulder") for k in names) else ys.min()
    y_hi = dst[[names.index(k) for k in names if k.endswith("Hip")], 1].mean() + 0.05 if any(k.endswith("Hip") for k in names) else ys.max()
    x_lim = np.abs(dst[:, 0]).max() * 1.05 if len(dst) else 0.25
    rib_pts = np.vstack([mask_points(masks[n], affine) for n in masks if n.startswith(("rib_", "sternum"))]) \
        if any(n.startswith(("rib_", "sternum")) for n in masks) else None
    if rib_pts is not None:
        x_lim = min(x_lim, np.abs(warp(rib_pts)[:, 0]).max() + 0.04)
    def in_band(p):
        torso = (p[:, 1] > y_lo) & (p[:, 1] < y_hi) & (np.abs(p[:, 0]) < x_lim)
        head = (p[:, 1] <= y_lo + 0.02) & (np.abs(p[:, 0]) < 0.12) & (p[:, 1] > y_lo - 0.45)   # head + neck
        return torso | head

    mhr_tree = cKDTree(mhr_v)
    mhr_sel = np.flatnonzero(in_band(mhr_v))
    if verbose:
        print(f"torso+head band: y in [{y_lo:.2f},{y_hi:.2f}] |x|<{x_lim:.2f}; MHR verts {len(mhr_sel)}")
    sub_tree = cKDTree(mhr_v[mhr_sel])
    rng = np.random.default_rng(0)
    for smooth in rounds:
        cur = warp(skin_v_mm)
        sel = np.flatnonzero(in_band(cur))
        sel = rng.choice(sel, min(n_ctrl, len(sel)), replace=False)
        d, j = sub_tree.query(cur[sel])
        tgt = mhr_v[mhr_sel[j]]
        keep = d < np.percentile(d, 85)
        ctrl, disp = cur[sel][keep], (tgt - cur[sel])[keep]
        # landmarks stay anchored (identity prior far field is implicit: smoothing + degree-1 poly)
        f = RBFInterpolator(ctrl, disp, kernel="thin_plate_spline", smoothing=smooth * 1e-3, degree=1)
        warp.fields.append(f)
        if verbose:
            after = warp(skin_v_mm)
            sel2 = np.flatnonzero(in_band(after))
            d2, _ = sub_tree.query(after[sel2])
            print(f"  round smooth={smooth}: torso skin->MHR  mean {d2.mean() * 1000:.1f} mm  p95 {np.percentile(d2, 95) * 1000:.1f} mm")
    return warp, lm


# ---------------------------------------------------------------- meshes
def mask_to_mesh(entry, affine, tri_budget, smooth_iters=10):
    """Marching cubes -> Taubin smooth -> quadric decimation. Vertices in mm, MHR axes."""
    from skimage import measure
    import trimesh, fast_simplification
    m, lo = entry
    pad = np.pad(m.astype(np.float32), 2)
    pad = ndi.gaussian_filter(pad, 0.7)
    if pad.max() < 0.5:
        return None
    v, f, _, _ = measure.marching_cubes(pad, 0.5)
    v = v - 2 + lo
    mesh = trimesh.Trimesh(vox_to_mhr_mm(v, affine), f[:, ::-1].copy() if np.linalg.det(CT_TO_MHR) < 0 else f, process=True)
    # CT_TO_MHR is a proper rotation, but voxel->RAS may flip handedness; fix winding by volume sign
    if mesh.volume < 0:
        mesh.invert()
    if smooth_iters:
        import trimesh.smoothing as sm
        sm.filter_taubin(mesh, iterations=smooth_iters)
    nf = len(mesh.faces)
    if nf > tri_budget:
        vs, fs = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                              target_reduction=1.0 - tri_budget / nf)
        return vs.astype(np.float64), fs.astype(np.int32)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, np.int32)


# ---------------------------------------------------------------- containment
MIN_DEPTH_MM = {"skull": 4, "spine": 22, "ribcage": 10, "shoulder": 6, "pelvis_leg": 8, "muscle": 5,
                "neuro_endo": 6, "default": 8}


def _skin_depth(mesh, pts, chunk=3000):
    import trimesh
    d_all = []
    for i in range(0, len(pts), chunk):
        c, d, tri = trimesh.proximity.closest_point(mesh, pts[i:i + chunk])
        inside = np.einsum("ij,ij->i", pts[i:i + chunk] - c, mesh.face_normals[tri]) < 0
        d_all.append((np.where(inside, d, -d), -mesh.face_normals[tri]))
    return np.concatenate([a for a, _ in d_all]), np.vstack([n for _, n in d_all])


def enforce_containment(structs, mhr_v, mhr_f, rounds=4, n_ctrl=500, verbose=True):
    """Supine CT vs standing MHR: the skin-only warp leaves bones almost touching the back. Push
    anything shallower than MIN_DEPTH (per class group) inward with a smooth volumetric field, so
    structures stay mutually coherent (the same field moves every structure)."""
    import trimesh
    mesh = trimesh.Trimesh(mhr_v, mhr_f, process=False)
    rng = np.random.default_rng(0)
    margin = np.concatenate([np.full(len(s["verts"]), MIN_DEPTH_MM.get(group_of(s["name"]), MIN_DEPTH_MM["default"]) / 1000.0)
                             for s in structs])
    off = np.cumsum([0] + [len(s["verts"]) for s in structs])
    for r in range(rounds):
        P = np.vstack([s["verts"] for s in structs])
        depth, inward = _skin_depth(mesh, P)
        viol = depth < margin
        if verbose:
            print(f"  containment round {r}: {100 * viol.mean():.1f}% of vertices shallower than their margin, "
                  f"{100 * (depth < 0).mean():.1f}% outside the skin")
        if viol.mean() < 0.002:
            break
        need = np.flatnonzero(viol)
        pick = rng.choice(need, min(n_ctrl, len(need)), replace=False)
        ok = np.flatnonzero(~viol)
        anchors = rng.choice(ok, min(n_ctrl, len(ok)), replace=False)
        ctrl = np.vstack([P[pick], P[anchors]])
        disp = np.vstack([inward[pick] * (margin[pick] - depth[pick])[:, None], np.zeros((len(anchors), 3))])
        f = RBFInterpolator(ctrl, disp, kernel="thin_plate_spline", smoothing=2e-3, degree=1)
        step = f(P)
        for k, s in enumerate(structs):
            s["verts"] = s["verts"] + step[off[k]:off[k + 1]]


# ---------------------------------------------------------------- rigged atlas writer (v2)
def load_meshes(prefix):
    """Read structure meshes (metres, MHR rest frame) from a previously written atlas (v1 or v2)."""
    meta = json.load(open(prefix + ".json"))
    blob = open(prefix + ".bin", "rb").read()
    out = []
    for e in meta["structures"]:
        n = e["n_verts"]
        it = np.uint16 if e["idx_type"] == "u16" else np.uint32
        out.append({"name": e["name"],
                    "verts": np.frombuffer(blob, np.float32, n * 3, e["pos"]).reshape(n, 3).astype(np.float64),
                    "faces": np.frombuffer(blob, it, e["n_tris"] * 3, e["idx"]).reshape(-1, 3).astype(np.int32),
                    "volume_ml": e.get("volume_ml", 0.0)})
    return out, meta


def write_rigged_atlas(structs, mhr_npz, out_prefix, extra_meta=None, verbose=True, enforce=True):
    """structs: [{name, verts(m), faces, volume_ml}] in the MHR rest frame. Applies limb pose
    normalisation, builds the biomechanical rig (anatomy.rig) and writes atlas.json/.bin (v2)."""
    from . import rig
    z = np.load(mhr_npz)
    mhr_v, mhr_j = z["verts"].astype(np.float64), z["joints"].astype(np.float64)
    head_fn = lambda v: (lambda c, r, f: (c / 1000.0, r))(*joint_head_center(v * 1000.0, -1))
    for st in structs:
        st["verts"] = rig.normalise_limb(st["name"], st["verts"], mhr_j, head_fn)
    if enforce:
        enforce_containment(structs, mhr_v, z["faces"], verbose=verbose)
    cents = {st["name"]: st["verts"].mean(0) for st in structs}
    lv = rig.spine_levels(cents, mhr_j)
    segs = rig.build_segments(mhr_v, mhr_j, lv)
    if verbose:
        print("segment skin support:", {k: len(v) for k, v in segs.items()}, " spine levels", {k: round(v, 3) for k, v in lv.items()})
    blob = bytearray()
    def put(arr):
        blob.extend(b"\0" * ((-len(blob)) % 4))
        start = len(blob); blob.extend(np.ascontiguousarray(arr).tobytes())
        return start
    seg_meta = []
    for name in rig.SEGMENTS:
        ids = segs[name].astype(np.uint16)
        seg_meta.append({"name": name, "n": int(len(ids)), "idx": put(ids), "rest": put(mhr_v[ids].astype(np.float32))})
    chain = rig.chain_names(cents)
    chain_y = [float(cents[n][1]) for n in chain]
    out = []
    for st in sorted(structs, key=lambda d: d["name"]):
        name, v, f = st["name"], st["verts"], st["faces"]
        g = group_of(name)
        kind, _ = rig.classify(name)
        e = {"name": name, "display": display_name(name), "group": g, "color": list(GROUP_COLOR[g]),
             "uncertainty_mm": UNCERTAINTY_MM[g], "n_verts": int(len(v)), "n_tris": int(len(f)),
             "volume_ml": float(st.get("volume_ml", 0.0)), "centroid": v.mean(0).round(5).tolist()}
        e["pos"] = put(v.astype(np.float32))
        e["idx_type"] = "u16" if len(v) < 65536 else "u32"
        e["idx"] = put(f.astype(np.uint16 if len(v) < 65536 else np.uint32))
        att = rig.attach_for(name, chain) if kind == "bone_spine" and not name.startswith("vertebrae_") else None
        if kind == "soft":
            idx, w = rig.soft_frame_weights(name, v, lv, chain, chain_y)
            e["bind"] = "soft"; e["inf_idx"] = put(idx); e["inf_w"] = put(w)
        elif kind == "bone_spine" and name.startswith("vertebrae_"):
            e["bind"] = "vertebra"; e["chain_index"] = chain.index(name); e["bone_w"] = rig.bone_weights(name, e["centroid"], lv)
        elif att:
            e["bind"] = "attach"; e["attach"] = att
        else:
            e["bind"] = "bone"; e["bone_w"] = rig.bone_weights(name, e["centroid"], lv)
        out.append(e)
    meta = {"version": 2, "frame": "mhr_rest_metres_y_down_front_-z", "n_skin_vertices": int(len(mhr_v)),
            "segments": seg_meta, "spine_levels": lv, "chain": chain,
            "disclaimer": "Atlas-based illustrative anatomy. Not patient-specific. Not for clinical use.",
            "rig": "rigid segments from skin; bones rigid; vertebrae blend neighbouring segments; soft tissue LBS",
            "structures": out}
    meta.update(extra_meta or {})
    json.dump(meta, open(out_prefix + ".json", "w"))
    open(out_prefix + ".bin", "wb").write(bytes(blob))
    if verbose:
        print(f"wrote {out_prefix}.json/.bin: {len(out)} structures ({sum(1 for e in out if e['bind']!='soft')} rigid bones), "
              f"{sum(e['n_tris'] for e in out)} tris, {len(blob) / 1e6:.2f} MB")
    return meta


# ---------------------------------------------------------------- driver
def build(subject_dir, mhr_npz, out_prefix, total_tris=140000, verbose=True):
    z = np.load(mhr_npz)
    mhr_v, mhr_j = z["verts"].astype(np.float64), z["joints"].astype(np.float64)
    ct, masks = load_subject(subject_dir)
    if verbose:
        print(f"{len(masks)} non-empty structures; CT {ct.shape} @ {ct.header.get_zooms()} mm")
    skin_v, skin_f = extract_skin(ct)
    warp, lm = register_to_mhr(skin_v, masks, ct.affine, mhr_v, mhr_j, verbose=verbose)
    vol = {n: m.sum() for n, (m, _) in masks.items()}
    wts = {n: max(v, 1) ** (2 / 3) for n, v in vol.items()}
    tot = sum(wts.values())
    structs = []
    for name in sorted(masks):
        mesh = mask_to_mesh(masks[name], ct.affine, max(120, int(total_tris * wts[name] / tot)))
        if mesh is None:
            continue
        v_mm, f = mesh
        structs.append({"name": name, "verts": warp(v_mm), "faces": f, "volume_ml": float(vol[name] * 1.5 ** 3 / 1000)})
    np.savez_compressed(out_prefix + "_meshes.npz", names=[s["name"] for s in structs],
                        **{f"v_{s['name']}": s["verts"] for s in structs}, **{f"f_{s['name']}": s["faces"] for s in structs},
                        **{f"m_{s['name']}": np.array(s["volume_ml"]) for s in structs})
    extra = {"source": {"dataset": "TotalSegmentator dataset v2.0.1 (Wasserthal et al.), CC-BY 4.0",
                        "subject": os.path.basename(os.path.abspath(subject_dir)),
                        "labels": "ground-truth 117-class masks (non-empty classes only)"},
             "landmarks_mm": {k: np.asarray(v).round(1).tolist() for k, v in lm.items()}}
    return write_rigged_atlas(structs, mhr_npz, out_prefix, extra, verbose)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_dir"); ap.add_argument("mhr_npz"); ap.add_argument("out_prefix")
    ap.add_argument("--tris", type=int, default=140000)
    ap.add_argument("--rebind", help="re-run only the rig/binding stage on an existing atlas prefix (v1 or v2)")
    a = ap.parse_args()
    if a.rebind:
        structs, old = load_meshes(a.rebind)
        write_rigged_atlas(structs, a.mhr_npz, a.out_prefix, {"source": old.get("source"), "landmarks_mm": old.get("landmarks_mm")})
    else:
        build(a.subject_dir, a.mhr_npz, a.out_prefix, a.tris)
