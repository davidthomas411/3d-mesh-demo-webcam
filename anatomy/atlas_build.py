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


# ---------------------------------------------------------------- binding
def bind_structure(verts_m, skin_v, skin_tree, max_support=192, n_sample=600, k=3, seed=0):
    """Skin support set for a structure: the skin vertices nearest to its vertices (weighted by
    how often they are hit). At runtime the structure is moved by the weighted rigid transform
    that carries these skin vertices from rest to the posed mesh."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(verts_m), min(n_sample, len(verts_m)), replace=False)
    _, idx = skin_tree.query(verts_m[pick], k=k)
    ids, cnt = np.unique(idx.ravel(), return_counts=True)
    order = np.argsort(-cnt)[:max_support]
    ids, cnt = ids[order], cnt[order].astype(np.float32)
    return ids.astype(np.uint16), cnt / cnt.sum()


# ---------------------------------------------------------------- driver
def build(subject_dir, mhr_npz, out_prefix, total_tris=140000, verbose=True):
    z = np.load(mhr_npz)
    mhr_v, mhr_j, mhr_f = z["verts"].astype(np.float64), z["joints"].astype(np.float64), z["faces"]
    ct, masks = load_subject(subject_dir)
    if verbose:
        print(f"{len(masks)} non-empty structures; CT {ct.shape} @ {ct.header.get_zooms()} mm")
    skin_v, skin_f = extract_skin(ct)
    warp, lm = register_to_mhr(skin_v, masks, ct.affine, mhr_v, mhr_j, verbose=verbose)
    # triangle budget proportional to sqrt(volume) (surface-ish), floor 120
    vol = {n: m.sum() for n, (m, _) in masks.items()}
    wts = {n: max(v, 1) ** (2 / 3) for n, v in vol.items()}
    tot = sum(wts.values())
    skin_tree = cKDTree(mhr_v)
    structs, blob, off = [], bytearray(), 0
    def put(arr):
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        pad = (-len(blob)) % 4
        blob.extend(b"\0" * pad)
        start = len(blob); blob.extend(b)
        return start, len(b)
    for name in sorted(masks):
        mesh = mask_to_mesh(masks[name], ct.affine, max(120, int(total_tris * wts[name] / tot)))
        if mesh is None:
            continue
        v_mm, f = mesh
        v = warp(v_mm).astype(np.float32)
        ids, w = bind_structure(v, mhr_v, skin_tree)
        g = group_of(name)
        e = {"name": name, "display": display_name(name), "group": g, "color": list(GROUP_COLOR[g]),
             "uncertainty_mm": UNCERTAINTY_MM[g], "n_verts": int(len(v)), "n_tris": int(len(f)),
             "support_n": int(len(ids)), "volume_ml": float(vol[name] * 1.5 ** 3 / 1000)}
        e["pos"] = put(v)[0]
        e["idx"] = put(f.astype(np.uint16 if len(v) < 65536 else np.uint32))[0]
        e["idx_type"] = "u16" if len(v) < 65536 else "u32"
        e["support_idx"] = put(ids)[0]
        e["support_w"] = put(w)[0]
        e["support_rest"] = put(mhr_v[ids].astype(np.float32))[0]
        structs.append(e)
    meta = {"version": 1, "frame": "mhr_rest_metres_y_down_front_-z", "n_skin_vertices": int(len(mhr_v)),
            "source": {"dataset": "TotalSegmentator dataset v2.0.1 (Wasserthal et al.), CC-BY 4.0",
                       "subject": os.path.basename(os.path.abspath(subject_dir)),
                       "labels": "ground-truth 117-class masks (non-empty classes only)"},
            "disclaimer": "Atlas-based illustrative anatomy. Not patient-specific. Not for clinical use.",
            "landmarks_mm": {k: np.asarray(v).round(1).tolist() for k, v in lm.items()},
            "structures": structs}
    json.dump(meta, open(out_prefix + ".json", "w"))
    open(out_prefix + ".bin", "wb").write(bytes(blob))
    if verbose:
        print(f"wrote {out_prefix}.json/.bin: {len(structs)} structures, "
              f"{sum(e['n_tris'] for e in structs)} tris, {len(blob) / 1e6:.2f} MB")
    return meta, warp, skin_v, skin_f


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_dir"); ap.add_argument("mhr_npz"); ap.add_argument("out_prefix")
    ap.add_argument("--tris", type=int, default=140000)
    a = ap.parse_args()
    build(a.subject_dir, a.mhr_npz, a.out_prefix, a.tris)
