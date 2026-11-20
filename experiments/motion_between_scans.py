"""Motion between two REAL scans of one patient: recover bone poses, interpolate, warp, and check against the real second scan.

Inputs: two CBCT volumes of the same 4D-Lung patient (NIfTI) and their TotalSegmentator label directories (from s1_surface_vs_bone.py).
Pipeline
  1. per-bone rigid pose A->B (trimmed ICP on each TotalSegmentator bone surface)           [bones are rigid]
  2. skin landmarks: anterior body surface of A matched to B after a global rigid ICP         [stands in for the camera]
  3. backward-warp field (Gaussian-weighted landmark motion) + EXACT rigid motion inside each bone
  4. motion in between = minimum-jerk interpolation of the bone poses and skin motion; frames are scan A's CT warped through them
Endpoint test (t = 1): warp A's anatomical LABELS into B and compare with B's own TotalSegmentator labels (Dice), against baselines:
  identity | one rigid from the surface (what SGRT does) | one rigid from the bones | classical diffeomorphic demons | per-bone model
Organs (lungs, heart, aorta, ...) are NOT used as landmarks, so their Dice is a held-out test.
Caveats: CBCT slab (15 cm), labels are themselves automatic and noisy, one patient, bone poses are oracle (from B's image).

usage: python experiments/motion_between_scans.py WORKDIR DAY_A DAY_B OUTDIR [--patient 100_HM10395] [--frames 12] [--no-demons]
"""
import sys, os, json, glob
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from s1_surface_vs_bone import kabsch, icp

BONE_PREFIX = ("vertebrae_T", "rib_", "sternum", "scapula_", "clavicula_")
ORGANS = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
          "heart", "aorta", "esophagus", "trachea", "spinal_cord"]


class Scan:
    def __init__(self, nii, segdir):
        self.img = sitk.ReadImage(nii)
        self.hu = sitk.GetArrayFromImage(self.img).astype(np.float32)               # (z, y, x)
        self.o = np.array(self.img.GetOrigin()); self.s = np.array(self.img.GetSpacing())
        self.D = np.array(self.img.GetDirection()).reshape(3, 3)
        self.Dinv = np.linalg.inv(self.D)
        self.labels = {}
        for f in sorted(glob.glob(f"{segdir}/*.nii.gz")):
            n = os.path.basename(f)[:-7]
            if n.startswith(BONE_PREFIX) or n in ORGANS:
                m = sitk.GetArrayFromImage(sitk.ReadImage(f)) > 0
                if m.sum() > 50:
                    self.labels[n] = m

    def mm(self, ijk):                                  # ijk columns = (x index, y index, z index)
        return self.o + (np.asarray(ijk, float) * self.s) @ self.D.T

    def vox(self, p):
        return ((np.asarray(p, float) - self.o) @ self.Dinv.T) / self.s

    def sample(self, vol, p, order=1, cval=0.0):
        i = self.vox(p).T
        return ndi.map_coordinates(vol, [i[2], i[1], i[0]], order=order, mode="constant", cval=cval)

    def surface(self, mask, n=3000, seed=0):
        e = mask & ~ndi.binary_erosion(mask)
        z, y, x = np.nonzero(e)
        idx = np.stack([x, y, z], 1)
        if len(idx) > n:
            idx = idx[np.random.default_rng(seed).choice(len(idx), n, replace=False)]
        return self.mm(idx)

    def grid(self):
        z, y, x = np.indices(self.hu.shape)
        return self.mm(np.stack([x.ravel(), y.ravel(), z.ravel()], 1))


def body_surface(sc, ends=2):
    from skimage.filters import threshold_otsu
    a = sc.hu
    m = a > max(threshold_otsu(np.clip(a, -1500, 500)), -600)
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    for k in range(m.shape[0]):
        m[k] = ndi.binary_fill_holes(m[k])
    e = m & ~ndi.binary_erosion(m)
    e[:ends] = False; e[-ends:] = False                 # the slab's cut faces are not skin
    z, y, x = np.nonzero(e)
    pts = sc.mm(np.stack([x, y, z], 1))
    cy = pts[:, 1].mean() if len(pts) else 0.0
    return pts, m


def smoothstep5(t):                                     # minimum-jerk time law
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def rot_pow(R, s):                                      # R**s along the geodesic
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(Rot.from_matrix(R).as_rotvec() * s).as_matrix()


def bone_poses(A, B):
    """Rigid pose of each bone A->B: x' = R x + t, from trimmed ICP on the mask surfaces. Returns {name: (R, t, rms, c)}."""
    out = {}
    for n in sorted(set(A.labels) & set(B.labels)):
        if not n.startswith(BONE_PREFIX) or A.labels[n].sum() < 600 or B.labels[n].sum() < 600:
            continue
        pa, pb = A.surface(A.labels[n]), B.surface(B.labels[n])
        t0 = pb.mean(0) - pa.mean(0)
        R, t = icp(pa + t0, pb, iters=40, trim=0.8)
        # compose with the initial translation: x -> R (x + t0) + t
        tt = R @ t0 + t
        d = cKDTree(pb).query(pa @ R.T + tt)[0]
        out[n] = (R, tt, float(np.sqrt(np.mean(np.sort(d)[: int(0.8 * len(d))] ** 2))), pa.mean(0))
    return out


def landmark_set(A, B, poses, skin_a, skin_b, s=1.0, gate=15.0):
    """Landmarks X (in A) and their positions Y at motion fraction s (0..1): bones move rigidly, skin follows the matched surface."""
    X, Y, owner = [], [], []
    names = list(poses)
    for k, n in enumerate(names):
        R, t, _, c = poses[n]
        pa = A.surface(A.labels[n], n=1500)
        Rs = rot_pow(R, s)
        full = pa @ R.T + t                               # where the bone ends up at s = 1
        cfull = c @ R.T + t
        ys = (pa - c) @ Rs.T + c + s * (cfull - c)        # rotate about the bone's own centre, translate along the path
        X.append(pa); Y.append(ys); owner.append(np.full(len(pa), k))
    Rg, tg = icp(skin_a, skin_b, iters=40, trim=0.8)
    pa = skin_a[::7]
    ya = pa @ Rg.T + tg
    d, i = cKDTree(skin_b).query(ya)
    keep = d < gate
    X.append(pa[keep]); Y.append(pa[keep] + s * (skin_b[i[keep]] - pa[keep])); owner.append(np.full(keep.sum(), -1))
    return np.vstack(X), np.vstack(Y), np.concatenate(owner), names, (Rg, tg)


def warp_field(Xl, Yl, owner, poses, names, grid_pts, B, s, sigma=10.0, k=24, coarse=8.0):
    """Backward map for output positions g (in B's frame): source = g - u_hat(g), u_hat Gaussian-weighted from the landmark motion;
    inside a posed bone the exact rigid inverse is used instead."""
    U = Yl - Xl
    tree = cKDTree(Yl)
    lo = grid_pts.min(0); hi = grid_pts.max(0)
    axes = [np.arange(lo[a], hi[a] + coarse, coarse) for a in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, i = tree.query(G, k=k, workers=-1)
    w = np.exp(-(d ** 2 - d[:, :1] ** 2) / (2 * sigma ** 2))
    uh = ((w[..., None] * U[i]).sum(1) / w.sum(1, keepdims=True)).reshape(tuple(len(a) for a in axes) + (3,))
    c = ((grid_pts - lo) / coarse).T
    u = np.stack([ndi.map_coordinates(uh[..., a], c, order=1, mode="nearest") for a in range(3)], 1)
    src = grid_pts - u
    # exact rigid motion inside bones
    bone_id = np.zeros(B.hu.shape, np.uint8)
    for k_, n in enumerate(names):
        bone_id[B.labels[n]] = k_ + 1
    return src, bone_id


def apply_bone_override(src, grid_pts, poses, names, A, s, Yl, owner):
    """Replace smooth source positions by the exact rigid inverse where the posed bone is (verified against A's own label)."""
    bl = Yl[owner >= 0]; own = owner[owner >= 0]
    tree = cKDTree(bl)
    near = np.zeros(len(grid_pts), bool)
    d, j = tree.query(grid_pts, k=1, workers=-1)
    cand = np.flatnonzero(d < 8.0)
    for k_ in np.unique(own[j[cand]]):
        sel = cand[own[j[cand]] == k_]
        n = names[k_]
        R, t, _, c = poses[n]
        Rs = rot_pow(R, s); cfull = c @ R.T + t
        shift = c + s * (cfull - c)                        # centre of the posed bone
        xb = (grid_pts[sel] - shift) @ Rs + c              # inverse of  y = Rs (x - c) + shift
        inside = A.sample(A.labels[n].astype(np.float32), xb, order=0, cval=0) > 0
        src[sel[inside]] = xb[inside]
    return src


def warp_volume(A, B, poses, names, skin_a, skin_b, s, grid_pts):
    Xl, Yl, owner, _, glob_ = landmark_set(A, B, poses, skin_a, skin_b, s)
    src, _ = warp_field(Xl, Yl, owner, poses, names, grid_pts, B, s)
    src = apply_bone_override(src, grid_pts, poses, names, A, s, Yl, owner)
    return src


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def eval_labels(name, warped, B):
    rows = {}
    for n in B.labels:
        if n in warped:
            rows[n] = dice(warped[n], B.labels[n])
    bones = [v for k, v in rows.items() if k.startswith(BONE_PREFIX)]
    orgs = [v for k, v in rows.items() if k in ORGANS]
    return {"bones": float(np.nanmean(bones)) if bones else None, "organs": float(np.nanmean(orgs)) if orgs else None, "n_bones": len(bones), "n_organs": len(orgs)}


def rigid_warp_labels(A, B, R, t, grid):
    """Warp A's labels into B with one rigid transform x' = R x + t (backward: x = R^T (g - t))."""
    src = (grid - t) @ R
    return {n: A.sample(m.astype(np.float32), src, order=0, cval=0).reshape(B.hu.shape) > 0 for n, m in A.labels.items()}


def demons_labels(A, B):
    fixed = sitk.Cast(B.img, sitk.sitkFloat32); moving = sitk.Cast(A.img, sitk.sitkFloat32)
    moving = sitk.HistogramMatching(moving, fixed)
    d = sitk.FastSymmetricForcesDemonsRegistrationFilter(); d.SetNumberOfIterations(60); d.SetStandardDeviations(2.0)
    f2 = sitk.Shrink(fixed, [2, 2, 1]); m2 = sitk.Shrink(moving, [2, 2, 1])
    field = sitk.Cast(d.Execute(f2, m2), sitk.sitkVectorFloat64)
    tx = sitk.DisplacementFieldTransform(field)
    out = {}
    for n, m in A.labels.items():
        li = sitk.GetImageFromArray(m.astype(np.uint8)); li.CopyInformation(A.img)
        r = sitk.Resample(li, f2, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        out[n] = sitk.GetArrayFromImage(sitk.Resample(r, B.img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    work, day_a, day_b, outdir = a[:4]
    patient = a[a.index("--patient") + 1] if "--patient" in a else "100_HM10395"
    n_frames = int(a[a.index("--frames") + 1]) if "--frames" in a else 12
    os.makedirs(outdir, exist_ok=True)
    A = Scan(f"{work}/{patient}_{day_a}.nii.gz", f"{work}/seg_{day_a}")
    B = Scan(f"{work}/{patient}_{day_b}.nii.gz", f"{work}/seg_{day_b}")
    skin_a, _ = body_surface(A); skin_b, _ = body_surface(B)
    ya = skin_a[:, 1].mean(); yb = skin_b[:, 1].mean()
    skin_a = skin_a[skin_a[:, 1] < ya]; skin_b = skin_b[skin_b[:, 1] < yb]          # anterior surface only (what a camera sees)
    poses = bone_poses(A, B); names = list(poses)
    print(f"{len(names)} bones with poses; ICP rms (mm) median {np.median([p[2] for p in poses.values()]):.2f}; "
          f"rotation (deg) median {np.median([np.degrees(np.linalg.norm(__import__('scipy.spatial.transform', fromlist=['Rotation']).Rotation.from_matrix(p[0]).as_rotvec())) for p in poses.values()]):.2f}", flush=True)
    grid = B.grid()
    res = {}
    # identity
    ident = {n: m for n, m in A.labels.items()}
    res["identity (no motion)"] = eval_labels("id", ident, B)
    # one rigid from surface (SGRT-like)
    Rg, tg = icp(skin_a, skin_b, iters=40, trim=0.8)
    res["one rigid from skin surface (SGRT-like)"] = eval_labels("surf", rigid_warp_labels(A, B, Rg, tg, grid), B)
    # one rigid from bones
    PA = np.vstack([A.surface(A.labels[n], 800) for n in names]); PB = np.vstack([B.surface(B.labels[n], 800) for n in names])
    Rb, tb = icp(PA, PB, iters=40, trim=0.8)
    res["one rigid from bones (best single rigid)"] = eval_labels("bone", rigid_warp_labels(A, B, Rb, tb, grid), B)
    # classical diffeomorphic demons
    if "--no-demons" not in a:
        res["classical diffeomorphic demons"] = eval_labels("demons", demons_labels(A, B), B)
    # per-bone rig model (oracle bone poses) at t = 1
    src = warp_volume(A, B, poses, names, skin_a, skin_b, 1.0, grid)
    warped = {n: A.sample(m.astype(np.float32), src, order=0, cval=0).reshape(B.hu.shape) > 0 for n, m in A.labels.items()}
    res["per-bone rig model (oracle bone poses)"] = eval_labels("model", warped, B)
    for k, v in res.items():
        print(f"{k:46s} bone Dice {v['bones']:.3f} ({v['n_bones']} bones) | held-out organ Dice {v['organs']:.3f} ({v['n_organs']} organs)", flush=True)
    json.dump(res, open(f"{outdir}/endpoint_metrics.json", "w"), indent=1)
    # frames: A warped through the interpolated motion
    lo_, hi_ = np.percentile(A.hu, [2, 99.7])
    mid_z = B.hu.shape[0] // 2
    for fi, t in enumerate(np.linspace(0, 1, n_frames)):
        s_ = float(smoothstep5(t))
        src = warp_volume(A, B, poses, names, skin_a, skin_b, s_, grid)
        vol = A.sample(A.hu, src, order=1, cval=float(lo_)).reshape(B.hu.shape)
        np.save(f"{outdir}/frame_{fi:02d}.npy", vol.astype(np.float16))
        print("frame", fi, "t", round(float(t), 2), "s", round(s_, 3), flush=True)
    np.save(f"{outdir}/A.npy", A.hu.astype(np.float16)); np.save(f"{outdir}/B.npy", B.hu.astype(np.float16))
    np.save(f"{outdir}/Bbones.npy", np.any([B.labels[n] for n in names], axis=0))
